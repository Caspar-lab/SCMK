"""
gmean_supplement.py — Maximum G-mean of SCMK and the seven comparison detectors on the
20 manuscript datasets, in the same layout as Table 2 (AUC).

G-mean = sqrt(TPR * TNR) = sqrt(sensitivity * specificity); we report the maximum over the
ROC operating points (threshold-free, like AUC), computed from each method's scores.

Protocol (identical to the ROC/AUC pipeline):
  * SCMK, Disent-AD, DeepSVDD, LMKAD, ICL, NeuTraLAD : seed-{0,1,2} semi-splits, scored on
    the held-out test set -> mean_{+/-std} G-mean over the three seeds.
  * KFGOD, DFNO : transductive on the FULL dataset -> single G-mean (no seeds).

Run with the torch311 env (conda run -n torch311). Outputs:
  result/hybrid_score_semi/gmean_all.csv      per-dataset per-method mean/std
  Granular-CMK/hybrid_score/gmean_table.tex   Table-2-style LaTeX table (body + Average)
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd, scipy.io as sio
import torch  # before sklearn/matplotlib (Windows OpenMP load order)

NB_DIR = 'C:/OD/Shihao/5/Granular-CMK/hybrid_score'
sys.path.insert(0, NB_DIR)
from run_hybrid_score_semi import (split_indices, extract_components, _best_ocsvm_scores,
                                   _minmax, gauss_med_kernels, load_data, _BASE_CFG, NU_CANDIDATES)
from CMK_OCSVM_scatter import train_cmk_scatter
from sklearn.metrics import roc_curve

DATA_ROOT = 'C:/OD/Shihao/datasets'
ER        = 'C:/OD/Shihao/Experimental_results'
KFGOD_DIR = 'C:/OD/Shihao/KFGOD-main/results'
DFNO_DIR  = ER + '/DFNO_results'
IC_ROOT   = 'C:/OD/Shihao/5/ICL and NeuTraLAD/results_split/scores'
SEMI_DIR  = 'C:/OD/Shihao/5/result/hybrid_score_semi'
SEL_CSV   = os.path.join(SEMI_DIR, 'selection_scatter_semi_seed2.csv')
ALL_CSV   = os.path.join(SEMI_DIR, 'hybrid_semi_all.csv')
SEEDS = [0, 1, 2]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

METHODS = ['KFGOD', 'Disent_AD', 'DeepSVDD', 'DFNO', 'LMKAD', 'ICL', 'NeuTraLAD', 'scatter']
SPLIT_METHODS = ['Disent_AD', 'DeepSVDD', 'LMKAD', 'ICL', 'NeuTraLAD', 'scatter']  # mean+/-std
SINGLE_METHODS = ['KFGOD', 'DFNO']                                                 # full-data
LABEL = {'KFGOD': 'KFGOD', 'Disent_AD': 'Disent', 'DeepSVDD': 'DeepSVDD', 'DFNO': 'DFNO',
         'LMKAD': 'LMKAD', 'ICL': 'ICL', 'NeuTraLAD': 'NeuTraLAD', 'scatter': 'SCMK'}
# per-seed .mat dirs / suffixes for the deep+kernel one-class baselines
MAT = {'Disent_AD': ('Disent_AD_split_seed', '_DisentAD.mat'),
       'DeepSVDD':  ('DeepSVDD_split_seed', '_DeepSVDD.mat'),
       'LMKAD':     ('LMKAD_gauss_split_seed', '_LMKAD.mat')}
CSVM = {'ICL': 'ICL', 'NeuTraLAD': 'NeuTraL'}

ORDER = ['vertebral', 'thyroid', 'wbc_malignant_39_variant1', 'glass', 'ecoli',
         'pageblocks_1_258_variant1', 'wine', 'cardio', 'cardiotocography_2and3_33_variant1',
         'tic_tac_toe_negative_12_variant1', 'tic_tac_toe_negative_69_variant1', 'wpbc_variant1',
         'ionosphere_b_24_variant1', 'zoo_variant1', 'sick_sick_72_variant1', 'autos_variant1',
         'annealing_variant1', 'lymphography', 'bands_band_6_variant1', 'audiology_variant1']
NAME = {'vertebral': 'Vertebral', 'thyroid': 'Thyroid', 'wbc_malignant_39_variant1': 'WBC',
        'glass': 'Glass', 'ecoli': 'Ecoli', 'pageblocks_1_258_variant1': 'PageBlocks',
        'wine': 'Wine', 'cardio': 'Cardio', 'cardiotocography_2and3_33_variant1': 'Cardiotoco.',
        'tic_tac_toe_negative_12_variant1': 'TicTacToe-12', 'tic_tac_toe_negative_69_variant1': 'TicTacToe-69',
        'wpbc_variant1': 'WPBC', 'ionosphere_b_24_variant1': 'Ionosphere', 'zoo_variant1': 'Zoo',
        'sick_sick_72_variant1': 'Sick-72', 'autos_variant1': 'Autos', 'annealing_variant1': 'Annealing',
        'lymphography': 'Lympho.', 'bands_band_6_variant1': 'Bands-6', 'audiology_variant1': 'Audiology'}

# per-dataset best (dim, lambda) for SCMK (seed-2 best config, as in the ROC notebook)
_all = pd.read_csv(ALL_CSV); _s2 = _all[_all['split_seed'] == 2]
BEST_CFG = {st: (int(_s2[_s2.dataset == st].loc[_s2[_s2.dataset == st].auc.idxmax(), 'latent_dim']),
                 float(_s2[_s2.dataset == st].loc[_s2[_s2.dataset == st].auc.idxmax(), 'lambda_scatter']))
            for st in ORDER}


def gmean_max(y, s):
    """Maximum G-mean = max_t sqrt(TPR_t * (1 - FPR_t)) over ROC thresholds."""
    fpr, tpr, _ = roc_curve(y, s)
    g = np.sqrt(np.clip(tpr, 0, 1) * np.clip(1.0 - fpr, 0, 1))
    return float(np.nanmax(g))


def load_full_y(stem):
    t = sio.loadmat(os.path.join(DATA_ROOT, stem + '.mat'))['trandata'].astype(float)
    return (t[:, -1] != 0).astype(int)


def full_scores(path, y):
    s = np.asarray(sio.loadmat(path)['opt_out_scores'])[:, 0].ravel()
    return s if len(s) == len(y) else None


def seed_scores(method, seed, stem):
    """(y_test, scores) on the seed's held-out test set for a split method."""
    if method in MAT:
        d, suf = MAT[method]
        r = sio.loadmat(f'{ER}/{d}{seed}/{stem}{suf}')['res_single'][0, 0]
        return np.asarray(r['labels'], float).ravel().astype(int), np.asarray(r['opt_scores'], float).ravel()
    if method in CSVM:
        df = pd.read_csv(f'{IC_ROOT}/seed{seed}/{CSVM[method]}/{stem}_scores.csv')
        return df['label'].values.astype(int), df['anomaly_score'].values.astype(float)
    if method == 'scatter':
        X, y, _ = load_data(os.path.join(DATA_ROOT, stem + '.mat'))
        tr, te = split_indices((y != 0).astype(int), seed)
        dim, lam = BEST_CFG[stem]
        kernels = gauss_med_kernels(X[tr]); cfg = {**_BASE_CFG, 'lambda_scatter': lam}
        model = train_cmk_scatter(X[tr], np.zeros(len(tr), int), kernels, dim, device, cfg)
        Hnp, Hn = extract_components(model, X, device)
        Hd = np.concatenate(Hnp, axis=1); yt = (y[te] != 0).astype(int)
        _, _, sdir = _best_ocsvm_scores(Hd[te], yt, Hd[tr], 'linear', NU_CANDIDATES)
        _, _, snrm = _best_ocsvm_scores(Hn[te], yt, Hn[tr], 'rbf', NU_CANDIDATES)
        if sdir is None: return yt, _minmax(snrm)
        if snrm is None: return yt, _minmax(sdir)
        return yt, np.maximum(_minmax(sdir), _minmax(snrm))
    raise ValueError(method)


# ---- compute ----
rows = []
for stem in ORDER:
    y_full = load_full_y(stem)
    rec = {'stem': stem, 'name': NAME[stem]}
    rec['KFGOD_mean'] = gmean_max(y_full, full_scores(f'{KFGOD_DIR}/{stem}/{stem}_KFGOD.mat', y_full))
    rec['DFNO_mean'] = gmean_max(y_full, full_scores(f'{DFNO_DIR}/{stem}/{stem}_DFNO.mat', y_full))
    for m in SPLIT_METHODS:
        gs = [gmean_max(*seed_scores(m, sd, stem)) for sd in SEEDS]
        rec[f'{m}_mean'] = float(np.mean(gs)); rec[f'{m}_std'] = float(np.std(gs, ddof=1))
    rows.append(rec)
    print(f"{stem:<36} " + "  ".join(f"{LABEL[m]}={rec[f'{m}_mean']:.3f}" for m in METHODS), flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SEMI_DIR, 'gmean_all.csv'), index=False)


# ---- LaTeX table (mirror tab:comparison) ----
def fmt(mean, std, bold):
    m = f'{mean:.3f}'; inner = f'\\mathbf{{{m}}}' if bold else m
    return f'${inner}_{{\\pm{std:.3f}}}$' if std is not None else f'${inner}$'


lines = []
for _, r in df.iterrows():
    means = [r[f'{m}_mean'] for m in METHODS]
    rmax = max(round(v, 3) for v in means)
    cells = []
    for m in METHODS:
        mv = r[f'{m}_mean']; std = r.get(f'{m}_std') if m in SPLIT_METHODS else None
        cells.append(fmt(mv, std, round(mv, 3) == rmax))
    lines.append(f"{r['name']:<12} & " + ' & '.join(cells) + r' \\')
# Average row
avg = {m: df[f'{m}_mean'].mean() for m in METHODS}
amax = max(round(a, 3) for a in avg.values())
acells = []
for m in METHODS:
    std = df[f'{m}_std'].mean() if m in SPLIT_METHODS else None
    acells.append(fmt(avg[m], std, round(avg[m], 3) == amax))
avg_line = r'\textbf{Average} & ' + ' & '.join(acells) + r' \\'

body = '\n'.join(lines) + '\n\\midrule\n' + avg_line
table = r"""\begin{table*}[t]
\centering
\caption{Maximum G-mean ($\sqrt{\mathrm{TPR}\cdot\mathrm{TNR}}$) of SCMK against seven recent
detectors on twenty datasets, as a threshold-based complement to the AUC in
Table~\ref{tab:comparison}. The one-class methods (SCMK, Disent-AD, DeepSVDD, LMKAD, ICL,
NeuTraLAD) are reported as mean$_{\pm\text{std}}$ over three random splits (seeds $0,1,2$);
KFGOD and DFNO are transductive on the full set (single-valued). Best mean in each row is in
\textbf{bold}.}
\label{tab:gmean}
\scriptsize
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccccc c}
\toprule
Dataset & KFGOD & Disent & DeepSVDD & DFNO & LMKAD & ICL & NeuTraLAD & SCMK\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
}
\end{table*}
"""
open(os.path.join(NB_DIR, 'gmean_table.tex'), 'w', encoding='utf-8').write(table)
print('\n=== mean G-mean per method ===')
for m in METHODS:
    print(f'  {LABEL[m]:<10} {avg[m]:.4f}')
print('\nsaved: result/hybrid_score_semi/gmean_all.csv, Granular-CMK/hybrid_score/gmean_table.tex')
