"""
regen_roc_dfno_fulldata.py — Regenerate the per-dataset ROC comparison figures with the
CURRENT 8-method set, using DFNO scored TRANSDUCTIVELY on the FULL dataset
(C:/OD/Shihao/Experimental_results/DFNO_results), consistent with KFGOD.

Per-method evaluation protocol (unchanged except DFNO):
  * SCMK        : seed-2 semi split, live CMK training, scored on the held-out test set.
  * KFGOD, DFNO : transductive, scored on the FULL dataset (opt_out_scores).
  * Disent-AD/DeepSVDD/LMKAD : seed-2 split, cached per-sample test scores (res_single).
  * ICL/NeuTraLAD           : seed-2 split, cached per-sample test scores (CSV).

Mirrors plot_roc_compare.ipynb. Run with the torch311 python. Writes:
  result/hybrid_score_semi/roc_compare/{stem}_ROC.pdf   (one panel per dataset)
  result/hybrid_score_semi/roc_auc_regen.csv            (per-dataset per-method AUC)
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd, scipy.io as sio
import torch  # import torch before matplotlib/sklearn to fix Windows OpenMP load order

NB_DIR = 'C:/OD/Shihao/5/Granular-CMK/hybrid_score'
sys.path.insert(0, NB_DIR)
# Use the self-contained run_hybrid_score_semi (no matplotlib import) — importing the
# old run_hybrid_score pulls in matplotlib, which crashes torch training via OpenMP.
from run_hybrid_score_semi import (split_indices, extract_components, _best_ocsvm_scores,
                                   _minmax, gauss_med_kernels, load_data, _BASE_CFG, NU_CANDIDATES)
from CMK_OCSVM_scatter import train_cmk_scatter

from sklearn.metrics import roc_curve, roc_auc_score
# NOTE: matplotlib is imported LATER (after all torch computation) to avoid a
# Windows OpenMP (libiomp) crash when matplotlib is loaded alongside torch+libsvm.

DATA_ROOT = 'C:/OD/Shihao/datasets'
ER        = 'C:/OD/Shihao/Experimental_results'
KFGOD_DIR = 'C:/OD/Shihao/KFGOD-main/results'
DFNO_DIR  = ER + '/DFNO_results'                       # full-data transductive DFNO
ICL_DIR   = 'C:/OD/Shihao/5/ICL and NeuTraLAD/results_split/scores/seed2'
SEMI_DIR  = 'C:/OD/Shihao/5/result/hybrid_score_semi'
OUT_DIR   = os.path.join(SEMI_DIR, 'roc_compare')
SEL_CSV   = os.path.join(SEMI_DIR, 'selection_scatter_semi_seed2.csv')
ALL_CSV   = os.path.join(SEMI_DIR, 'hybrid_semi_all.csv')
SPLIT_SEED = 2
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# plotting order + styling (matches the previous figures)
METHODS = ['KFGOD', 'Disent_AD', 'DeepSVDD', 'DFNO', 'LMKAD', 'ICL', 'NeuTraLAD', 'scatter']
LABEL = {'KFGOD': 'KFGOD', 'Disent_AD': 'Disent-AD', 'DeepSVDD': 'DeepSVDD', 'DFNO': 'DFNO',
         'LMKAD': 'LMKAD', 'ICL': 'ICL', 'NeuTraLAD': 'NeuTraLAD', 'scatter': 'SCMK'}
STYLE = {'KFGOD': ('#FF8C00', 'x', 1.1), 'Disent_AD': ('#BA55D3', '^', 1.1),
         'DeepSVDD': ('#20B2AA', '<', 1.1), 'DFNO': ('#00BFFF', 'v', 1.1),
         'LMKAD': ('#6B8E23', '>', 1.1), 'ICL': ('#8B4513', 's', 1.1),
         'NeuTraLAD': ('#FF69B4', 'p', 1.1), 'scatter': ('#DC143C', 'o', 1.5)}
DISP = {'vertebral': 'Vertebral', 'thyroid': 'Thyroid', 'wbc_malignant_39_variant1': 'WBC',
        'glass': 'Glass', 'ecoli': 'Ecoli', 'pageblocks_1_258_variant1': 'PageBlocks',
        'wine': 'Wine', 'cardio': 'Cardio', 'cardiotocography_2and3_33_variant1': 'Cardiotoco.',
        'tic_tac_toe_negative_69_variant1': 'TicTacToe-69',
        'tic_tac_toe_negative_12_variant1': 'TicTacToe-12', 'wpbc_variant1': 'WPBC',
        'ionosphere_b_24_variant1': 'Ionosphere', 'zoo_variant1': 'Zoo',
        'sick_sick_72_variant1': 'Sick-72', 'autos_variant1': 'Autos',
        'annealing_variant1': 'Annealing', 'lymphography': 'Lympho.',
        'bands_band_6_variant1': 'Bands-6', 'audiology_variant1': 'Audiology'}

sel = pd.read_csv(SEL_CSV); DATASETS = list(sel['dataset'])
_all = pd.read_csv(ALL_CSV); _s2 = _all[_all['split_seed'] == SPLIT_SEED]
BEST_CFG = {}
for st in DATASETS:
    sub = _s2[_s2['dataset'] == st]
    br = sub.loc[sub['auc'].idxmax()]
    BEST_CFG[st] = (int(br['latent_dim']), float(br['lambda_scatter']))


def load_full_y(stem):
    t = sio.loadmat(os.path.join(DATA_ROOT, stem + '.mat'))['trandata'].astype(float)
    return (t[:, -1] != 0).astype(int)


def full_data_scores(path, y):
    if not os.path.exists(path):
        return None
    s = np.asarray(sio.loadmat(path)['opt_out_scores'])[:, 0].ravel()
    return s if len(s) == len(y) else None


def seed2_mat_curve(path, field='res_single'):
    r = sio.loadmat(path)[field][0, 0]
    s = np.asarray(r['opt_scores'], float).ravel()
    y = np.asarray(r['labels'], float).ravel().astype(int)
    return y, s


def seed2_csv_curve(path):
    d = pd.read_csv(path)
    return d['label'].values.astype(int), d['anomaly_score'].values.astype(float)


def scatter_test_scores(stem, train_idx, test_idx):
    X, y, _ = load_data(os.path.join(DATA_ROOT, stem + '.mat'))
    dim, lam = BEST_CFG[stem]
    kernels = gauss_med_kernels(X[train_idx])
    cfg = {**_BASE_CFG, 'lambda_scatter': lam}
    model = train_cmk_scatter(X[train_idx], np.zeros(len(train_idx), int), kernels, dim, device, cfg)
    H_norm_per, H_norms = extract_components(model, X, device)
    H_dir = np.concatenate(H_norm_per, axis=1)
    y_test = (y[test_idx] != 0).astype(int)
    _, _, s_dir = _best_ocsvm_scores(H_dir[test_idx], y_test, H_dir[train_idx], 'linear', NU_CANDIDATES)
    _, _, s_nrm = _best_ocsvm_scores(H_norms[test_idx], y_test, H_norms[train_idx], 'rbf', NU_CANDIDATES)
    if s_dir is None: return y_test, _minmax(s_nrm)
    if s_nrm is None: return y_test, _minmax(s_dir)
    return y_test, np.maximum(_minmax(s_dir), _minmax(s_nrm))


def method_curve(m, stem, y_full, train_idx, test_idx):
    """return (y_eval, scores) for method m on its native evaluation set, or None."""
    if m == 'scatter':
        return scatter_test_scores(stem, train_idx, test_idx)
    if m == 'KFGOD':
        s = full_data_scores(os.path.join(KFGOD_DIR, stem, f'{stem}_KFGOD.mat'), y_full)
        return (y_full, s) if s is not None else None
    if m == 'DFNO':
        s = full_data_scores(os.path.join(DFNO_DIR, stem, f'{stem}_DFNO.mat'), y_full)
        return (y_full, s) if s is not None else None
    if m == 'Disent_AD':
        return seed2_mat_curve(os.path.join(ER, 'Disent_AD_split_seed2', f'{stem}_DisentAD.mat'))
    if m == 'DeepSVDD':
        return seed2_mat_curve(os.path.join(ER, 'DeepSVDD_split_seed2', f'{stem}_DeepSVDD.mat'))
    if m == 'LMKAD':
        return seed2_mat_curve(os.path.join(ER, 'LMKAD_gauss_split_seed2', f'{stem}_LMKAD.mat'))
    if m == 'ICL':
        return seed2_csv_curve(os.path.join(ICL_DIR, 'ICL', f'{stem}_scores.csv'))
    if m == 'NeuTraLAD':
        return seed2_csv_curve(os.path.join(ICL_DIR, 'NeuTraL', f'{stem}_scores.csv'))
    return None


print(f'setup ok | datasets={len(DATASETS)} | device={device}', flush=True)
# ---- Phase 1: compute all ROC curves (torch + sklearn; NO matplotlib loaded) ----
ALL_CURVES = {}   # stem -> {method: (fpr, tpr, auc)}
rows = []
for stem in DATASETS:
    print(f'>> {stem}', flush=True)
    y_full = load_full_y(stem)
    train_idx, test_idx = split_indices(y_full, SPLIT_SEED)
    curves = {}
    rec = {'dataset': stem}
    for m in METHODS:
        try:
            out = method_curve(m, stem, y_full, train_idx, test_idx)
            if out is None: continue
            y_eval, sc = out
            fpr, tpr, _ = roc_curve(y_eval, sc)
            auc = roc_auc_score(y_eval, sc)
            curves[m] = (fpr, tpr, auc); rec[LABEL[m]] = round(auc, 4)
        except Exception as e:
            print('ERR', m, stem, ':', e)
    ALL_CURVES[stem] = curves
    rows.append(rec)
    print(f'{stem:<36} ' + '  '.join(f'{LABEL[m]}={curves[m][2]:.3f}' for m in METHODS if m in curves), flush=True)

pd.DataFrame(rows).to_csv(os.path.join(SEMI_DIR, 'roc_auc_regen.csv'), index=False)
print('compute done; loading matplotlib for plotting...', flush=True)

# ---- Phase 2: plot (matplotlib only; torch work already finished) ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'Times New Roman'

for stem in DATASETS:
    curves = ALL_CURVES.get(stem, {})
    if not curves: continue
    fig = plt.figure(figsize=(4, 3), dpi=150)
    plt.plot([0, 1], [0, 1], color='gray', lw=0.5, linestyle=(0, (8, 8)))
    for m in METHODS:
        if m not in curves: continue
        fpr, tpr, auc = curves[m]; col, mk, lw = STYLE[m]
        plt.plot(fpr, tpr, label=f'{LABEL[m]} ({auc:.3f})', color=col, marker=mk,
                 markevery=max(len(fpr) // 8, 1), markersize=3, lw=lw)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 20, 40, 60, 80, 100], fontsize=7)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 20, 40, 60, 80, 100], fontsize=7)
    plt.xlim(-0.05, 1.02); plt.ylim(-0.05, 1.02); plt.grid(True)
    plt.title(DISP.get(stem, stem[:30]), fontsize=9)
    plt.legend(prop={'size': 5}, ncol=2, loc='lower right')
    fig.patch.set_facecolor('white')
    plt.savefig(os.path.join(OUT_DIR, stem + '_ROC.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.close()

print('\nsaved ROC pdfs to', OUT_DIR)
print('saved AUC table to', os.path.join(SEMI_DIR, 'roc_auc_regen.csv'))
