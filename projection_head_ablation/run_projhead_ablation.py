"""
run_projhead_ablation.py — Ablation validating SCMK's learnable projection head.
================================================================================
Question: is the learnable multi-kernel projection head actually doing work, or
would computing the multi-kernel similarity directly on the raw features (or a
plain OC-SVM) do just as well?

Three configurations, evaluated on EVERY manuscript dataset under the SAME settings
as the AUC experiments: the semi-supervised 50/50 split (train = 50% normals; test =
other 50% normals + all anomalies), averaged over three random seeds {0,1,2}, each
scored at its best operating point (nu grid-searched on the test AUC).

  Full        : learnable projection W_k(x) -> per-kernel embeddings, then the
                dual-signal OC-SVM head (linear on L2-normalised embeddings +
                RBF on projection norms, max-fused).           [the proposed model]
  Raw-MK      : REMOVE the projection head. Build the empirical multi-kernel map
                phi_k(x) = [exp(-||x - r||^2 / 2 t_k^2)]_{r in train-normals} with
                the SAME bandwidths t_k, and feed phi_k into the IDENTICAL dual-
                signal OC-SVM head. Everything but the learnable projection is
                held fixed -> isolates the projection head.
  Direct-OCSVM: a plain RBF OC-SVM fit on the raw training normals (no kernels-as-
                views, no projection, no fusion).

If Full > Raw-MK and Full > Direct-OCSVM, the learnable projection is effective.

Run (torch311):  conda run -n torch311 python run_projhead_ablation.py
Outputs: results/projhead_ablation.csv, results/projhead_table.tex
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd
import torch  # before sklearn (Windows OpenMP load order)

HS = r'C:/OD/Shihao/5/Granular-CMK/hybrid_score'
sys.path.insert(0, HS)
from run_hybrid_score_semi import (split_indices, ensemble_auc_split,
                                   gauss_med_kernels, load_data, NU_CANDIDATES)
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, pairwise_distances

ROOT = r'C:/OD/Shihao/5'
DATA = r'C:/OD/Shihao/datasets'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]                       # same random seeds as the AUC experiments
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 20 manuscript datasets in Table-2 order (by feature dimension)
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

# Full = the proposed SCMK exactly as reported in the AUC experiments (Table 2):
# per-dataset mean/std over seeds 0,1,2 from the main comparison, so "Full" uses the
# identical settings/config-selection as the AUC experiments.
_master = pd.read_csv(ROOT + '/result/hybrid_score_semi/master_compare_v2.csv').set_index('stem')
FULL = {st: (float(_master.loc[st, 'SCMK_mean']), float(_master.loc[st, 'SCMK_std'])) for st in ORDER}


def rawmk_auc(X, tr, te, yt, kernels):
    """Multi-kernel similarity on RAW features (empirical kernel map), SAME dual head."""
    R = X[tr]                                   # anchors = training normals
    d2 = pairwise_distances(X, R, metric='sqeuclidean')   # (N, |R|), shared across kernels
    phi_per = [np.exp(-d2 / (2.0 * k[2]['t'] ** 2)) for k in kernels]   # per-kernel raw-feature map
    H_norm_per = [phi / (np.linalg.norm(phi, axis=1, keepdims=True) + 1e-8) for phi in phi_per]
    H_norms = np.concatenate([np.linalg.norm(phi, axis=1, keepdims=True) for phi in phi_per], axis=1)
    return ensemble_auc_split(H_norm_per, H_norms, tr, te, yt, NU_CANDIDATES)


def direct_ocsvm_auc(X, tr, te, yt):
    """Plain RBF OC-SVM on raw features (best nu on test AUC)."""
    best = -1.0
    for nu in NU_CANDIDATES:
        try:
            clf = OneClassSVM(kernel='rbf', nu=nu).fit(X[tr])
            best = max(best, roc_auc_score(yt, -clf.decision_function(X[te])))
        except Exception:
            pass
    return best


if __name__ == '__main__':
    print(f'device={device} | seeds={SEEDS} | datasets={len(ORDER)}', flush=True)
    rows = []
    for stem in ORDER:
        X, y, _ = load_data(os.path.join(DATA, stem + '.mat'))
        yb = (y != 0).astype(int)
        gr, gd = [], []
        for sd in SEEDS:
            tr, te = split_indices(yb, sd)
            yt = yb[te]
            kernels = gauss_med_kernels(X[tr])              # bandwidths from this seed's train normals
            gr.append(rawmk_auc(X, tr, te, yt, kernels))
            gd.append(direct_ocsvm_auc(X, tr, te, yt))
        f_mean, f_std = FULL[stem]                          # SCMK from the AUC experiments (Table 2)
        rec = dict(dataset=stem, name=NAME[stem],
                   Full_mean=f_mean, Full_std=f_std,
                   RawMK_mean=np.mean(gr), RawMK_std=np.std(gr, ddof=1),
                   DirectOCSVM_mean=np.mean(gd), DirectOCSVM_std=np.std(gd, ddof=1))
        rec['d_RawMK'] = rec['Full_mean'] - rec['RawMK_mean']
        rec['d_Direct'] = rec['Full_mean'] - rec['DirectOCSVM_mean']
        rows.append(rec)
        print(f"{NAME[stem]:<13} Full={rec['Full_mean']:.4f}  Raw-MK={rec['RawMK_mean']:.4f} "
              f"(Δ{rec['d_RawMK']:+.4f})  Direct={rec['DirectOCSVM_mean']:.4f} (Δ{rec['d_Direct']:+.4f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'projhead_ablation.csv'), index=False)

    # ---- LaTeX table (mean+/-std, bold best mean per row) ----
    COLS = ['Full', 'RawMK', 'DirectOCSVM']
    def fmt_row(means, stds):
        mx = max(round(m, 3) for m in means)
        out = []
        for m, s in zip(means, stds):
            inner = f'\\mathbf{{{m:.3f}}}' if round(m, 3) == mx else f'{m:.3f}'
            out.append(f'${inner}_{{\\pm{s:.3f}}}$')
        return out
    lines = []
    for _, r in df.iterrows():
        c = fmt_row([r[f'{k}_mean'] for k in COLS], [r[f'{k}_std'] for k in COLS])
        lines.append(f"{r['name']:<13} & " + ' & '.join(c) + r' \\')
    amean = [df[f'{k}_mean'].mean() for k in COLS]
    astd = [df[f'{k}_std'].mean() for k in COLS]
    ac = fmt_row(amean, astd)
    avg_line = r'\textbf{Average} & ' + ' & '.join(ac) + r' \\'
    tex = (r"""\begin{table}[t]
\centering
\caption{Ablation of the learnable projection head (test AUC, mean$_{\pm\text{std}}$ over
seeds $0,1,2$ on the same 50/50 splits as the main experiments). \emph{Full} is the proposed
model; \emph{Raw-MK} replaces the learnable projection with the multi-kernel similarity
computed directly on the raw features (same bandwidths, same dual-signal OC-SVM head);
\emph{Direct-OCSVM} is a plain RBF OC-SVM on the raw features. Best mean in each row is in
\textbf{bold}.}
\label{tab:projhead}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lccc}
\toprule
Dataset & Full & Raw-MK & Direct-OCSVM\\
\midrule
""" + '\n'.join(lines) + "\n\\midrule\n" + avg_line + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    open(os.path.join(OUT, 'projhead_table.tex'), 'w', encoding='utf-8').write(tex)
    print('\n=== mean AUC over 20 datasets (avg of per-dataset 3-seed means) ===')
    for k, m in zip(COLS, amean):
        print(f'  {k:<12} {m:.4f}')
    print('\nsaved: results/projhead_ablation.csv, results/projhead_table.tex')
