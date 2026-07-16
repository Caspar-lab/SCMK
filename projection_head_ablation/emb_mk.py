"""
emb_mk.py — New variant: learned embeddings + gauss_med_kernels -> OC-SVM.
=========================================================================
Goal: verify that the multi-kernel scheme *centered on gauss_med_kernels* is
meaningful by applying it to the LEARNED embeddings (not the raw features), and
report it side by side with Full and Raw-MK.

  Full            : learned projection -> dual-signal OC-SVM head            [proposed]
  Raw-MK          : gauss_med_kernels empirical map on RAW features -> dual head
  Emb-MK (dual)   : gauss_med_kernels empirical map on the LEARNED embedding Z,
                    then the IDENTICAL dual head  (== Raw-MK with X -> Z)
  Emb-MK (precomp): average those Gaussian maps into ONE combined kernel and feed
                    a single kernel='precomputed' OC-SVM

Z = concat of the K L2-normalised projection-head embeddings (the canonical learned
representation). Emb-MK changes ONLY the representation source vs Raw-MK (raw X ->
learned Z), holding gauss_med_kernels + the scoring head fixed. So:
  - Raw-MK  vs  Emb-MK  isolates whether the kernels become meaningful *after* the
    learned projection (i.e. is the gauss_med_kernels-centered MK meaningful on Z?).
  - dual  vs  precomp   checks the conclusion is robust to how the OC-SVM consumes it.

Bandwidths t_k come from gauss_med_kernels estimated on Z's train-normal geometry, so
the 5 multi-scale Gaussians are matched to the learned space (not the raw space).

Config per dataset = the (latent_dim, lambda) that the AUC experiments selected
(hybrid_semi_best.csv), seeds {0,1,2} on the same 50/50 splits as everything else.

Run (torch311):  conda run -n torch311 python emb_mk.py
Outputs: results/embmk_variant.csv, prints the side-by-side table.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd
import torch  # before sklearn (Windows OpenMP load order)

HERE = os.path.dirname(os.path.abspath(__file__))
HS   = r'C:/OD/Shihao/5/Granular-CMK/hybrid_score'
sys.path.insert(0, HERE)       # for raw_mk
sys.path.insert(0, HS)
from run_hybrid_score_semi import (split_indices, ensemble_auc_split, extract_components,
                                   gauss_med_kernels, load_data, NU_CANDIDATES)
from CMK_OCSVM import TRAIN_CFG as BASE_CFG
from CMK_OCSVM_scatter import train_cmk_scatter
from raw_mk import raw_mk_representation                     # reuse Raw-MK machinery on Z
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, pairwise_distances

ROOT = r'C:/OD/Shihao/5'
DATA = r'C:/OD/Shihao/datasets'
OUT  = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- references for the side-by-side table (same 20 datasets / seeds as everything else)
_master = pd.read_csv(ROOT + '/result/hybrid_score_semi/master_compare_v2.csv')
ORDER   = _master['stem'].tolist()                           # Table-2 order
NAME    = dict(zip(_master['stem'], _master['name']))
FULL    = {r['stem']: (float(r['SCMK_mean']), float(r['SCMK_std'])) for _, r in _master.iterrows()}
_rawcsv = pd.read_csv(os.path.join(OUT, 'projhead_ablation.csv')).set_index('dataset')
RAWMK   = {st: (float(_rawcsv.loc[st, 'RawMK_mean']), float(_rawcsv.loc[st, 'RawMK_std'])) for st in ORDER}
_best   = pd.read_csv(ROOT + '/result/hybrid_score_semi/hybrid_semi_best.csv').set_index('dataset')
CFG     = {st: (int(_best.loc[st, 'best_dim']), float(_best.loc[st, 'best_lambda'])) for st in ORDER}


def learned_Z(X, tr, dim, lam):
    """Train CMKNet on train-normals -> concat L2-normalised head embeddings = (N, K*d)."""
    kernels = gauss_med_kernels(X[tr])                       # training kernels (on raw X)
    cfg     = {**BASE_CFG, 'lambda_scatter': lam}
    model   = train_cmk_scatter(X[tr], np.zeros(len(tr), int), kernels, dim, device, cfg)
    H_norm_per, _ = extract_components(model, X, device)
    return np.concatenate(H_norm_per, axis=1)                # canonical learned embedding Z


def embmk_precomp_auc(Z, tr, te, yt, kernels_Z):
    """Average the 5 Gaussian maps on Z into one combined kernel -> precomputed OC-SVM."""
    d2   = pairwise_distances(Z, Z[tr], metric='sqeuclidean')            # (N, |tr|)
    comb = np.mean([np.exp(-d2 / (2.0 * k[2]['t'] ** 2)) for k in kernels_Z], axis=0)
    Kfit, Kscr = comb[tr], comb[te]                          # (|tr|,|tr|), (|te|,|tr|)
    best = -1.0
    for nu in NU_CANDIDATES:
        try:
            clf  = OneClassSVM(kernel='precomputed', nu=nu).fit(Kfit)
            best = max(best, roc_auc_score(yt, -clf.decision_function(Kscr)))
        except Exception:
            pass
    return best


if __name__ == '__main__':
    print(f'device={device} | seeds={SEEDS} | datasets={len(ORDER)}', flush=True)
    rows = []
    for stem in ORDER:
        X, y, _ = load_data(os.path.join(DATA, stem + '.mat'))
        yb = (y != 0).astype(int)
        dim, lam = CFG[stem]
        dual, prec = [], []
        for sd in SEEDS:
            tr, te = split_indices(yb, sd)
            yt = yb[te]
            Z  = learned_Z(X, tr, dim, lam)                  # learned embeddings for this split
            kernels_Z = gauss_med_kernels(Z[tr])             # 5 multi-scale Gaussians on Z
            H_norm_per, H_norms = raw_mk_representation(Z, tr, kernels_Z)   # == Raw-MK on Z
            dual.append(ensemble_auc_split(H_norm_per, H_norms, tr, te, yt, NU_CANDIDATES))
            prec.append(embmk_precomp_auc(Z, tr, te, yt, kernels_Z))
        fm, fs = FULL[stem]; rm, rs = RAWMK[stem]
        rec = dict(dataset=stem, name=NAME[stem],
                   Full_mean=fm, Full_std=fs, RawMK_mean=rm, RawMK_std=rs,
                   EmbMKdual_mean=np.mean(dual), EmbMKdual_std=np.std(dual, ddof=1),
                   EmbMKprecomp_mean=np.mean(prec), EmbMKprecomp_std=np.std(prec, ddof=1))
        rec['d_dual_vs_raw'] = rec['EmbMKdual_mean'] - rm
        rec['d_dual_vs_full'] = rec['EmbMKdual_mean'] - fm
        rows.append(rec)
        print(f"{NAME[stem]:<13} Full={fm:.4f}  Raw-MK={rm:.4f}  "
              f"Emb-MK(dual)={rec['EmbMKdual_mean']:.4f}  "
              f"Emb-MK(precomp)={rec['EmbMKprecomp_mean']:.4f}  "
              f"(vs Raw {rec['d_dual_vs_raw']:+.4f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'embmk_variant.csv'), index=False)

    COLS = [('Full', 'Full'), ('Raw-MK', 'RawMK'),
            ('Emb-MK(dual)', 'EmbMKdual'), ('Emb-MK(precomp)', 'EmbMKprecomp')]
    print('\n=== mean AUC over 20 datasets ===')
    means = {}
    for label, key in COLS:
        m = df[f'{key}_mean'].mean(); means[label] = m
        print(f'  {label:<16} {m:.4f}')
    print(f"\n  Raw-MK -> Emb-MK(dual) lift: {means['Emb-MK(dual)'] - means['Raw-MK']:+.4f}")
    print(f"  Emb-MK(dual)  vs Full:       {means['Emb-MK(dual)'] - means['Full']:+.4f}")
    print(f"  best config on N datasets: Emb-MK(dual)>=Raw-MK on "
          f"{int((df['EmbMKdual_mean'] >= df['RawMK_mean']).sum())}/{len(df)}")
    print('\nsaved: results/embmk_variant.csv')
