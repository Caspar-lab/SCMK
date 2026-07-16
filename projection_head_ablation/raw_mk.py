"""
raw_mk.py — Standalone Raw-MK variant, for inspecting the workflow.
===================================================================
Raw-MK is the ablation row that REMOVES the learnable projection head and replaces
the learned representation with a multi-kernel similarity computed directly on the
RAW features — while holding the kernels (gauss_med_kernels bandwidths) and the entire
dual-signal OC-SVM scoring head identical to `Full`. Any gap vs Full is therefore
attributable to the projection head.

This file isolates that one variant end-to-end (no Full / Direct-OCSVM around it) so
the pipeline is easy to read. The logic is identical to `rawmk_auc` inside
`run_projhead_ablation.py`; running this prints each step's shape for inspection.

    Workflow
    --------
    (1) split            : semi-supervised 50/50 -> train = normals, test = rest+anoms
    (2) bandwidths       : gauss_med_kernels(X[train]) -> 5 Gaussian scales t_k
    (3) empirical map    : phi_k(x) = [exp(-||x - r||^2 / 2 t_k^2)]_{r in train-normals}
                           i.e. each sample -> its RBF similarity to every training normal,
                           one (N, |train|) feature block per kernel scale
    (4) dual signal      : direction = L2-normalise each block  (angular)
                           magnitude = row-norm of each block   (how "close" overall)
    (5) score            : identical dual-signal OC-SVM head (linear on directions +
                           RBF on magnitudes, max-fused), best nu on test AUC

Run (torch311):  conda run -n torch311 python raw_mk.py [dataset_stem]
                 default dataset_stem = zoo_variant1 (a datapoint where Raw-MK collapses)
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np
import torch  # before sklearn (Windows OpenMP load order)

HS = r'C:/OD/Shihao/5/Granular-CMK/hybrid_score'
sys.path.insert(0, HS)
from run_hybrid_score_semi import (split_indices, ensemble_auc_split,
                                   gauss_med_kernels, load_data, NU_CANDIDATES)
from sklearn.metrics import pairwise_distances

DATA  = r'C:/OD/Shihao/datasets'
SEEDS = [0, 1, 2]        # same seeds as the AUC experiments


def raw_mk_representation(X, train_idx, kernels, verbose=False):
    """
    Build the Raw-MK dual-signal representation on raw features.

    Returns (H_norm_per, H_norms):
      H_norm_per : list of K arrays (N, |train|) — each kernel block, L2-normalised
                   per row -> the DIRECTION signal (angular pattern of similarities).
      H_norms    : (N, K) — per-block row norm -> the MAGNITUDE signal (overall closeness
                   to the training-normal cloud at each kernel scale).
    """
    R  = X[train_idx]                                       # anchors = training normals
    d2 = pairwise_distances(X, R, metric='sqeuclidean')     # (N, |R|), shared across kernels
    if verbose:
        print(f"  anchors R = X[train]        shape {R.shape}")
        print(f"  d2 = ||x - r||^2            shape {d2.shape}  (rows=all samples, cols=train normals)")

    # (3) empirical Gaussian kernel map per scale t_k  ->  phi_k in [0,1]^(N x |R|)
    phi_per = [np.exp(-d2 / (2.0 * k[2]['t'] ** 2)) for k in kernels]

    # (4) split into direction (L2-normalised) and magnitude (row norm) signals
    H_norm_per = [phi / (np.linalg.norm(phi, axis=1, keepdims=True) + 1e-8) for phi in phi_per]
    H_norms    = np.concatenate(
        [np.linalg.norm(phi, axis=1, keepdims=True) for phi in phi_per], axis=1)   # (N, K)
    if verbose:
        for k, (kn, phi) in enumerate(zip(kernels, phi_per)):
            print(f"  kernel {k}: t={kn[2]['t']:.4g}  phi_k {phi.shape}  "
                  f"-> direction block {H_norm_per[k].shape}")
        print(f"  H_norms (magnitude signal)  shape {H_norms.shape}  (one column per kernel)")
    return H_norm_per, H_norms


def raw_mk_auc(X, train_idx, test_idx, y_test, kernels, verbose=False):
    """Raw-MK representation -> identical dual-signal OC-SVM head -> best test AUC."""
    H_norm_per, H_norms = raw_mk_representation(X, train_idx, kernels, verbose)
    return ensemble_auc_split(H_norm_per, H_norms, train_idx, test_idx, y_test, NU_CANDIDATES)


if __name__ == '__main__':
    stem = sys.argv[1] if len(sys.argv) > 1 else 'zoo_variant1'
    X, y, _ = load_data(os.path.join(DATA, stem + '.mat'))
    yb = (y != 0).astype(int)
    print(f'dataset={stem}  N={X.shape[0]}  D={X.shape[1]}  anomalies={int(yb.sum())}\n')

    aucs = []
    for i, sd in enumerate(SEEDS):
        tr, te = split_indices(yb, sd)
        yt = yb[te]
        kernels = gauss_med_kernels(X[tr])                  # bandwidths from this seed's train normals
        verbose = (i == 0)                                  # print the workflow once
        if verbose:
            print(f'--- seed {sd}: workflow (train normals={len(tr)}, test={len(te)}) ---')
        auc = raw_mk_auc(X, tr, te, yt, kernels, verbose=verbose)
        if verbose:
            print(f'--- seed {sd}: Raw-MK test AUC = {auc:.4f} ---\n')
        aucs.append(auc)

    print(f'Raw-MK AUC over seeds {SEEDS}: '
          f'{np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}')
