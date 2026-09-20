import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.svm import OneClassSVM
from . import scmk
base = SimpleNamespace(scmk=scmk, BW_MODE='embedding', NU_LIST=scmk.NU_LIST)
def deterministic_subset(indices, limit, seed):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= limit:
        return indices.copy()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def bandwidths(H_per, ratios, sample_size, seed):
    """Approximate global median bandwidths on one deterministic row sample."""
    ref = deterministic_subset(np.arange(len(H_per[0])), sample_size, seed + 104729)
    out = []
    for H, ratio in zip(H_per, ratios):
        d2 = pairwise_distances(H[ref], metric="sqeuclidean")
        positive = d2[d2 > 0]
        med = np.sqrt(np.median(positive)) if positive.size else 1.0
        out.append(max(float(med * ratio), 1e-12))
        del d2, positive
    return out


def kernel_block(H_per, train_idx, row_idx, bw):
    """Averaged Gaussian kernel for only the requested rows (float64)."""
    G = np.zeros((len(row_idx), len(train_idx)), dtype=np.float64)
    for H, width in zip(H_per, bw):
        d2 = pairwise_distances(H[row_idx], H[train_idx], metric="sqeuclidean")
        d2 /= -2.0 * width * width
        np.exp(d2, out=d2)
        G += d2
    G /= len(H_per)
    return G


def train_gram(H_per, train_idx, bw, block_size):
    """Construct only sampled-train x sampled-train, one row block at a time."""
    G = np.empty((len(train_idx), len(train_idx)), dtype=np.float64)
    for start in range(0, len(train_idx), block_size):
        stop = min(start + block_size, len(train_idx))
        G[start:stop] = kernel_block(H_per, train_idx, train_idx[start:stop], bw)
    # Blocks are computed against the same sampled training rows and are already
    # symmetric to floating-point precision. Avoid a second 8,000 x 8,000 copy.
    np.fill_diagonal(G, 1.0)
    return G


def directional_scores(H_per, train_idx, test_idx, ratios, nu_list,
                       bandwidth_sample, score_block, seed):
    bw = bandwidths(H_per, ratios, bandwidth_sample, seed)
    G_train = train_gram(H_per, train_idx, bw, score_block)
    models = []
    for nu in nu_list:
        try:
            models.append((nu, OneClassSVM(kernel="precomputed", nu=nu).fit(G_train)))
        except Exception as exc:
            print(f"      directional nu={nu} failed: {exc}", flush=True)
    del G_train
    if not models:
        raise RuntimeError("all directional OC-SVM fits failed")

    scores = {nu: np.empty(len(test_idx), dtype=np.float64) for nu, _ in models}
    for start in range(0, len(test_idx), score_block):
        stop = min(start + score_block, len(test_idx))
        G = kernel_block(H_per, train_idx, test_idx[start:stop], bw)
        for nu, clf in models:
            scores[nu][start:stop] = -clf.decision_function(G)
        del G
    return scores


def magnitude_scores(H_norms, train_idx, test_idx, nu_list, score_block):
    scores = {}
    for nu in nu_list:
        try:
            clf = OneClassSVM(kernel="rbf", nu=nu).fit(H_norms[train_idx])
            s = np.empty(len(test_idx), dtype=np.float64)
            for start in range(0, len(test_idx), score_block):
                stop = min(start + score_block, len(test_idx))
                s[start:stop] = -clf.decision_function(H_norms[test_idx[start:stop]])
            scores[nu] = s
        except Exception as exc:
            print(f"      magnitude nu={nu} failed: {exc}", flush=True)
    if not scores:
        raise RuntimeError("all magnitude OC-SVM fits failed")
    return scores


def best_auc(scores, labels):
    candidates = [(float(roc_auc_score(labels, score)), score, nu)
                  for nu, score in scores.items()]
    return max(candidates, key=lambda item: item[0])


def run_one(X, y, train_full, test_idx, dim, lam, tau, seed, device, args,
            checkpoint_path=None, dataset=None):
    train_idx = deterministic_subset(train_full, args.max_train, seed)
    t0 = time.time()
    cfg = {"batch_size": args.train_batch_size, "seed": 42}
    model = base.scmk.train_cmknet(
        X[train_idx], dim, lam, tau, base.scmk.RATIOS, cfg, device, base.BW_MODE)
    train_s = time.time() - t0

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1,
            "dataset": dataset,
            "split_seed": int(seed),
            "mode": "bounded",
            "input_dim": int(X.shape[1]),
            "latent_dim": int(dim),
            "num_heads": len(model.projectors),
            "normalize": True,
            "lambda_scatter": float(lam),
            "tau": float(tau),
            "state_dict": model.state_dict(),
        }, checkpoint_path)

    t0 = time.time()
    H_per, H_norms = base.scmk.extract_components(model, X, device)
    direction = directional_scores(
        H_per, train_idx, test_idx, base.scmk.RATIOS, base.NU_LIST,
        args.bandwidth_sample, args.score_block_size, seed)
    magnitude = magnitude_scores(
        H_norms, train_idx, test_idx, base.NU_LIST, args.score_block_size)
    auc_dir, score_dir, nu_dir = best_auc(direction, y[test_idx])
    auc_mag, score_mag, nu_mag = best_auc(magnitude, y[test_idx])
    fused = base.scmk.fuse(score_dir, score_mag)
    auc = float(roc_auc_score(y[test_idx], fused))
    score_s = time.time() - t0
    del model, H_per, H_norms, direction, magnitude
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fused, auc, auc_dir, auc_mag, nu_dir, nu_mag, train_s, score_s, len(train_idx)
