"""
scmk.py -- Scatter-regularized Contrastive Multi-Kernel learning (SCMK)
======================================================================
Self-contained, single-file implementation of the SCMK one-class anomaly
detector, consolidated from the original research code
(Granular-CMK/CMK_OCSVM.py, CMK_OCSVM_scatter.py,
 scmk_experiments/run_hybrid_score_semi_debug.py, run_scmk_emb_precom_amp_grid.py).

Pipeline
--------
1. Multi-scale Gaussian kernel bank: K=5 relative scales r = (0.1,0.5,1,2,5).
2. CMKNet: K bias-free linear heads W_k : R^D -> R^d, outputs L2-normalised
   onto the unit sphere (one head per kernel scale).
3. Cross-kernel InfoNCE (contrastive) loss with temperature tau and
   embedding-calibrated Gaussian bandwidths (kernel index = contrastive view).
4. Multi-kernel scatter regularizer L_scat = -(1/K) sum_k ||mu_k||^2 (compactness).
5. Dual-signal detector on the frozen heads:
     - directional branch: precomputed multi-scale Gaussian Gram OC-SVM on the
       unit-norm embeddings (angular),
     - magnitude branch: RBF OC-SVM on the per-head projection norms (radial),
     fused by a per-sample max of the two min-max normalised scores.

Two usage modes
---------------
* SCMK class (fit on normals, score new samples with a fixed nu) -- for building on.
* run_scmk() / reproduce_dataset() -- paper-faithful reproduction (oracle nu chosen
  by best test AUC, matching the manuscript's evaluation protocol).

Author handover: see HANDOVER.md.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as scio
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.svm import OneClassSVM

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ======================================================================
# Config
# ======================================================================
RATIOS = (0.1, 0.5, 1.0, 2.0, 5.0)          # multi-scale kernel bank (K=5)
NU_LIST = (0.01, 0.05, 0.1, 0.2)            # OC-SVM nu candidates (oracle grid)
TRAIN_CFG = dict(epochs=100, batch_size=512, lr=0.01, normalize=True, seed=42)

# Directories searched by locate() for `<name>.mat`; edit for your machine.
from pathlib import Path
DATA_DIRS = [str(Path(__file__).resolve().parents[1] / 'data' / 'raw')]


def default_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ======================================================================
# Data loading / preprocessing
# ======================================================================
def _detect_columns(X):
    """Heuristic: a column is nominal iff it has <=20 distinct integer values."""
    nominal, numeric = [], []
    for c in range(X.shape[1]):
        col = X[:, c]
        if len(np.unique(col)) <= 20 and np.all(col == col.astype(int)):
            nominal.append(c)
        else:
            numeric.append(c)
    return nominal, numeric


def _preprocess(X_raw):
    """One-hot encode nominal columns, min-max scale numeric columns to [0,1]."""
    nominal_cols, numeric_cols = _detect_columns(X_raw)
    parts = []
    if nominal_cols:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        parts.append(enc.fit_transform(X_raw[:, nominal_cols]).astype(np.float32))
    if numeric_cols:
        parts.append(MinMaxScaler().fit_transform(X_raw[:, numeric_cols]).astype(np.float32))
    assert parts, 'no valid feature columns found'
    return np.hstack(parts)


def load_data(path):
    """Load a `.mat` (var 'trandata', last column = label) or `.csv` (no header,
    last column = label). Returns (X, y) with y in {0=normal, 1=anomaly}."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        import pandas as pd
        data = pd.read_csv(path, header=None).values.astype(np.float64)
        y = (data[:, -1] != 0).astype(int)
    else:
        import io
        with open(path, 'rb') as fh:
            d = scio.loadmat(io.BytesIO(fh.read()))
        data = d['trandata'].astype(np.float64)
        y = (data[:, -1] != 0).astype(int)
    X = _preprocess(data[:, :-1])
    return X, y


def locate(name):
    """Resolve a dataset stem to a `.mat` path via DATA_DIRS."""
    if name.lower().endswith('.mat') and os.path.exists(name):
        return name
    for d in DATA_DIRS:
        p = os.path.join(d, name + '.mat')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'dataset not found: {name}')


def split_indices(y, seed, frac=0.5):
    """Semi-supervised split: `frac` of normals -> train; the rest of normals plus
    ALL anomalies -> test. Returns (train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    normal = np.where(y == 0)[0]
    anom = np.where(y != 0)[0]
    perm = rng.permutation(normal)
    k = int(round(len(perm) * frac))
    return perm[:k], np.concatenate([perm[k:], anom])


# ======================================================================
# Kernel bank
# ======================================================================
def gauss_med_kernels(X_normal, ratios=RATIOS):
    """Median-heuristic Gaussian bank on normal samples: bandwidth t_k = med * r_k.
    Returns list of (name, 'Gaussian', {'t': t_k})."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_normal), min(500, len(X_normal)), replace=False)
    med = np.median(pairwise_distances(X_normal[idx], metric='euclidean'))
    return [(f'G-{med * r:.3g}', 'Gaussian', {'t': max(med * r, 1e-3)}) for r in ratios]


# ======================================================================
# Network + kernels
# ======================================================================
class CMKNet(nn.Module):
    """K independent bias-free linear heads W_k: R^D -> R^d; L2-normalised output."""

    def __init__(self, input_dim, latent_dim, n_kernels, normalize=True):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(input_dim, latent_dim, bias=False) for _ in range(n_kernels)
        ])
        self.normalize = normalize

    def forward(self, x):
        hs = [p(x) for p in self.projectors]
        if self.normalize:
            hs = [F.normalize(h, dim=1) for h in hs]
        return hs


def _eu_dist2(a, b):
    aa = (a * a).sum(1, keepdim=True)
    bb = (b * b).sum(1, keepdim=True)
    return (aa + bb.T - 2 * a @ b.T).clamp(min=0)


def _gaussian_gram(Fm, t):
    return torch.exp(-_eu_dist2(Fm, Fm) / (2 * t ** 2))


# ======================================================================
# Losses
# ======================================================================
def _median_dist(Fm, max_n=256):
    n = Fm.shape[0]
    if n > max_n:
        Fm = Fm[torch.randperm(n, device=Fm.device)[:max_n]]
    return torch.pdist(Fm.detach()).median().clamp(min=1e-3)


def cross_kernel_loss(hs, ratios, raw_bw, tau=1.0, bw_mode='embedding'):
    """Cross-kernel InfoNCE over all C(K,2) head pairs.
    bw_mode='embedding': bandwidth = median(embedding pdist) * ratio (recommended).
    bw_mode='raw': use fixed raw-feature bandwidths in `raw_bw`."""
    K, B = len(hs), hs[0].shape[0]
    dev = hs[0].device
    mask = torch.eye(B, device=dev).repeat(2, 2) * (1 - torch.eye(2 * B, device=dev))
    logits_mask = 1 - torch.eye(2 * B, device=dev)
    total, n = 0.0, 0
    for k in range(K):
        for l in range(k + 1, K):
            Fkl = torch.cat([hs[k], hs[l]], dim=0)
            if bw_mode == 'embedding':
                med = _median_dist(Fkl)
                tk, tl = med * ratios[k], med * ratios[l]
            else:
                tk, tl = raw_bw[k], raw_bw[l]
            K_avg = (_gaussian_gram(Fkl, float(tk)) + _gaussian_gram(Fkl, float(tl))) / 2
            logits = torch.exp(K_avg / tau)
            log_prob = torch.log(logits) - torch.log((logits * logits_mask).sum(1, keepdim=True))
            loss = -(mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
            total += loss.mean()
            n += 1
    return total / max(n, 1)


def scatter_loss(hs):
    """Multi-kernel scatter regularizer: L_scat = -(1/K) sum_k ||mu_k||^2."""
    total = sum(h.mean(dim=0).pow(2).sum() for h in hs)
    return -total / len(hs)


# ======================================================================
# Training
# ======================================================================
def train_cmknet(X_tr, latent_dim, lambda_scatter, tau, ratios=RATIOS,
                 cfg=None, device=None, bw_mode='embedding', verbose=False):
    """Train CMKNet on normal training features. Returns the fitted model."""
    cfg = {**TRAIN_CFG, **(cfg or {})}
    device = device or default_device()
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    N, D = X_tr.shape
    raw_bw = [k[2]['t'] for k in gauss_med_kernels(X_tr, ratios)]  # used only if bw_mode='raw'
    model = CMKNet(D, latent_dim, len(ratios), cfg['normalize']).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(N)
        ec = es = 0.0
        nb = 0
        for i in range(0, N, cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            if len(idx) < 4:
                continue
            hs = model(X_t[idx].to(device))
            Lc = cross_kernel_loss(hs, ratios, raw_bw, tau=tau, bw_mode=bw_mode)
            Ls = scatter_loss(hs) if lambda_scatter > 0 else torch.zeros((), device=device)
            loss = Lc + lambda_scatter * Ls
            opt.zero_grad()
            loss.backward()
            opt.step()
            ec += Lc.item()
            es += float(Ls.item())
            nb += 1
        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == cfg['epochs']):
            print(f'  epoch {epoch:>3d}  L_cross={ec / nb:.4f}  L_scatter={es / nb:+.4f}')
    return model


# ======================================================================
# Dual-signal detector
# ======================================================================
@torch.no_grad()
def extract_components(model, X, device=None, batch_size=2048):
    """Return (H_norm_per, H_norms): list of K unit-norm embedding arrays (N x d),
    and an N x K matrix of per-head raw projection norms ||W_k x||."""
    device = device or default_device()
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    per = [[] for _ in model.projectors]
    for i in range(0, len(X), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        for k, h in enumerate([p(xb) for p in model.projectors]):
            per[k].append(h.cpu().numpy())
    H_raw_per = [np.concatenate(p, axis=0) for p in per]
    H_norm_per = [h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8) for h in H_raw_per]
    H_norms = np.concatenate([np.linalg.norm(h, axis=1, keepdims=True) for h in H_raw_per], axis=1)
    return H_norm_per, H_norms


def _minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(s)


def _multi_kernel_gram(H_norm_per, ratios):
    """Averaged multi-scale Gaussian Gram over the unit-norm per-head embeddings."""
    K, N = len(H_norm_per), H_norm_per[0].shape[0]
    Gram = np.zeros((N, N))
    for k in range(K):
        d2 = pairwise_distances(H_norm_per[k], metric='sqeuclidean')
        med = np.sqrt(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
        t = med * ratios[k]
        Gram += np.exp(-d2 / (2.0 * t * t))
    return Gram / K


def gram_scores(H_norm_per, ratios, tr, te, nu):
    """Directional branch: precomputed multi-kernel Gram OC-SVM. Higher = anomalous."""
    Gram = _multi_kernel_gram(H_norm_per, ratios)
    clf = OneClassSVM(kernel='precomputed', nu=nu).fit(Gram[np.ix_(tr, tr)])
    return -clf.decision_function(Gram[np.ix_(te, tr)])


def rbf_norm_scores(H_norms, tr, te, nu):
    """Magnitude branch: RBF OC-SVM on the per-head norm vectors. Higher = anomalous."""
    clf = OneClassSVM(kernel='rbf', nu=nu).fit(H_norms[tr])
    return -clf.decision_function(H_norms[te])


def fuse(s_dir, s_mag):
    """Per-sample max of the two min-max normalised branch scores."""
    return np.maximum(_minmax(s_dir), _minmax(s_mag))


# ======================================================================
# High-level API
# ======================================================================
class SCMK:
    """Deployable SCMK detector: fit on normal samples, score new samples.

    Uses a single fixed `nu` (no label access), so score_samples() is a genuine
    detector. For paper-faithful oracle-nu reproduction use run_scmk() instead.
    """

    def __init__(self, latent_dim=64, lambda_scatter=10.0, tau=0.2, nu=0.1,
                 ratios=RATIOS, epochs=100, batch_size=512, lr=0.01, seed=42,
                 bw_mode='embedding', device=None):
        self.latent_dim = latent_dim
        self.lambda_scatter = lambda_scatter
        self.tau = tau
        self.nu = nu
        self.ratios = ratios
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.bw_mode = bw_mode
        self.device = device or default_device()

    def fit(self, X_train_normal):
        cfg = dict(epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                   normalize=True, seed=self.seed)
        self.X_train_ = np.asarray(X_train_normal, dtype=np.float32)
        self.model_ = train_cmknet(self.X_train_, self.latent_dim, self.lambda_scatter,
                                   self.tau, self.ratios, cfg, self.device, self.bw_mode)
        return self

    def score_samples(self, X_test):
        """Fused anomaly scores for X_test (higher = more anomalous). Transductive:
        the Gram/OC-SVM are fit on the stored train normals and applied to X_test."""
        X = np.vstack([self.X_train_, np.asarray(X_test, dtype=np.float32)])
        ntr = len(self.X_train_)
        tr, te = np.arange(ntr), np.arange(ntr, len(X))
        H_norm_per, H_norms = extract_components(self.model_, X, self.device)
        s_dir = gram_scores(H_norm_per, self.ratios, tr, te, self.nu)
        s_mag = rbf_norm_scores(H_norms, tr, te, self.nu)
        return fuse(s_dir, s_mag)


def run_scmk(X, y, train_idx, test_idx, latent_dim, lambda_scatter, tau,
             ratios=RATIOS, nu_list=NU_LIST, device=None, bw_mode='embedding',
             cfg=None, verbose=False, return_model=False):
    """Paper-faithful single run: train on X[train_idx], score X[test_idx].
    Each branch's nu is chosen by best test AUC (oracle protocol, as in the paper).
    Returns dict(y_test, score, auc, auc_dir, auc_mag)."""
    device = device or default_device()
    y = (np.asarray(y) != 0).astype(int)
    tr, te = np.asarray(train_idx), np.asarray(test_idx)
    y_te = y[te]
    model = train_cmknet(X[tr], latent_dim, lambda_scatter, tau, ratios, cfg, device,
                         bw_mode, verbose)
    H_norm_per, H_norms = extract_components(model, X, device)

    def best_over_nu(score_fn):
        best_a, best_s = -1.0, None
        for nu in nu_list:
            try:
                s = score_fn(nu)
                a = roc_auc_score(y_te, s)
                if a > best_a:
                    best_a, best_s = a, s
            except Exception:
                pass
        return best_a, best_s

    a_dir, s_dir = best_over_nu(lambda nu: gram_scores(H_norm_per, ratios, tr, te, nu))
    a_mag, s_mag = best_over_nu(lambda nu: rbf_norm_scores(H_norms, tr, te, nu))
    fused = fuse(s_dir, s_mag)
    result = dict(y_test=y_te, score=fused, auc=float(roc_auc_score(y_te, fused)),
                  auc_dir=float(a_dir), auc_mag=float(a_mag))
    if return_model:
        result['model'] = model
    return result


# Per-dataset best (latent_dim, lambda_scatter, tau), oracle-selected by fused AUC.
# (Manuscript 20-set configuration; extend/retune for new data.)
BEST = {
    'mammography': (32, 1.0, 0.1), 'thyroid': (64, 1.0, 0.2),
    'wbc_malignant_39_variant1': (16, 1.0, 0.1), 'glass': (64, 0.0, 0.2),
    'ecoli': (64, 1.0, 0.5), 'pageblocks_1_258_variant1': (128, 1.0, 0.05),
    'wine': (64, 0.0, 0.7), 'cardio': (16, 10.0, 0.2),
    'cardiotocography_2and3_33_variant1': (64, 1.0, 0.3),
    'tic_tac_toe_negative_12_variant1': (256, 1000.0, 1.0),
    'tic_tac_toe_negative_69_variant1': (256, 1000.0, 1.0),
    'nursery_variant1': (32, 1.0, 0.2), 'ionosphere_b_24_variant1': (16, 100.0, 0.2),
    'zoo_variant1': (256, 100.0, 0.2), 'sick_sick_72_variant1': (128, 1.0, 0.1),
    'autos_variant1': (16, 100.0, 0.3), 'annealing_variant1': (256, 1000.0, 1.0),
    'lymphography': (256, 100.0, 0.1), 'bands_band_6_variant1': (128, 100.0, 0.3),
    'audiology_variant1': (32, 1000.0, 1.0),
}


def reproduce_dataset(name, seed=0, device=None, verbose=False):
    """Load a benchmark dataset by stem, split at `seed`, run SCMK at its best config."""
    X, y = load_data(locate(name))
    tr, te = split_indices(y, seed)
    dim, lam, tau = BEST.get(name, (64, 10.0, 0.2))
    return run_scmk(X, y, tr, te, dim, lam, tau, device=device, verbose=verbose)
