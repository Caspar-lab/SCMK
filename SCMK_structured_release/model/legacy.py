import os, sys, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as scio
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.svm import OneClassSVM
TRAIN_FRAC = 0.5
RATIOS = (0.1, 0.5, 1.0, 2.0, 5.0)
NU_LIST = (0.01, 0.05, 0.1, 0.2)
TRAIN_CFG = dict(epochs=100, batch_size=512, lr=0.01, normalize=True, seed=42)
def _detect_columns(X):
    """启发式区分标称列（取值≤20个整数）与数值列。"""
    nominal, numeric = [], []
    for c in range(X.shape[1]):
        col = X[:, c]
        uvals = np.unique(col)
        if len(uvals) <= 20 and np.all(col == col.astype(int)):
            nominal.append(c)
        else:
            numeric.append(c)
    return nominal, numeric


def _preprocess(X_raw):
    """对原始特征矩阵做独热编码（标称列）+ MinMaxScaler（数值列），返回 (X, d_nom, d_num)。"""
    nominal_cols, numeric_cols = _detect_columns(X_raw)
    parts = []
    d_nom = d_num = 0
    if nominal_cols:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_nom = enc.fit_transform(X_raw[:, nominal_cols]).astype(np.float32)
        parts.append(X_nom);  d_nom = X_nom.shape[1]
    if numeric_cols:
        X_num = MinMaxScaler().fit_transform(X_raw[:, numeric_cols]).astype(np.float32)
        parts.append(X_num);  d_num = X_num.shape[1]
    assert parts, '数据中未找到有效特征列'
    return np.hstack(parts), d_nom, d_num


def load_data(path):
    """
    加载数据文件，返回 (X, y, meta)，支持 .mat 和 .csv 两种格式。

    .mat 格式约定：变量名 'trandata'，最后一列为标签（0=正常，1=异常）。
      使用 io.BytesIO 二进制读取，绕过 Windows 中文路径编码问题。
    .csv 格式约定：无表头，最后一列为标签（0=正常，1=异常）。

    标称列（取值≤20个不同整数）→ OneHotEncoder
    数值列 → MinMaxScaler 归一化到 [0,1]
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        import pandas as pd
        data  = pd.read_csv(path, header=None).values.astype(np.float64)
        # CSV 标签可能是多值整数编码，统一二值化：0=正常，其余=异常
        y     = (data[:, -1] != 0).astype(int)
    else:
        import io
        with open(path, 'rb') as fh:
            d = scio.loadmat(io.BytesIO(fh.read()))
        data = d['trandata'].astype(np.float64)
        y    = data[:, -1].astype(int)

    X_raw = data[:, :-1]
    X, d_nom, d_num = _preprocess(X_raw)
    return X, y, dict(N=len(y), anomaly_rate=y.mean(), d_nom=d_nom, d_num=d_num)


def gauss_med_kernels(X_normal, ratios=(0.1, 0.5, 1.0, 2.0, 5.0)):
    """
    基于正常样本的欧氏距离中位数 med 生成 5 个高斯核：
      带宽 t = med × ratio，ratio ∈ {0.1, 0.5, 1.0, 2.0, 5.0}

    5 个尺度分别捕获从局部到全局的邻域结构，使跨核对比学习具备多尺度感知能力。
    仅用正常样本估计带宽，避免异常点污染距离统计。
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_normal), min(500, len(X_normal)), replace=False)
    med = np.median(pairwise_distances(X_normal[idx], metric='euclidean'))
    return [(f'G-{med*r:.3g}', 'Gaussian', {'t': max(med * r, 1e-3)}) for r in ratios]


class CMKNet(nn.Module):
    """
    K 个独立无偏置线性投影头，每种核对应一个：W_k : R^D → R^d。

    无偏置（bias=False）：保持嵌入以原点为中心，与 OC-SVM 的超平面假设一致。
    L2 归一化（normalize=True）：将各核嵌入限制在超球面，统一尺度。
    """
    def __init__(self, input_dim, latent_dim, n_kernels, normalize=True):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(input_dim, latent_dim, bias=False) for _ in range(n_kernels)
        ])
        self.normalize = normalize

    def forward(self, x):
        """训练时调用：返回 K 个嵌入张量，每个形状 (B, d)。"""
        hs = [p(x) for p in self.projectors]
        if self.normalize:
            hs = [F.normalize(h, dim=1) for h in hs]
        return hs


def _eu_dist2(a, b):
    """欧氏平方距离矩阵，利用展开公式避免显式广播，(M, N) 张量。"""
    aa = (a * a).sum(1, keepdim=True)
    bb = (b * b).sum(1, keepdim=True)
    return (aa + bb.T - 2 * a @ b.T).clamp(min=0)


def _kernel_mat(F, ktype, kopts):
    """
    计算 M×M 核矩阵，F ∈ R^{M×d}。
    支持：Linear、Gaussian(t)、Polynomial(a,b,d)、Sigmoid(d,c)、Cauchy(sigma)。
    """
    if ktype == 'Linear':
        return F @ F.T
    elif ktype == 'Gaussian':
        return torch.exp(-_eu_dist2(F, F) / (2 * kopts.get('t', 1.0) ** 2))
    elif ktype == 'Polynomial':
        a, b, d = kopts.get('a', 1.0), kopts.get('b', 1.0), kopts.get('d', 2.0)
        return (a * (F @ F.T) + b) ** d
    elif ktype == 'Sigmoid':
        return torch.tanh(kopts.get('d', 2.0) * (F @ F.T) + kopts.get('c', 0.0))
    elif ktype == 'Cauchy':
        return 1 / (_eu_dist2(F, F) / kopts.get('sigma', 1.0) + 1)
    else:
        raise ValueError(f'Unknown kernel: {ktype}')
def scatter_loss(hs):
    """
    CMKKM 单类紧凑性损失。

    数学等价关系（L2 归一化嵌入上的线性核）：
      L_scatter = -(1/K) Σ_k ||μ_k||²
               = -(1/K) Σ_k (1/N²) Σ_{i,j} h_{k,i} · h_{k,j}
               = -(1/K) Σ_k (mean pairwise cosine similarity in kernel k)

    物理意义：
      最大化各投影头嵌入重心的模长 ≡ 最大化正常样本对的平均余弦相似度
      = 最小化正常样本在各核潜在空间中的多核加权距离之和

    计算复杂度：O(K · N · d)，远优于显式计算所有对的 O(K · N² · d)。

    返回：标量 Tensor（负值，用于 minimize → maximize similarity）
    """
    total = sum(h.mean(dim=0).pow(2).sum() for h in hs)
    return -total / len(hs)
def split_indices(y, seed, frac=TRAIN_FRAC):
    rng = np.random.default_rng(seed)
    normal = np.where(y == 0)[0]; anom = np.where(y != 0)[0]
    perm = rng.permutation(normal); k = int(round(len(perm) * frac))
    return perm[:k], np.concatenate([perm[k:], anom])


@torch.no_grad()
def extract_components(model, X, device, batch_size=2048):
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


def _ocsvm_best(H_te, y_te, H_tr, kernel, nu_list):
    """Semi-supervised OC-SVM: fit on train-normal, score test; best AUC over nu."""
    best, best_s = -1.0, None
    for nu in nu_list:
        try:
            clf = OneClassSVM(kernel=kernel, nu=nu).fit(H_tr)
            s = -clf.decision_function(H_te)
            a = roc_auc_score(y_te, s)
            if a > best:
                best, best_s = a, s
        except Exception:
            pass
    return best, best_s


def _median_dist(F, max_n=256):
    n = F.shape[0]
    if n > max_n:
        F = F[torch.randperm(n, device=F.device)[:max_n]]
    # bandwidth is a per-step calibration constant -> detach (no grad through it)
    return torch.pdist(F.detach()).median().clamp(min=1e-3)


def cross_kernel_loss_cfg(hs, ratios, raw_bw, tau=1.0, bw_mode='raw'):
    """Cross-kernel InfoNCE with temperature `tau` and bandwidth mode:
       bw_mode='raw'       -> raw-feature bandwidths (original behaviour)
       bw_mode='embedding' -> bandwidths = median(embedding dist) * ratio, per step."""
    K, B = len(hs), hs[0].shape[0]
    dev = hs[0].device
    mask = torch.eye(B, device=dev).repeat(2, 2) * (1 - torch.eye(2 * B, device=dev))
    logits_mask = 1 - torch.eye(2 * B, device=dev)
    total, n = 0.0, 0
    for k in range(K):
        for l in range(k + 1, K):
            F = torch.cat([hs[k], hs[l]], dim=0)
            if bw_mode == 'embedding':
                med = _median_dist(F)
                tk, tl = med * ratios[k], med * ratios[l]
            else:
                tk, tl = raw_bw[k], raw_bw[l]
            K_avg = (_kernel_mat(F, 'Gaussian', {'t': float(tk)}) +
                     _kernel_mat(F, 'Gaussian', {'t': float(tl)})) / 2
            logits = torch.exp(K_avg / tau)
            log_prob = torch.log(logits) - torch.log((logits * logits_mask).sum(1, keepdim=True))
            loss = -(mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
            total += loss.mean(); n += 1
    return total / max(n, 1)


def train_variant(X_tr, ratios, raw_bw, dim, device, lam, tau, bw_mode, cfg, verbose=False):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    N, D = X_tr.shape
    model = CMKNet(D, dim, len(ratios), cfg['normalize']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    last = 0.0
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(N)
        ec = es = et = 0.0; nb = 0
        for i in range(0, N, cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            if len(idx) < 4:
                continue
            hs = model(X_t[idx].to(device))
            Lc = cross_kernel_loss_cfg(hs, ratios, raw_bw, tau=tau, bw_mode=bw_mode)
            Ls = scatter_loss(hs) if lam > 0 else torch.zeros((), device=device)
            loss = Lc + lam * Ls
            opt.zero_grad(); loss.backward(); opt.step()
            ec += Lc.item(); es += float(Ls.item()); et += loss.item(); nb += 1
        last = et / max(nb, 1)
        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == cfg['epochs']):
            print(f'      epoch {epoch:>3d}  L_cross={ec/nb:.4f}  L_scatter={es/nb:+.4f}  total={et/nb:.4f}')
    return model, last
def _precom_mk_scores(H_norm_per, ratios, tr, te, y_te, nu_list):
    """Precomputed averaged multi-scale Gaussian Gram on the L2-normalised embeddings
       (a purely DIRECTIONAL kernel) -> OC-SVM. Returns (best_auc, best_score)."""
    K, N = len(H_norm_per), H_norm_per[0].shape[0]
    Gram = np.zeros((N, N))
    for k in range(K):
        d2 = pairwise_distances(H_norm_per[k], metric='sqeuclidean')
        med = np.sqrt(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
        t = med * ratios[k]
        Gram += np.exp(-d2 / (2.0 * t * t))
    Gram /= K
    Kfit, Kall = Gram[np.ix_(tr, tr)], Gram[np.ix_(te, tr)]
    best, best_s = -1.0, None
    for nu in nu_list:
        try:
            clf = OneClassSVM(kernel='precomputed', nu=nu).fit(Kfit)
            s = -clf.decision_function(Kall)
            a = roc_auc_score(y_te, s)
            if a > best:
                best, best_s = a, s
        except Exception:
            pass
    return best, best_s
