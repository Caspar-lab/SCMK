"""
CMK_AD — CMK for Anomaly Detection on Mixed-type Data

Pipeline:
  1. 加载混合类型数据，自动识别标称列（view 1）和数值列（view 2）
  2. 标称列 → 独热编码；数值列 → 标准化
  3. Mini-batch CMK 阶段：训练投影网络 W（跨视图对比核损失）
  4. 所有样本通过 W 映射到嵌入空间
  5. KNN 异常评分（嵌入空间 vs 原始特征），AUC-ROC 评估
"""

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as scio
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


# ─── 数据加载与视图分离 ────────────────────────────────────────────────────────

def detect_columns(X):
    """
    启发式判断各列是标称型还是数值型。
    规则：唯一值 ≤ 20 且全为整数 → 标称列。
    """
    nominal, numeric = [], []
    for c in range(X.shape[1]):
        col = X[:, c]
        uvals = np.unique(col)
        if len(uvals) <= 20 and np.all(col == col.astype(int)):
            nominal.append(c)
        else:
            numeric.append(c)
    return nominal, numeric


def load_mixed_data(path):
    """
    加载 .mat 文件（格式：trandata, 最后一列为标签）。
    返回:
        X_nom  : (N, D_nom_enc) 独热编码的标称特征
        X_num  : (N, D_num) 标准化的数值特征
        y      : (N,) 标签，1=异常，0=正常
        meta   : dict，包含列索引、编码器等
    """
    import io
    with open(path, 'rb') as fh:
        d = scio.loadmat(io.BytesIO(fh.read()))
    data = d['trandata'].astype(np.float64)
    y = data[:, -1].astype(int)
    X = data[:, :-1]

    nominal_cols, numeric_cols = detect_columns(X)
    assert len(nominal_cols) > 0, '未检测到标称列'
    assert len(numeric_cols) > 0, '未检测到数值列'

    # 标称视图：独热编码（0/1，已自然落在 [0,1]）
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_nom = enc.fit_transform(X[:, nominal_cols]).astype(np.float32)

    # 数值视图：StandardScaler 用于 CMK 训练（零均值单方差）
    std_scaler = MinMaxScaler()
    X_num = std_scaler.fit_transform(X[:, numeric_cols]).astype(np.float32)

    # MinMaxScaler 数值用于 KNN 基线（特征级 [0,1]，与独热量级一致）
    mm_scaler = MinMaxScaler()
    X_num_mm = mm_scaler.fit_transform(X).astype(np.float32)

    meta = dict(nominal_cols=nominal_cols, numeric_cols=numeric_cols,
                encoder=enc, std_scaler=std_scaler, mm_scaler=mm_scaler,
                X_num_mm=X_num_mm,
                N=len(y), anomaly_rate=y.mean())
    return X_nom, X_num, y, meta


# ─── 投影网络 ──────────────────────────────────────────────────────────────────

class FCNet(nn.Module):
    """两视图独立线性投影 W_v: D_v → latent_dim（无偏置，可选 L2 归一化）。"""

    def __init__(self, d_nom, d_num, latent_dim=64, normalize=True):
        super().__init__()
        self.fc_nom = nn.Linear(d_nom, latent_dim, bias=False)
        self.fc_num = nn.Linear(d_num, latent_dim, bias=False)
        self.normalize = normalize

    def forward(self, x_nom, x_num):
        h1 = self.fc_nom(x_nom)
        h2 = self.fc_num(x_num)
        if self.normalize:
            h1 = F.normalize(h1, dim=1)
            h2 = F.normalize(h2, dim=1)
        return h1, h2

    @torch.no_grad()
    def embed(self, x_nom, x_num):
        h1, h2 = self.forward(x_nom, x_num)
        return torch.cat([h1, h2], dim=1)


# ─── Mini-batch 对比核损失（CMK 损失的批次化版本）────────────────────────────

def _eu_dist2(a, b):
    aa = (a * a).sum(1, keepdim=True)
    bb = (b * b).sum(1, keepdim=True)
    return (aa + bb.T - 2 * a @ b.T).clamp(min=0)


def contrastive_kernel_loss(h1, h2, kernel_type='Linear', kernel_opts=None):
    """
    Mini-batch CMK 对比损失。
    正样本对：batch 内相同索引的跨视图嵌入 (h1_i, h2_i)。
    h1, h2 : (B, latent_dim)
    """
    if kernel_opts is None:
        kernel_opts = {}
    B = h1.shape[0]
    device = h1.device
    features = torch.cat([h1, h2], dim=0)  # (2B, d)

    # 核矩阵 (2B, 2B)
    ktype = kernel_type
    if ktype == 'Linear':
        K = features @ features.T
    elif ktype == 'Gaussian':
        t = kernel_opts.get('t', 1.0)
        D = _eu_dist2(features, features)
        K = torch.exp(-D / (2 * t ** 2))
    elif ktype == 'Polynomial':
        a = kernel_opts.get('a', 1.0)
        b = kernel_opts.get('b', 1.0)
        d = kernel_opts.get('d', 2.0)
        K = (a * (features @ features.T) + b) ** d
    elif ktype == 'Sigmoid':
        d = kernel_opts.get('d', 2.0)
        c = kernel_opts.get('c', 0.0)
        K = torch.tanh(d * (features @ features.T) + c)
    elif ktype == 'Cauchy':
        sigma = kernel_opts.get('sigma', 1.0)
        D = _eu_dist2(features, features)
        K = 1 / (D / sigma + 1)
    else:
        raise ValueError(f'Unknown kernel: {ktype}')

    # 正样本掩码：(i, i+B) 和 (i+B, i) 互为正对
    mask = torch.eye(B, device=device).repeat(2, 2)
    logits_mask = 1 - torch.eye(2 * B, device=device)
    mask = mask * logits_mask

    logits = torch.exp(K)
    log_prob = torch.log(logits) - torch.log((logits * logits_mask).sum(1, keepdim=True))
    loss = -(mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
    return loss.mean()


# ─── 训练 ─────────────────────────────────────────────────────────────────────

def train_cmk(X_nom, X_num, latent_dim=64, epochs=100, batch_size=512,
              lr=0.01, kernel_type='Linear', kernel_opts=None,
              normalize=True, device=None, seed=42, print_freq=20):
    """
    Mini-batch CMK 训练，返回训练好的 FCNet 模型。
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if kernel_opts is None:
        kernel_opts = {}

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    N = X_nom.shape[0]
    model = FCNet(X_nom.shape[1], X_num.shape[1], latent_dim, normalize).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_nom_t = torch.tensor(X_nom, dtype=torch.float32)
    X_num_t = torch.tensor(X_num, dtype=torch.float32)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i: i + batch_size]
            if len(idx) < 4:       # 跳过过小的批次
                continue
            xn = X_nom_t[idx].to(device)
            xv = X_num_t[idx].to(device)
            h1, h2 = model(xn, xv)
            loss = contrastive_kernel_loss(h1, h2, kernel_type, kernel_opts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % print_freq == 0:
            print(f'    epoch {epoch:4d}/{epochs}  loss={epoch_loss/n_batches:.4f}'
                  f'  time={time.time()-t0:.1f}s')

    return model


# ─── 嵌入提取与 KNN 评分 ──────────────────────────────────────────────────────

@torch.no_grad()
def get_embeddings(model, X_nom, X_num, batch_size=2048, device=None):
    """将全部样本通过训练好的 W 映射到嵌入空间，返回 (N, 2*latent_dim) numpy。"""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    X_nom_t = torch.tensor(X_nom, dtype=torch.float32)
    X_num_t = torch.tensor(X_num, dtype=torch.float32)
    parts = []
    for i in range(0, len(X_nom), batch_size):
        xn = X_nom_t[i: i + batch_size].to(device)
        xv = X_num_t[i: i + batch_size].to(device)
        parts.append(model.embed(xn, xv).cpu().numpy())
    return np.concatenate(parts, axis=0)


def knn_anomaly_score(embeddings, k=5):
    """KNN 异常评分：每个样本到其 k 个最近邻的平均距离（越大越异常）。"""
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nn.fit(embeddings)
    dists, _ = nn.kneighbors(embeddings)
    return dists[:, 1:].mean(axis=1)   # 排除自身


def best_knn_auc(embeddings, y, k_min=2, k_max=60):
    """
    在 [k_min, k_max] 范围内搜索最优 K，返回 (best_k, best_auc)。
    预先一次性算出最大邻居距离矩阵，避免重复建索引。
    """
    nn = NearestNeighbors(n_neighbors=k_max + 1, metric='euclidean', n_jobs=-1)
    nn.fit(embeddings)
    dists, _ = nn.kneighbors(embeddings)   # (N, k_max+1), col 0 = self
    best_k, best_auc = k_min, -1.0
    for k in range(k_min, k_max + 1):
        scores = dists[:, 1:k + 1].mean(axis=1)
        auc = roc_auc_score(y, scores)
        if auc > best_auc:
            best_auc, best_k = auc, k
    return best_k, best_auc


def _baseline_concat(X_nom, X_num_mm):
    """MinMax 归一化拼接：X_nom 已 0/1，X_num_mm 已 MinMaxScaler → 量级一致。"""
    return np.hstack([X_nom, X_num_mm])


def best_knn_baseline_auc(X_nom, X_num_mm, y, k_min=2, k_max=60):
    """KNN 基线（MinMax 归一化）在 [k_min, k_max] 内搜索最优 K。"""
    return best_knn_auc(_baseline_concat(X_nom, X_num_mm), y, k_min, k_max)




# ─── 5 种标准核的定义（与 CMK 论文一致）────────────────────────────────────────

ALL_KERNELS = [
    ('Gaussian',   'Gaussian',   {'t': 1.0}),
    ('Linear',     'Linear',     {}),
    ('Polynomial', 'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 2.0}),
    ('Sigmoid',    'Sigmoid',    {'d': 2.0, 'c': 0.0}),
    ('Cauchy',     'Cauchy',     {'sigma': 1.0}),
]


# ─── 多核 CMK 训练 ─────────────────────────────────────────────────────────────

def train_multi_kernel_cmk(X_nom, X_num, kernels=None, latent_dim=64,
                            epochs=100, batch_size=512, lr=0.01,
                            normalize=True, device=None, seed=42, verbose=True):
    """
    为每种核独立训练一个 W，收集各核嵌入后拼接，形成多核联合表示。

    遵循 CMK 论文的多核思路：
      - 每种核对应一组投影参数 W_k
      - 各核嵌入 H_k ∈ R^{N × 2d} 拼接为 H_all ∈ R^{N × 2dK}
      - KNN 在 H_all 上计算，等价于多核均值距离（L2 归一化后）

    返回: (H_all, per_kernel_aucs_placeholder)
    """
    if kernels is None:
        kernels = ALL_KERNELS
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    all_embs = []
    for name, ktype, kopts in kernels:
        if verbose:
            print(f'    [{name}] 训练 W ...', end='', flush=True)
        t0 = time.time()
        model = train_cmk(
            X_nom, X_num,
            latent_dim=latent_dim, epochs=epochs, batch_size=batch_size,
            lr=lr, kernel_type=ktype, kernel_opts=kopts,
            normalize=normalize, device=device, seed=seed,
            print_freq=epochs + 1,   # 不打印 epoch 进度
        )
        emb = get_embeddings(model, X_nom, X_num, device=device)
        all_embs.append(emb)
        if verbose:
            print(f' done ({time.time()-t0:.1f}s)')

    # 拼接所有核的嵌入：(N, 2 * latent_dim * K)
    H_all = np.concatenate(all_embs, axis=1)
    return H_all, all_embs   # all_embs 可用于查看单核结果


# ─── 单数据集实验 ──────────────────────────────────────────────────────────────

def run_experiment(path, latent_dim=64, epochs=100, batch_size=512,
                   lr=0.01, kernels=None, k_min=2, k_max=60,
                   normalize=True, seed=42, verbose=True):
    """
    对比三种方法（K 在 [k_min, k_max] 内搜索最优）：
      1. KNN 基线（原始特征）
      2. 单核 CMK+KNN（每种核单独，取最优 AUC）
      3. 多核 CMK+KNN（所有核嵌入拼接后 KNN）

    返回 dict 含各方法最优 AUC-ROC、最优 K 及数据集信息。
    """
    if kernels is None:
        kernels = ALL_KERNELS

    if verbose:
        print(f'  加载: {os.path.basename(path)}')
    X_nom, X_num, y, meta = load_mixed_data(path)
    N, ar = meta['N'], meta['anomaly_rate']
    if verbose:
        print(f'  N={N}  D_nom={X_nom.shape[1]}  D_num={X_num.shape[1]}  '
              f'异常率={ar*100:.1f}%  K搜索范围=[{k_min},{k_max}]')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X_num_mm = meta['X_num_mm']   # MinMax 归一化数值，用于 KNN 基线

    # ── 1a. 手动 KNN 基线（MinMax 归一化，搜索最优 K）────────────────────────
    t0 = time.time()
    best_k_knn, auc_knn = best_knn_baseline_auc(X_nom, X_num_mm, y, k_min, k_max)
    t_knn = time.time() - t0
    if verbose:
        print(f'  [KNN-MinMax]  AUC={auc_knn:.4f}  best_k={best_k_knn}  ({t_knn:.1f}s)')

    # ── 2 & 3. 多核 CMK（同时获得单核和多核结果）────────────────────────────
    t0 = time.time()
    if verbose:
        print(f'  [多核 CMK  ]  训练 {len(kernels)} 种核 ...')
    H_all, per_embs = train_multi_kernel_cmk(
        X_nom, X_num, kernels=kernels,
        latent_dim=latent_dim, epochs=epochs, batch_size=batch_size,
        lr=lr, normalize=normalize, device=device, seed=seed, verbose=verbose,
    )
    t_cmk = time.time() - t0

    # 单核 AUC（每种核搜索最优 K）
    single_aucs = {}
    single_best_ks = {}
    for (name, _, _), emb in zip(kernels, per_embs):
        bk, bauc = best_knn_auc(emb, y, k_min, k_max)
        single_aucs[name] = bauc
        single_best_ks[name] = bk

    # 多核 AUC（拼接嵌入，搜索最优 K）
    best_k_mk, auc_mk = best_knn_auc(H_all, y, k_min, k_max)
    best_single_name = max(single_aucs, key=single_aucs.get)
    auc_best_single = single_aucs[best_single_name]
    best_k_single = single_best_ks[best_single_name]

    if verbose:
        for name, auc in single_aucs.items():
            print(f'    [{name:<11}]  AUC={auc:.4f}  best_k={single_best_ks[name]}')
        print(f'  [多核 CMK  ]  AUC={auc_mk:.4f}  best_k={best_k_mk}  ({t_cmk:.1f}s)')

    return dict(
        dataset=os.path.splitext(os.path.basename(path))[0],
        auc_knn=auc_knn, best_k_knn=best_k_knn,
        auc_best_single=auc_best_single, best_kernel=best_single_name,
        best_k_single=best_k_single,
        auc_mk=auc_mk, best_k_mk=best_k_mk,
        single_aucs=single_aucs, single_best_ks=single_best_ks,
        N=N, anomaly_rate=ar,
        d_nom=X_nom.shape[1], d_num=X_num.shape[1],
        t_knn=t_knn, t_cmk=t_cmk,
    )


if __name__ == '__main__':
    path = r'D:\Microsoft\documents\博士课题\异常检测\论文\5\dataset\mixed\abalone_variant1.mat'
    res = run_experiment(path, epochs=50, batch_size=256, latent_dim=64, verbose=True)
    print(f'\n  KNN={res["auc_knn"]:.4f}  '
          f'Best-Single({res["best_kernel"]})={res["auc_best_single"]:.4f}  '
          f'Multi-Kernel={res["auc_mk"]:.4f}')
