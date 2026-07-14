"""
cmk_rw.py — CMK-RW: Cross-Kernel Contrastive Learning + Adaptive Fusion + Random Walk
=======================================================================================

对 CMK+OC-SVM 的三项核心改进：

  改进1 | 多样化核矩阵（Stage 2）
    原问题：CMK 各投影头被 InfoNCE 目标函数训练成"一致嵌入"(h_k_i≈h_l_i)，
            导致 K 个潜空间核几乎相同，η 梯度消失、无法分化。
    方案  ：在拼接嵌入 H_all (N,K*d) 上计算多尺度高斯核+余弦核，
            不同尺度/类型在 H_all 上确实捕获不同的相似结构，η 优化有意义的梯度信号。

  改进2 | 逐样本自适应核权重 η_i（Stage 2）
    原问题：CMK+OC-SVM 等权拼接，全局超平面无法适应局部多样性
    方案  ：每个样本独立 η_i ∈ Δ^P，最小化图保持损失+核多样性正则

  改进3 | 全连接图随机游走异常评分（Stage 3）
    原问题：稀疏 k-NN 对称图导致异常样本度数与正常样本相近，φ 无法区分
    方案  ：用 η 融合核直接构建全样本相似度矩阵 A（DMFAD 全连接图逻辑），
            全局孤立的异常样本列和小 → φ 低 → 异常得分高

【与 DMFAD 的关系】
  Stage 2 (η优化) 直接源自 DMFAD objectfunction.py；
  Stage 3 (随机游走) 直接源自 DMFAD GMKAD.py GMKAD() 函数。
  创新：用 CMK 的跨核对比嵌入 H_all 作为特征，替换 DMFAD 的原始特征，
        提供更有判别力的表示空间，使 MKL 融合和随机游走都受益。
"""

import os, sys, time
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, pairwise_distances

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CMK_OCSVM import (load_data, gauss_med_kernels, train_cmk,
                        get_embeddings, TRAIN_CFG)


# ─── 默认超参数 ────────────────────────────────────────────────────────────────

LATENT_DIMS = [16, 32, 64, 128, 256]

# Stage 2：η 优化
ETA_CFG = dict(
    n_gauss    = 5,      # 多尺度高斯核的数量
    n_iters    = 150,    # Adam 迭代轮数
    lr         = 0.05,   # 学习率
    lambda_reg = 0.01,   # 核多样性正则强度 λ
    k_eta      = 10,     # η 优化 k-NN 图的邻居数
)

# Stage 3：随机游走
RW_CFG = dict(
    damping    = 0.1,    # 重启概率 d（DMFAD 取 0.1）
    max_iter   = 1000,   # 最大迭代轮数
    tol        = 1e-4,   # L1 收敛阈值
    k_sparse   = None,   # 若非 None，则对相似度矩阵做 top-k 截断（大 N 时用）
)

# 逐样本 η 优化的最大 N（超过则改为全局共享权重）
PER_SAMPLE_MAX_N = 2000


# ─── Stage 2：在 H_all 上计算多样化核矩阵 ─────────────────────────────────────

def latent_ensemble_kernels(H_all, n_gauss=5):
    """
    在 CMK 拼接嵌入 H_all (N, K*d) 上计算 P = n_gauss+1 种核矩阵。

    多尺度高斯核（n_gauss 个，带宽从细到粗对数均匀分布）：
      细带宽捕获局部微结构（哪些样本最近邻相似），粗带宽捕获全局宏结构。
      在 H_all 上这些核确实不同（vs. 在各 CMK 投影头上它们几乎相同）。

    余弦核（1 个）：
      方向相似度（归一化内积），与基于欧氏距离的高斯核结构互补。

    输入：H_all  (N, K*d)  CMK 拼接嵌入
    输出：Km      (N, N, P) float32，值域 [0,1]
    """
    N    = len(H_all)
    P    = n_gauss + 1
    Km   = np.empty((N, N, P), dtype=np.float32)

    # 中位数带宽估计（子采样加速）
    rng  = np.random.default_rng(42)
    idx  = rng.choice(N, min(N, 500), replace=False)
    sq_s = pairwise_distances(H_all[idx], metric='sqeuclidean')
    med2 = max(float(np.median(sq_s[sq_s > 0])), 1e-10)

    # 全量平方欧氏距离矩阵
    sq_full = pairwise_distances(H_all, metric='sqeuclidean').astype(np.float32)

    # 多尺度高斯核（σ²范围：0.1×med² 到 10×med²）
    for k, r in enumerate(np.geomspace(0.1, 10.0, n_gauss)):
        Km[:, :, k] = np.exp(-sq_full / (2.0 * float(med2) * r))

    # 余弦核，归一化到 [0,1]
    Hn = H_all / np.maximum(
        np.linalg.norm(H_all, axis=1, keepdims=True), 1e-10)
    cos = np.clip(Hn @ Hn.T, -1.0, 1.0).astype(np.float32)
    Km[:, :, -1] = (cos + 1.0) / 2.0

    return Km


def kernel_correlation_matrix(Km):
    """
    核相关矩阵 M ∈ R^{P×P}，M_{kl} = Tr(K_k · K_l) / N²。
    （直接来自 DMFAD kernel_utils.calculate_kernel_correlation_matrix）
    M_{kl} 越大 → 核 k 与 l 越冗余；正则项惩罚将 η 集中于冗余核对。
    """
    N, _, P = Km.shape
    flat    = Km.transpose(2, 0, 1).reshape(P, N * N)   # (P, N²)，与 DMFAD 一致
    return (flat @ flat.T / float(N * N)).astype(np.float32)


def knn_graph(dist, k):
    """对称 k-NN 邻接矩阵（0/1 二值），用于 η 优化的初始图。"""
    from sklearn.neighbors import NearestNeighbors
    N  = dist.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, metric='precomputed').fit(dist)
    _, indices = nn.kneighbors(dist)
    W  = np.zeros((N, N), dtype=np.float32)
    for i, idx in enumerate(indices):
        W[i, idx[1:]] = 1.0
    return np.maximum(W, W.T)


# ─── Stage 2：η 优化 ──────────────────────────────────────────────────────────

def _optimize_eta_ps(Km, M, cfg):
    """
    逐样本 η（N ≤ PER_SAMPLE_MAX_N）。

    目标函数（直接来自 DMFAD objectfunction.py）：
      J = Σ_{(i,j)∈kNN} W_ij · d_ij(η)  +  λ · Σ_i η_i M η_i^T
    其中 d_ij(η) = K_c[i,i]+K_c[j,j]−2K_c[i,j]，K_c 为 η 融合核。
    """
    N, _, P = Km.shape

    # 用均值核的距离构建初始 k-NN 图
    Km_mean = Km.mean(axis=2)
    Kd      = np.diag(Km_mean)
    dist0   = np.sqrt(np.clip(Kd[:, None] + Kd[None, :] - 2.0 * Km_mean, 0.0, None))
    W_t     = torch.tensor(knn_graph(dist0, k=cfg['k_eta']), dtype=torch.float32)

    Km_t = torch.tensor(Km, dtype=torch.float32)
    M_t  = torch.tensor(M,  dtype=torch.float32)

    logit = torch.zeros(N, P, requires_grad=True)
    opt   = optim.Adam([logit], lr=cfg['lr'])

    for _ in range(cfg['n_iters']):
        eta   = torch.softmax(logit, dim=1)                               # (N, P)
        K_c   = torch.einsum('ik,jk,ijk->ij', eta, eta, Km_t)            # (N, N)
        Kd_t  = K_c.diagonal()
        dist2 = (Kd_t.unsqueeze(1) + Kd_t.unsqueeze(0) - 2.0 * K_c).clamp(0)
        loss  = ((W_t * dist2).sum()
                 + cfg['lambda_reg'] * torch.einsum('ik,kl,il->', eta, M_t, eta))
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        return torch.softmax(logit, dim=1).numpy().astype(np.float32)     # (N, P)


def _optimize_eta_global(Km_full, M, cfg):
    """全局共享 η（N > PER_SAMPLE_MAX_N）：子采样 800 个点估计目标。"""
    N, _, P = Km_full.shape
    rng  = np.random.default_rng(42)
    idx  = rng.choice(N, min(N, 800), replace=False)
    Km   = Km_full[np.ix_(idx, idx)]

    Km_mean = Km.mean(axis=2)
    Kd      = np.diag(Km_mean)
    dist0   = np.sqrt(np.clip(Kd[:, None] + Kd[None, :] - 2.0 * Km, 0.0, None))
    W_t     = torch.tensor(knn_graph(dist0, k=cfg['k_eta']), dtype=torch.float32)

    Km_t = torch.tensor(Km, dtype=torch.float32)
    M_t  = torch.tensor(M,  dtype=torch.float32)

    logit = torch.zeros(P, requires_grad=True)
    opt   = optim.Adam([logit], lr=cfg['lr'])

    for _ in range(cfg['n_iters']):
        eta   = torch.softmax(logit, dim=0)                               # (P,)
        K_c   = torch.einsum('k,ijk->ij', eta * eta, Km_t)               # (sub, sub)
        Kd_t  = K_c.diagonal()
        dist2 = (Kd_t.unsqueeze(1) + Kd_t.unsqueeze(0) - 2.0 * K_c).clamp(0)
        loss  = ((W_t * dist2).sum()
                 + cfg['lambda_reg'] * (eta @ M_t @ eta))
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        eta_g = torch.softmax(logit, dim=0).numpy()                       # (P,)
    return np.tile(eta_g, (N, 1)).astype(np.float32)                      # (N, P)


def optimize_eta(Km, M, eta_cfg=None):
    """统一入口，自动按 N 选择逐样本或全局权重优化。返回 eta (N, P)。"""
    cfg = eta_cfg or ETA_CFG
    return (_optimize_eta_ps(Km, M, cfg) if Km.shape[0] <= PER_SAMPLE_MAX_N
            else _optimize_eta_global(Km, M, cfg))


def combined_kernel(Km, eta):
    """融合核：K_comb[i,j] = Σ_k η[i,k] · η[j,k] · Km[i,j,k]。"""
    return np.einsum('ik,jk,ijk->ij', eta, eta, Km, optimize=True)


# ─── Stage 3：随机游走异常评分 ─────────────────────────────────────────────────

def random_walk_score(K_comb, seed_mask=None,
                      damping=0.1, max_iter=1000, tol=1e-4, k_sparse=None):
    """
    One-class 随机游走异常评分。

    【设计动机】
    DMFAD 在原始特征空间中运行随机游走，异常样本天然孤立。
    CMK 的 H_all 潜空间无此保证：异常样本可能形成紧密子簇（如 cardio 心律不齐），
    导致均匀重启时异常 φ 反而更高（AUC<0.5）。

    【One-class 修正】
    通过 seed_mask（正常训练样本掩码）构建非均匀重启分布：
      φ_restart[i] = 1/N_normal  if i ∈ normal train set,  0 otherwise
    迭代：φ_{t+1} = d·φ_restart + (1−d)·φ_t·P

    物理意义：
      正常样本被持续"重新注入"概率；
      异常样本的概率随迭代流向相邻正常区域（P[异常→正常] > 0），
      但无法从重启项补充 → φ[异常] < φ[正常]，与原始特征空间中的孤立性无关。

    seed_mask=None 时退化为均匀重启（DMFAD 原始格式），适用于特征空间已知分离时。

    k_sparse：若非 None，则对 A 每行保留 top-k_sparse（大 N 时内存优化）。
    """
    N = K_comb.shape[0]
    A = K_comb.copy().astype(np.float64)
    np.fill_diagonal(A, 0.0)

    if k_sparse is not None and k_sparse < N - 1:
        thresh = np.partition(A, -k_sparse, axis=1)[:, -k_sparse: -k_sparse + 1]
        A      = np.where(A >= thresh, A, 0.0)
        A      = np.maximum(A, A.T)

    row_sum = A.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
    P       = A / row_sum                    # (N, N) row-stochastic

    # 重启分布
    if seed_mask is not None:
        n_seed  = int(seed_mask.sum())
        restart = np.zeros(N, dtype=np.float64)
        restart[seed_mask] = 1.0 / max(n_seed, 1)
    else:
        restart = np.ones(N, dtype=np.float64) / N   # DMFAD 均匀重启

    phi = restart.copy()
    for _ in range(max_iter):
        phi_new = damping * restart + (1.0 - damping) * (phi @ P)
        if np.linalg.norm(phi_new - phi, 1) < tol:
            phi = phi_new
            break
        phi = phi_new

    return (1.0 - phi).astype(np.float32)    # 高值 = 异常


# ─── 完整流水线 ────────────────────────────────────────────────────────────────

def cmk_rw_score(X, y, kernels, latent_dim, device,
                 train_cfg=None, eta_cfg=None, rw_cfg=None,
                 train_ratio=0.7, random_state=42):
    """
    CMK-RW 三阶段流水线（无数据泄露）。

    【数据划分设计】
    正常样本按 train_ratio 分为两份：
      - 训练集（70%）：用于 CMK 对比训练 + 随机游走重启种子
      - 验证集（30%）：与全部异常样本合并，构成无泄露评估集

    AUC 仅在评估集上计算。若用全部 y==0 做种子，随机游走 φ 值
    直接由 seed_mask 决定（训练正常≈1，异常≈0），AUC 退化为读取标签。

    train_ratio : 正常样本中用于训练+种子的比例（默认 0.7）
    random_state: 随机分割种子

    返回：(auc, scores_full, eta, elapsed_s)
      auc         — 在验证集（val_normals ∪ anomalies）上的 ROC-AUC
      scores_full — 全 N 个样本的异常分数
    """
    cfg_t = train_cfg or TRAIN_CFG
    cfg_e = eta_cfg   or ETA_CFG
    cfg_r = rw_cfg    or RW_CFG
    t0    = time.time()

    # ── 无泄露数据划分 ────────────────────────────────────────────────────────
    normal_idx = np.where(y == 0)[0]
    anom_idx   = np.where(y == 1)[0]
    rng        = np.random.default_rng(random_state)
    perm       = rng.permutation(len(normal_idx))
    n_tr       = max(1, int(len(normal_idx) * train_ratio))
    train_nidx = normal_idx[perm[:n_tr]]   # 训练正常样本（CMK + RW 种子）
    val_nidx   = normal_idx[perm[n_tr:]]   # 验证正常样本（仅用于评估，模型未见）

    # train_cmk 内部用 y_tr==0 筛选训练样本，只暴露训练正常样本
    y_tr = np.ones(len(X), dtype=np.int64)
    y_tr[train_nidx] = 0

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    model = train_cmk(X, y_tr, kernels, latent_dim, device, cfg_t)
    H_all = get_embeddings(model, X, device)                        # (N, K*d)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    Km     = latent_ensemble_kernels(H_all, n_gauss=cfg_e['n_gauss'])
    M      = kernel_correlation_matrix(Km)                          # (P, P)
    eta    = optimize_eta(Km, M, cfg_e)                             # (N, P)
    K_comb = combined_kernel(Km, eta)                               # (N, N)

    # ── Stage 3：种子 = 训练正常样本，不含验证正常样本 ───────────────────────
    N         = len(y)
    k_sp      = cfg_r.get('k_sparse') if N > 3000 else None
    seed_mask = np.zeros(N, dtype=bool)
    seed_mask[train_nidx] = True                                    # 仅训练正常样本

    scores = random_walk_score(K_comb,
                               seed_mask = seed_mask,
                               damping   = cfg_r['damping'],
                               max_iter  = cfg_r['max_iter'],
                               tol       = cfg_r['tol'],
                               k_sparse  = k_sp)

    # ── AUC：验证正常样本 + 全部异常样本（无泄露） ───────────────────────────
    eval_idx = np.concatenate([val_nidx, anom_idx])
    auc      = roc_auc_score(y[eval_idx], scores[eval_idx])
    return auc, scores, eta, time.time() - t0


# ─── 单数据集主程序 ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pandas as pd

    _default = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'dataset', 'numerical', 'cardio.mat')
    _path = sys.argv[1] if len(sys.argv) > 1 else _default

    RESULT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'result')
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f'loading: {os.path.basename(_path)}')
    X, y, meta = load_data(_path)
    stem        = os.path.splitext(os.path.basename(_path))[0]
    print(f'N={meta["N"]}  D={X.shape[1]}  anomaly={meta["anomaly_rate"]*100:.1f}%'
          f'  n_normal={(y==0).sum()}')

    kernels = gauss_med_kernels(X[y == 0])
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  K={len(kernels)}  n_gauss={ETA_CFG["n_gauss"]}\n{"─"*58}')

    rows = []
    for dim in LATENT_DIMS:
        auc, scores, eta, elapsed = cmk_rw_score(X, y, kernels, dim, device)
        eta_str = ', '.join(f'{v:.2f}' for v in eta.mean(axis=0))
        print(f'dim={dim:>3d}  AUC={auc:.4f}  ({elapsed:.1f}s)  eta_mean=[{eta_str}]')
        rows.append(dict(dataset=stem, latent_dim=dim,
                         auc_rw=round(auc, 6), elapsed_s=round(elapsed, 2)))

    df  = pd.DataFrame(rows)
    csv = os.path.join(RESULT_DIR, f'{stem}_cmk_rw.csv')
    df.to_csv(csv, index=False)
    best = max(rows, key=lambda r: r['auc_rw'])
    print(f'\n{"="*45}')
    print(f'best: dim={best["latent_dim"]}  AUC={best["auc_rw"]:.4f}')
    print(f'saved: {csv}')
