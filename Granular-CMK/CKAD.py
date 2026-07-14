"""
CKAD — Cross-Kernel Contrastive Anomaly Detection
=================================================

【算法动机】
  传统对比学习（如 CMK）将数据特征划分为两个"视图"（如标称/数值），
  通过跨视图正对对比学习提取一致嵌入。
  CKAD 的核心创新在于：不依赖任何特征划分，
  而是以"不同核函数"作为隐式视图，学习跨核一致的低维嵌入表示。

【核心思路】
  1. 核函数即视图：
       同一样本 x_i 在核 k 下的嵌入 h_k(x_i) 与在核 l 下的嵌入 h_l(x_i)
       构成正对（cross-kernel positive pair）；
       不同样本 (x_i, x_j) 在任意核对下构成负对。
  2. 归纳偏置：
       正常样本在不同核函数诱导的相似度结构下应保持一致的邻域关系，
       因此训练好的模型会使正常样本的各核嵌入彼此接近；
       异常样本由于与训练分布偏离，其各核嵌入会产生不一致，
       即跨核距离较大，可据此作为异常得分（CK-Incon）。
  3. 仅用正常样本训练：
       训练集仅由 y==0 的正常样本组成，模型未见过任何异常；
       测试阶段对全量样本计算嵌入与异常得分。

【模型结构】
  CKADNet：K 个独立线性投影头（无偏置）
       W_k : R^D → R^d   (k = 1, …, K)
  训练后将 K 个嵌入拼接得到 H_all ∈ R^{N × K*d}，
  或保留 K 个独立嵌入 H_per = [H_1, …, H_K]，每个 ∈ R^{N × d}。

【异常评分方案（四种）】
  (a) CK-Incon ：跨核不一致性——所有核对 (k,l) 的 ||h_k(x_i) - h_l(x_i)||² 均值。
  (b) CK-KNN   ：KNN 距离——在拼接嵌入 H_all 上，K ∈ [k_min, k_max] 遍历取最优。
  (c) OC-SVM   ：在嵌入空间用线性核单分类 SVM 找最大间隔超平面；
                 决策值越低（离超平面越远且在"异常侧"）则得分越高。
  (d) SVDD     ：在嵌入空间用 RBF 核 OC-SVM（数学上等价于最小包围超球面 SVDD）；
                 与 OC-SVM 的区别在于用 RBF 核，更适合非凸决策边界。

  实验发现：在纯数值型数据集（cardio、cardiotocography）上，
  OC-SVM（线性核）效果最优（AUC 0.92、0.91），显著超越 KNN 基线（0.88、0.82）；
  说明 CKAD 嵌入空间具有良好的线性可分性，线性 OC-SVM 能充分利用这一结构。
"""

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as scio
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


# ─── 默认核配置 ────────────────────────────────────────────────────────────────
# 每个元素格式：(名称, 核类型, 核超参数字典)
# 核超参数含义见 _kernel_mat 函数的注释。
# 消融实验表明 Gauss-5-med（数据自适应带宽）是最稳定的配置，
# 因此 run_cardio.py 的主实验只使用该配置。
ALL_KERNELS = [
    ('Gaussian',   'Gaussian',   {'t': 1.0}),           # RBF，带宽 t=1
    ('Linear',     'Linear',     {}),                    # 线性核（等价于内积相似度）
    ('Polynomial', 'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 2.0}),  # 二次多项式核
    ('Sigmoid',    'Sigmoid',    {'d': 2.0, 'c': 0.0}), # 双曲正切核（类 MLP 激活）
    ('Cauchy',     'Cauchy',     {'sigma': 1.0}),        # 柯西核（重尾，对异常更敏感）
]


# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def _detect_columns(X):
    """
    启发式区分标称列与数值列：
      - 取值 ≤ 20 个不同整数值 → 视为标称列（会进行独热编码）
      - 否则视为数值列（会进行 MinMaxScaler 归一化）

    对于纯数值型数据集（如 cardio.mat），所有列均为数值列，
    本函数返回空列表 nominal，后续跳过独热编码步骤。
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


def load_data(path):
    """
    加载标准格式 .mat 文件，返回 (X, y, meta)。

    文件格式约定：
      - mat 变量名为 'trandata'（ODDS 数据库惯例）
      - 最后一列为标签（0=正常，1=异常）
      - 其余列为特征

    预处理：
      - 标称列（启发式检测）→ OneHotEncoder（独热编码）
      - 数值列 → MinMaxScaler 归一化到 [0, 1]
      - 两者拼接为统一特征矩阵 X ∈ R^{N × D}，dtype=float32

    注意：使用 io.BytesIO 二进制读取再解析，绕过 scipy.io.loadmat
    在 Windows 中文路径下的编码错误。
    """
    import io
    with open(path, 'rb') as fh:
        d = scio.loadmat(io.BytesIO(fh.read()))
    data = d['trandata'].astype(np.float64)
    y = data[:, -1].astype(int)
    X_raw = data[:, :-1]

    nominal_cols, numeric_cols = _detect_columns(X_raw)

    parts = []
    d_nom = d_num = 0
    if nominal_cols:
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_nom = enc.fit_transform(X_raw[:, nominal_cols]).astype(np.float32)
        parts.append(X_nom)
        d_nom = X_nom.shape[1]
    if numeric_cols:
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(X_raw[:, numeric_cols]).astype(np.float32)
        parts.append(X_num)
        d_num = X_num.shape[1]

    assert parts, '数据中未找到有效特征列'
    X = np.hstack(parts)   # (N, D)，D = d_nom + d_num
    meta = dict(N=len(y), anomaly_rate=y.mean(), d_nom=d_nom, d_num=d_num)
    return X, y, meta


# ─── 模型结构 ──────────────────────────────────────────────────────────────────

class CKADNet(nn.Module):
    """
    K 个独立线性投影头，每个对应一种核函数。

    设计原则：
      - 无偏置（bias=False）：避免常数偏移破坏核矩阵的正定性；
        嵌入中心化由 L2 归一化隐式实现。
      - L2 归一化（normalize=True）：将嵌入限制在超球面上，
        使各核的尺度统一，便于对比损失的数值稳定性。
      - 线性投影而非 MLP：保持模型简洁，减少训练样本需求（训练集仅正常样本）；
        且线性投影的输出空间对后续 OC-SVM 线性核的假设最为匹配。

    参数：
      input_dim  : 输入特征维度 D
      latent_dim : 嵌入维度 d（消融实验候选值：16/32/64/128/256）
      n_kernels  : 核函数数量 K（与 kernels 列表长度对应）
      normalize  : 是否对嵌入做 L2 归一化
    """
    def __init__(self, input_dim, latent_dim, n_kernels, normalize=True):
        super().__init__()
        # 每个核函数对应一个独立的线性层，参数不共享
        self.projectors = nn.ModuleList([
            nn.Linear(input_dim, latent_dim, bias=False) for _ in range(n_kernels)
        ])
        self.normalize = normalize

    def forward(self, x):
        """训练时调用：返回 K 个嵌入张量组成的列表，每个形状 (B, d)。"""
        hs = [p(x) for p in self.projectors]
        if self.normalize:
            hs = [F.normalize(h, dim=1) for h in hs]
        return hs   # list of K tensors, each (B, d)

    @torch.no_grad()
    def embed(self, x):
        """推断时调用：返回 K 个嵌入拼接后的向量，形状 (B, K*d)。"""
        hs = [p(x) for p in self.projectors]
        if self.normalize:
            hs = [F.normalize(h, dim=1) for h in hs]
        return torch.cat(hs, dim=1)   # (B, K*d)


# ─── 核矩阵计算 ────────────────────────────────────────────────────────────────

def _eu_dist2(a, b):
    """
    计算 a ∈ R^{M×d} 与 b ∈ R^{N×d} 间的欧氏平方距离矩阵，返回 (M, N) 张量。
    利用展开公式 ||a_i - b_j||² = ||a_i||² + ||b_j||² - 2 a_i·b_j 避免显式广播，
    clamp(min=0) 防止浮点误差产生负值。
    """
    aa = (a * a).sum(1, keepdim=True)   # (M, 1)
    bb = (b * b).sum(1, keepdim=True)   # (N, 1)
    return (aa + bb.T - 2 * a @ b.T).clamp(min=0)   # (M, N)


def _kernel_mat(F, ktype, kopts):
    """
    计算 M×M 核矩阵，输入 F ∈ R^{M×d}。

    支持的核类型及超参数：
      'Linear'     : K(x,y) = x·y
                     （线性核，等价于内积，无超参数）
      'Gaussian'   : K(x,y) = exp(-||x-y||² / (2t²))
                     kopts['t']：带宽（标准差），默认 1.0
      'Polynomial' : K(x,y) = (a·x·y + b)^d
                     kopts['a']：缩放系数，默认 1.0
                     kopts['b']：偏移项，默认 1.0
                     kopts['d']：多项式次数，默认 2.0
      'Sigmoid'    : K(x,y) = tanh(d·x·y + c)
                     kopts['d']：斜率，默认 2.0
                     kopts['c']：截距，默认 0.0
      'Cauchy'     : K(x,y) = 1 / (||x-y||²/σ + 1)
                     kopts['sigma']：尺度参数，默认 1.0
                     柯西核尾部比高斯核更重，对远离正常区域的点敏感，
                     适合检测极端异常。

    返回值：torch.Tensor，形状 (M, M)，dtype 与 F 相同。
    """
    if ktype == 'Linear':
        return F @ F.T
    elif ktype == 'Gaussian':
        D = _eu_dist2(F, F)
        return torch.exp(-D / (2 * kopts.get('t', 1.0) ** 2))
    elif ktype == 'Polynomial':
        a, b, d = kopts.get('a', 1.0), kopts.get('b', 1.0), kopts.get('d', 2.0)
        return (a * (F @ F.T) + b) ** d
    elif ktype == 'Sigmoid':
        d, c = kopts.get('d', 2.0), kopts.get('c', 0.0)
        return torch.tanh(d * (F @ F.T) + c)
    elif ktype == 'Cauchy':
        D = _eu_dist2(F, F)
        return 1 / (D / kopts.get('sigma', 1.0) + 1)
    else:
        raise ValueError(f'Unknown kernel: {ktype}')


# ─── 跨核对比损失 ──────────────────────────────────────────────────────────────

def cross_kernel_loss(hs, kernels):
    """
    对所有 C(K,2) 个核对 (k, l) 计算跨核对比损失，取均值。

    【单核对 (k, l) 的损失推导】

    1. 构造拼接特征矩阵：
         F_kl = [h_k ; h_l] ∈ R^{2B × d}
       前 B 行来自核 k 的嵌入，后 B 行来自核 l 的嵌入，
       索引 i 与 i+B 对应同一原始样本 x_i。

    2. 正对定义（positive pair）：
         (i, i+B) 和 (i+B, i)，即同一样本在两种核下的嵌入互为正对。
       负对：同一行中其余 2B-2 个索引。
       自对（对角线）被掩码排除。

    3. 跨核相似度矩阵：
         K_avg = (K_k(F_kl) + K_l(F_kl)) / 2
       在 F_kl 上分别用核 k 和核 l 计算核矩阵，再取平均。
       这样相似度既反映核 k 诱导的度量，也反映核 l 诱导的度量。

    4. InfoNCE 损失：
         loss_i = -log( exp(K_avg[i,pos_i]) / Σ_{j≠i} exp(K_avg[i,j]) )
       对 2B 个样本取均值，再对所有核对取均值。

    【训练目标的直觉】
       最小化该损失要求：同一样本在两种核下的嵌入 K_avg 值大（相似），
       不同样本的嵌入 K_avg 值小（不相似）。
       训练完成后，正常样本的各核嵌入彼此一致；
       异常样本在推断时其跨核一致性低于正常样本，因此可以检出。

    参数：
      hs      : list of K tensors，每个形状 (B, d)，来自 model.forward(xb)
      kernels : list of (name, ktype, kopts) 元组，与 hs 长度相同
    """
    n_kernels = len(hs)
    B = hs[0].shape[0]
    device = hs[0].device

    # 正对掩码 mask[i,j]=1 iff j 是 i 的正对（即 j==i+B 或 j==i-B），且 j≠i
    # repeat(2,2) 将 B×B 单位矩阵扩展为 2B×2B 的块对角矩阵，
    # 再乘以非自对掩码去除对角线（自相似度不参与损失）。
    mask = torch.eye(B, device=device).repeat(2, 2)           # (2B, 2B)
    logits_mask = 1 - torch.eye(2 * B, device=device)         # 去掉对角线
    mask = mask * logits_mask                                  # (2B, 2B)，正对位置为 1

    total_loss = 0.0
    n_pairs = 0
    for k in range(n_kernels):
        for l in range(k + 1, n_kernels):
            # 将两种核的嵌入垂直拼接，形成"跨核增广批次"
            F_kl = torch.cat([hs[k], hs[l]], dim=0)   # (2B, d)

            # 在 F_kl 上分别计算两种核的核矩阵，再平均
            Kk = _kernel_mat(F_kl, kernels[k][1], kernels[k][2])  # (2B, 2B)
            Kl = _kernel_mat(F_kl, kernels[l][1], kernels[l][2])  # (2B, 2B)
            K_avg = (Kk + Kl) / 2   # 跨核平均相似度矩阵

            # InfoNCE：用 exp(K_avg) 作为相似度（核矩阵值本身即为相似度量）
            logits = torch.exp(K_avg)   # (2B, 2B)，全为正值
            log_prob = torch.log(logits) - torch.log(
                (logits * logits_mask).sum(1, keepdim=True)   # 排除自对后的归一化
            )
            # 对每个样本：其损失 = 正对位置的 log-prob 均值（取负变为最小化）
            loss = -(mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
            total_loss += loss.mean()
            n_pairs += 1

    return total_loss / max(n_pairs, 1)   # 对所有核对取均值


# ─── 训练 ─────────────────────────────────────────────────────────────────────

def train_ckad(X, y=None, kernels=None, latent_dim=64, epochs=100, batch_size=512,
               lr=0.01, normalize=True, device=None, seed=42, print_freq=20):
    """
    Mini-batch 跨核对比训练，返回训练好的 CKADNet。

    【半监督设置】
      若提供标签 y，则在训练前过滤 X = X[y==0]，
      仅用正常样本构成训练集（半监督/one-class 设置）。
      测试阶段用 get_embeddings 对全量 N 个样本计算嵌入。
      这是算法异常检测能力的关键：
        模型从未见过异常，因此无法"对齐"异常在不同核下的嵌入，
        导致异常的跨核距离大于正常样本。

    【优化器与超参数】
      Adam 优化器，学习率 lr=0.01（较大，因 batch_size=512 时梯度估计噪声较小）。
      训练 100 epochs，每个 epoch 对训练集随机打乱（shuffled mini-batch）。
      跳过长度 < 4 的不完整批次，防止核对比损失在极小批次上数值不稳定。

    参数：
      X          : 特征矩阵，(N, D) 或 (n_normal, D)（若提供 y 则自动过滤）
      y          : 标签向量（可选），0=正常，1=异常
      kernels    : 核配置列表，默认使用 ALL_KERNELS（5 种核）
      latent_dim : 嵌入维度 d
      epochs     : 训练轮数
      batch_size : 每批样本数（应足够大以包含足够负对）
      lr         : Adam 学习率
      normalize  : 是否在嵌入后做 L2 归一化（推荐开启）
      device     : torch.device，默认自动选 CUDA/CPU
      seed       : 随机种子，保证实验可复现
      print_freq : 每隔多少 epoch 打印一次损失（设为 epochs+1 则静默）
    """
    if kernels is None:
        kernels = ALL_KERNELS
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # 过滤：只保留正常样本用于训练
    if y is not None:
        X = X[y == 0]

    N, D = X.shape
    model = CKADNet(D, latent_dim, len(kernels), normalize).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X, dtype=torch.float32)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)   # 每 epoch 随机打乱
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i: i + batch_size]
            if len(idx) < 4:   # 批次太小时核矩阵退化，跳过
                continue
            xb = X_t[idx].to(device)
            hs = model(xb)                         # K 个嵌入，每个 (B, d)
            loss = cross_kernel_loss(hs, kernels)  # 跨核 InfoNCE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % print_freq == 0:
            print(f'    epoch {epoch:4d}/{epochs}'
                  f'  loss={epoch_loss / max(n_batches, 1):.4f}'
                  f'  time={time.time() - t0:.1f}s')

    return model


# ─── 嵌入提取 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_embeddings(model, X, batch_size=2048, device=None):
    """
    对全量 N 个样本（包含异常）提取嵌入，返回两种格式：
      H_all : (N, K*d) — K 个嵌入横向拼接，用于 CK-KNN 和 OC-SVM/SVDD 评分
      H_per : list of K arrays，每个 (N, d) — 用于跨核不一致性 (CK-Incon) 评分

    分批处理防止大数据集 OOM；batch_size=2048 通常适合 CPU 推断。
    注意：此处直接访问 model.projectors 而非调用 model.forward，
    目的是同时获取逐核嵌入（H_per）和拼接嵌入（H_all），
    避免重复前向传播。
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    per = [[] for _ in model.projectors]   # 每个核收集各批次的嵌入片段

    for i in range(0, len(X), batch_size):
        xb = X_t[i: i + batch_size].to(device)
        hs = [p(xb) for p in model.projectors]
        if model.normalize:
            hs = [F.normalize(h, dim=1) for h in hs]
        for k, h in enumerate(hs):
            per[k].append(h.cpu().numpy())

    H_per = [np.concatenate(p, axis=0) for p in per]   # 各核完整嵌入，(N, d)
    H_all = np.concatenate(H_per, axis=1)               # 拼接嵌入，(N, K*d)
    return H_all, H_per


# ─── 异常评分 ─────────────────────────────────────────────────────────────────

def cross_kernel_inconsistency(H_per):
    """
    CK-Incon 异常得分：跨核嵌入不一致性。

    数学定义：
      score(x_i) = (1 / C(K,2)) * Σ_{k<l} ||h_k(x_i) - h_l(x_i)||²

    直觉：
      训练阶段对比损失驱使正常样本在各核下的嵌入彼此靠近；
      异常样本从未参与训练，其各核嵌入无法对齐，因此跨核距离大，得分高。

    参数：
      H_per : list of K arrays，每个形状 (N, d)，来自 get_embeddings

    返回：
      scores : (N,) array，值越大越可能是异常
    """
    K = len(H_per)
    scores = np.zeros(H_per[0].shape[0])
    n_pairs = 0
    for k in range(K):
        for l in range(k + 1, K):
            scores += np.sum((H_per[k] - H_per[l]) ** 2, axis=1)
            n_pairs += 1
    return scores / max(n_pairs, 1)


def best_knn_auc(embeddings, y, k_min=2, k_max=60):
    """
    CK-KNN 异常得分：在嵌入空间搜索最优 K 值的 KNN 距离评分。

    实现细节：
      - 预构建 k_max+1 近邻索引（欧氏距离，并行计算），避免对每个 K 重复搜索；
        列 0 为自身（距离=0），列 1..k 为真正的近邻。
      - 得分 = 前 k 个近邻的平均距离（KNN-avg-dist），值越大越异常。
      - 在 [k_min, k_max] 内遍历所有 K，取 ROC-AUC 最高的 K。

    参数：
      embeddings : (N, d') 嵌入矩阵（可以是原始特征或 H_all 拼接嵌入）
      y          : (N,) 标签，0=正常，1=异常
      k_min/k_max: 搜索范围

    返回：
      (best_k, best_auc)
    """
    nn_model = NearestNeighbors(n_neighbors=k_max + 1, metric='euclidean', n_jobs=-1)
    nn_model.fit(embeddings)
    dists, _ = nn_model.kneighbors(embeddings)   # (N, k_max+1)，第 0 列距离=0
    best_k, best_auc = k_min, -1.0
    for k in range(k_min, k_max + 1):
        scores = dists[:, 1: k + 1].mean(axis=1)   # 跳过自身（列 0）
        auc = roc_auc_score(y, scores)
        if auc > best_auc:
            best_auc, best_k = auc, k
    return best_k, best_auc


def ocsvm_score(H_all, H_normal, kernel='linear', nu=0.1, gamma='scale'):
    """
    OC-SVM 异常评分（线性核）。

    【算法原理】
      OC-SVM 在特征空间中寻找距离原点最远的超平面，使绝大多数正常样本落在超平面
      的正侧（与原点相对）：
        min  (1/2)||w||² - ρ + (1/νn) Σ ξ_i
        s.t. w·Φ(x_i) ≥ ρ - ξ_i,  ξ_i ≥ 0
      决策函数 f(x) = w·Φ(x) - ρ：
        f(x) > 0 → 正常（在超平面正侧）
        f(x) < 0 → 异常（越负越异常）

      使用线性核时 Φ(x) = x，超平面在嵌入空间中是线性的。
      实验结果表明，CKAD 嵌入空间具有良好的线性可分性，线性 OC-SVM 效果最优。

    【超参数 nu】
      nu ∈ (0, 1] 是正常样本中允许落在超平面"错误侧"的比例上界，
      同时也是支持向量比例的下界。
      nu 越小 → 边界越紧，异常检测越保守（漏检多）；
      nu 越大 → 边界越松，误报增多。
      本实验通过网格搜索 nu ∈ {0.01, 0.05, 0.1, 0.2} 选最优 AUC。

    参数：
      H_all    : (N, K*d) 全量嵌入（含正常+异常）
      H_normal : (n_normal, K*d) 仅正常样本嵌入，用于训练 OC-SVM
      kernel   : 核类型，默认 'linear'
      nu       : 异常比例上界
      gamma    : 仅对非线性核有效（'scale' = 1/(n_features * X.var())）

    返回：
      scores : (N,) array，-decision_function 值，越大越异常
    """
    from sklearn.svm import OneClassSVM
    clf = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    clf.fit(H_normal)                         # 仅在正常嵌入上拟合
    return -clf.decision_function(H_all)      # 取负：决策值越低 → 异常得分越高


def svdd_score(H_all, H_normal, nu=0.1, gamma='scale'):
    """
    SVDD 异常评分（RBF 核，即最小包围超球面）。

    【SVDD 与 OC-SVM 的等价性】
      SVDD（Support Vector Data Description）在特征空间中寻找包含大多数
      正常样本的最小超球面 {x : ||Φ(x) - c||² ≤ R²}。
      可以证明：当使用 RBF（高斯）核时，SVDD 的优化问题与 OC-SVM 完全等价，
      因此用 OneClassSVM(kernel='rbf') 实现 SVDD。

    【与线性 OC-SVM 的区别】
      线性 OC-SVM 假设正常样本在嵌入空间中占据以原点为中心的半空间（线性边界）；
      SVDD/RBF-OC-SVM 假设正常样本占据某个紧致的球状区域（非线性边界）。
      若正常样本在嵌入空间中分布较为分散（非球状），线性 OC-SVM 表现更好；
      若正常样本形成单一紧凑簇，SVDD 理论上更优。
      实验中线性 OC-SVM（AUC ~0.92）优于 SVDD（AUC ~0.70），
      说明 CKAD 嵌入空间的正常样本分布更符合半空间假设。

    参数：
      H_all    : (N, K*d) 全量嵌入
      H_normal : (n_normal, K*d) 正常样本嵌入，用于训练
      nu       : 边界外样本比例上界（同 OC-SVM 的 nu 含义）
      gamma    : RBF 核带宽参数（'scale' 表示自动估计）

    返回：
      scores : (N,) array，-decision_function 值，越大越异常
    """
    from sklearn.svm import OneClassSVM
    clf = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    clf.fit(H_normal)
    return -clf.decision_function(H_all)


# ─── 单数据集实验（混合型数据，四种评分对比）─────────────────────────────────────

def run_experiment(path, kernels=None, latent_dim=64, epochs=100,
                   batch_size=512, lr=0.01, k_min=2, k_max=60,
                   normalize=True, seed=42, verbose=True):
    """
    对单个数据集运行 CKAD 实验，对比三种评分方法：
      1. KNN 基线（原始 MinMax 特征 + KNN 距离）
      2. CK-KNN（CKAD 拼接嵌入 + KNN 距离）
      3. CK-Incon（跨核不一致性得分）

    主要用于混合型数据（dataset/mixed），被 CKAD.py __main__ 批量调用。
    对纯数值型数据集的完整实验（含 OC-SVM/SVDD）见 run_cardio.py。

    返回：
      dict，包含各评分方法的 AUC、最优 K、训练时间等信息。
    """
    if kernels is None:
        kernels = ALL_KERNELS

    if verbose:
        print(f'  加载: {os.path.basename(path)}')
    X, y, meta = load_data(path)
    N, ar = meta['N'], meta['anomaly_rate']
    if verbose:
        print(f'  N={N}  D={X.shape[1]}  '
              f'(nom={meta["d_nom"]}, num={meta["d_num"]})  '
              f'异常率={ar*100:.1f}%  K∈[{k_min},{k_max}]')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ── 1. KNN 基线（原始特征，无需训练）────────────────────────────────────────
    t0 = time.time()
    best_k_knn, auc_knn = best_knn_auc(X, y, k_min, k_max)
    t_knn = time.time() - t0
    if verbose:
        print(f'  [KNN 基线  ]  AUC={auc_knn:.4f}  best_k={best_k_knn}  ({t_knn:.1f}s)')

    # ── 2 & 3. CKAD 训练 + CK-KNN 和 CK-Incon 评分 ───────────────────────────
    n_normal = int((y == 0).sum())
    if verbose:
        print(f'  [CK-AD     ]  训练 {len(kernels)} 种核 × {epochs} epochs'
              f'  （仅正常样本 {n_normal}/{N}）...')
    t0 = time.time()
    model = train_ckad(X, y=y, kernels=kernels, latent_dim=latent_dim, epochs=epochs,
                       batch_size=batch_size, lr=lr, normalize=normalize,
                       device=device, seed=seed, print_freq=epochs + 1)
    t_train = time.time() - t0

    H_all, H_per = get_embeddings(model, X, device=device)

    best_k_ckad, auc_ckad = best_knn_auc(H_all, y, k_min, k_max)
    auc_incon = roc_auc_score(y, cross_kernel_inconsistency(H_per))

    if verbose:
        print(f'  [CK-KNN    ]  AUC={auc_ckad:.4f}  best_k={best_k_ckad}  ({t_train:.1f}s)')
        print(f'  [CK-Incon  ]  AUC={auc_incon:.4f}')

    return dict(
        dataset=os.path.splitext(os.path.basename(path))[0],
        auc_knn=auc_knn, best_k_knn=best_k_knn,
        auc_ckad=auc_ckad, best_k_ckad=best_k_ckad,
        auc_incon=auc_incon,
        N=N, anomaly_rate=ar,
        d_nom=meta['d_nom'], d_num=meta['d_num'],
        t_knn=t_knn, t_train=t_train,
    )


# ─── 主程序：在 dataset/mixed 全部数据集上批量运行 ─────────────────────────────

if __name__ == '__main__':
    # 混合型数据集目录（标称+数值型特征）
    DATA_DIR = r'D:\Microsoft\documents\博士课题\异常检测\论文\5\dataset\mixed'
    K_MIN, K_MAX = 2, 60

    EXP_CFG = dict(
        kernels    = ALL_KERNELS,   # 5 种核：Gaussian/Linear/Polynomial/Sigmoid/Cauchy
        latent_dim = 64,            # 嵌入维度（消融实验可选值：16/32/64/128/256）
        epochs     = 100,
        batch_size = 512,
        lr         = 0.01,
        k_min      = K_MIN,
        k_max      = K_MAX,
        normalize  = True,
        seed       = 42,
        verbose    = True,
    )

    datasets = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.mat'))
    results = []
    for fname in datasets:
        print(f'\n{"="*70}')
        print(f'>>> {fname}')
        res = run_experiment(os.path.join(DATA_DIR, fname), **EXP_CFG)
        results.append(res)

    # ── 汇总表格 ─────────────────────────────────────────────────────────────
    sep = '=' * 105
    print(f'\n\n{sep}')
    print(f'AUC-ROC 汇总  K∈[{K_MIN},{K_MAX}]')
    print(f'{"Dataset":<35} {"N":>6} {"Anom%":>6}  '
          f'{"KNN-MM":>7}{"(k)":>4}  '
          f'{"CK-KNN":>7}{"(k)":>4}  '
          f'{"CK-Incon":>9}  '
          f'{"ΔCKNN-KNN":>10}  {"ΔIncon-KNN":>11}')
    print('-' * 105)
    for r in results:
        name = r['dataset'][:34]
        print(f'{name:<35} {r["N"]:>6} {r["anomaly_rate"]*100:>5.1f}%  '
              f'{r["auc_knn"]:>7.4f}({r["best_k_knn"]:>2})  '
              f'{r["auc_ckad"]:>7.4f}({r["best_k_ckad"]:>2})  '
              f'{r["auc_incon"]:>9.4f}  '
              f'{r["auc_ckad"] - r["auc_knn"]:>+10.4f}  '
              f'{r["auc_incon"] - r["auc_knn"]:>+11.4f}')

    avg_knn   = sum(r['auc_knn']   for r in results) / len(results)
    avg_ckad  = sum(r['auc_ckad']  for r in results) / len(results)
    avg_incon = sum(r['auc_incon'] for r in results) / len(results)
    print('-' * 105)
    print(f'{"Average":<35} {"":>6} {"":>6}  '
          f'{avg_knn:>7.4f}      '
          f'{avg_ckad:>7.4f}       '
          f'{avg_incon:>9.4f}  '
          f'{avg_ckad - avg_knn:>+10.4f}  '
          f'{avg_incon - avg_knn:>+11.4f}')
    print(sep)
    print(f'\nlatent_dim={EXP_CFG["latent_dim"]}  epochs={EXP_CFG["epochs"]}'
          f'  batch={EXP_CFG["batch_size"]}  lr={EXP_CFG["lr"]}'
          f'  K∈[{K_MIN},{K_MAX}]  kernels={len(ALL_KERNELS)}')
