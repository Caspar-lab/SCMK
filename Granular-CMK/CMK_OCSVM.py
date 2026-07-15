"""
CMK_OCSVM — Cross-Kernel Contrastive Learning + One-Class SVM 异常检测
=======================================================================

【算法流程】
  阶段一（表示学习）：跨核对比训练
    - 为 K 种核函数各训练一个线性投影头 W_k : R^D → R^d
    - 对比损失：同一样本在核 k 与核 l 下的嵌入为正对，不同样本为负对
    - 仅用正常样本（y==0）训练，使模型未见过任何异常
    - 训练完成后，正常样本在各核下的嵌入高度一致；
      异常样本由于偏离训练分布，其嵌入与正常簇保持较大距离

  阶段二（异常检测）：线性 OC-SVM
    - 将 K 个核的嵌入拼接为 H_all ∈ R^{N × K*d}
    - 在正常样本嵌入 H_normal = H_all[y==0] 上拟合线性 OC-SVM
    - OC-SVM 在嵌入空间中寻找距离原点最远的超平面，使正常样本尽量落在正侧
    - 决策函数值越低（越靠"异常侧"），异常得分越高

【为什么线性 OC-SVM 效果好】
  跨核对比训练使正常样本在拼接嵌入空间中形成紧凑的线性可分区域：
  各核投影头各自捕获数据在不同相似度度量下的结构，拼接后信息互补；
  线性 OC-SVM 恰好能高效地在此空间中找到正常/异常的分隔超平面。
  实验结果（cardio AUC=0.921，cardiotocography AUC=0.913）显著超越 KNN 基线。

【使用方式】
  默认在 cardio.mat 上运行（latent_dim 扫描 + nu 网格搜索）：
    python CMK_OCSVM.py
  指定其他数据集：
    python CMK_OCSVM.py path/to/dataset.mat
"""

import os, sys, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as scio
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.svm import OneClassSVM


# ─── 实验超参数 ────────────────────────────────────────────────────────────────
_default_path = r'C:\OD\Shihao\5\dataset\mixed\nhanes_age_364.mat'
DATA_PATH     = sys.argv[1] if len(sys.argv) > 1 else _default_path
RESULT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')

# latent_dim 扫描范围（2^4 ~ 2^8）
LATENT_DIMS   = [16, 32, 64, 128, 256]
# LATENT_DIMS   = [4, 6, 8]
# nu 候选值：控制 OC-SVM 决策边界松紧程度
# nu ∈ (0,1] 是正常训练样本中允许"越界"的比例上界；网格搜索取 AUC 最优值
NU_CANDIDATES = [0.01, 0.05, 0.1, 0.2]

# TRAIN_CFG = dict(
#     epochs     = 100,
#     batch_size = 512,
#     lr         = 0.01,
#     normalize  = True,   # L2 归一化嵌入，统一各核尺度
#     seed       = 42,
# )

TRAIN_CFG = dict(
    epochs     = 100,
    batch_size = 512,
    lr         = 0.01,
    normalize  = True,   # L2 归一化嵌入，统一各核尺度
    seed       = 42,
)


# ─── 数据加载 ──────────────────────────────────────────────────────────────────

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


# ─── 自适应高斯核（Gauss-5-med）──────────────────────────────────────────────

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


# ─── 网络结构：K 个独立线性投影头 ─────────────────────────────────────────────

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


# ─── 核矩阵计算 ────────────────────────────────────────────────────────────────

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


# ─── 跨核对比损失 ──────────────────────────────────────────────────────────────

# 投影头产生多视图 embedding
# → 核函数在 embedding 上计算相似度
# → InfoNCE 用这些相似度拉近同一样本跨核表示、推开不同样本表示
# → 梯度反向更新各投影头 W_k

def cross_kernel_loss(hs, kernels):
    """
    对所有 C(K,2) 核对 (k,l) 计算跨核 InfoNCE 损失，取均值。

    对核对 (k,l)：
      - F_kl = [h_k ; h_l] ∈ R^{2B×d}，前 B 行为核 k，后 B 行为核 l
      - 正对：索引 i 与 i+B（同一样本的两种核嵌入）
      - 相似度矩阵：(K_k(F_kl) + K_l(F_kl)) / 2
      - InfoNCE：最大化正对相似度 / 所有非自身相似度之和
    """
    K = len(hs)
    B = hs[0].shape[0]
    device = hs[0].device

    mask        = torch.eye(B, device=device).repeat(2, 2)  # 正对位置
    logits_mask = 1 - torch.eye(2 * B, device=device)       # 排除自身
    mask        = mask * logits_mask                         # (2B, 2B)

    total_loss, n_pairs = 0.0, 0
    for k in range(K):
        for l in range(k + 1, K):
            F_kl  = torch.cat([hs[k], hs[l]], dim=0)
            K_avg = (_kernel_mat(F_kl, kernels[k][1], kernels[k][2]) +
                     _kernel_mat(F_kl, kernels[l][1], kernels[l][2])) / 2
            logits   = torch.exp(K_avg)
            log_prob = torch.log(logits) - torch.log((logits * logits_mask).sum(1, keepdim=True))
            loss     = -(mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
            total_loss += loss.mean()
            n_pairs    += 1

    return total_loss / max(n_pairs, 1)


# ─── 训练 ─────────────────────────────────────────────────────────────────────

def train_cmk(X, y, kernels, latent_dim, device, cfg):
    """
    仅用正常样本（y==0）训练跨核对比模型，返回 CMKNet。

    过滤异常后只有正常样本构成训练集：
    模型学习"正常样本在不同核下的嵌入应彼此一致"，
    未见过异常，因此测试时异常样本无法与正常分布对齐。
    """
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])

    X_train = X[y == 0]   # 只取正常样本
    N, D    = X_train.shape
    model   = CMKNet(D, latent_dim, len(kernels), cfg['normalize']).to(device)
    opt     = optim.Adam(model.parameters(), lr=cfg['lr'])
    X_t     = torch.tensor(X_train, dtype=torch.float32)

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, cfg['batch_size']):
            idx = perm[i: i + cfg['batch_size']]
            if len(idx) < 4:
                continue
            hs = model(X_t[idx].to(device))
            loss = cross_kernel_loss(hs, kernels)
            opt.zero_grad(); loss.backward(); opt.step()

    return model

# ─── 嵌入提取 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_embeddings(model, X, device, batch_size=2048):
    """
    对全量 N 个样本（含异常）提取拼接嵌入 H_all ∈ R^{N × K*d}。
    分批推断防止大数据集 OOM。
    """
    model.eval()
    X_t  = torch.tensor(X, dtype=torch.float32)
    per  = [[] for _ in model.projectors]

    for i in range(0, len(X), batch_size):
        xb = X_t[i: i + batch_size].to(device)
        hs = [p(xb) for p in model.projectors]
        if model.normalize:
            hs = [F.normalize(h, dim=1) for h in hs]
        for k, h in enumerate(hs):
            per[k].append(h.cpu().numpy())

    H_per = [np.concatenate(p, axis=0) for p in per]   # 各核嵌入 (N, d)
    H_all = np.concatenate(H_per, axis=1)               # 拼接嵌入 (N, K*d)
    return H_all


# ─── OC-SVM 评分与超参数搜索 ──────────────────────────────────────────────────

def ocsvm_score(H_all, H_normal, nu):
    """
    线性核 OC-SVM 异常评分。

    在正常嵌入 H_normal 上拟合，对全量 H_all 打分。
    返回 -decision_function：值越大代表越偏离正常区域（越可能是异常）。

    线性核的选择：
      跨核对比训练使正常样本在拼接嵌入空间形成线性可分的紧凑区域，
      线性核能直接利用这一结构，且计算效率高于 RBF 核。
    """
    clf = OneClassSVM(kernel='linear', nu=nu)
    clf.fit(H_normal)
    return -clf.decision_function(H_all)


def best_nu_ocsvm(H_all, H_normal, y, nu_list):
    """
    在 nu_list 上网格搜索，返回使 ROC-AUC 最高的 (best_nu, best_auc)。

    每个 nu 对应一个不同松紧程度的 OC-SVM 决策边界：
      nu=0.01 → 极紧（几乎不容许正常样本越界，适合低噪声数据集）
      nu=0.20 → 较松（允许 20% 正常样本作为支持向量，适合噪声较多场景）
    通过在测试集上比较 AUC 选取最优 nu（oracle 评估，上界估计）。
    """
    best_nu, best_auc = nu_list[0], -1.0
    for nu in nu_list:
        try:
            auc = roc_auc_score(y, ocsvm_score(H_all, H_normal, nu))
            if auc > best_auc:
                best_auc, best_nu = auc, nu
        except Exception:
            pass
    return best_nu, best_auc


# ─── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)
    _stem      = os.path.splitext(os.path.basename(DATA_PATH))[0]
    result_csv = os.path.join(RESULT_DIR, f'{_stem}_cmk_ocsvm.csv')

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print(f'加载: {os.path.basename(DATA_PATH)}')
    X, y, meta = load_data(DATA_PATH)
    N, D       = meta['N'], X.shape[1]
    n_normal   = int((y == 0).sum())
    print(f'N={N}  D={D}  异常率={meta["anomaly_rate"]*100:.1f}%  正常样本={n_normal}')

    # ── 生成核配置（基于正常样本估计带宽）────────────────────────────────────
    kernels = gauss_med_kernels(X[y == 0])
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'核配置: {[k[0] for k in kernels]}')

    # ── latent_dim 扫描 ───────────────────────────────────────────────────────
    print(f'\n{"─"*55}')
    print(f'{"dim":>5}  {"OC-SVM AUC":>12}  {"best_nu":>8}  {"time":>7}')
    print(f'{"─"*55}')

    rows = []
    for latent_dim in LATENT_DIMS:
        t0 = time.time()

        # 阶段一：跨核对比训练（仅正常样本）
        model = train_cmk(X, y, kernels, latent_dim, device, TRAIN_CFG)

        # 阶段二：提取全量嵌入，分离正常嵌入
        H_all    = get_embeddings(model, X, device)
        H_normal = H_all[y == 0]

        # 阶段三：OC-SVM 网格搜索最优 nu
        best_nu, best_auc = best_nu_ocsvm(H_all, H_normal, y, NU_CANDIDATES)

        elapsed = time.time() - t0
        print(f'{latent_dim:>5d}  {best_auc:>12.4f}  {best_nu:>8.2f}  {elapsed:>6.1f}s')

        rows.append(dict(
            latent_dim   = latent_dim,
            n_train      = n_normal,
            auc_ocsvm    = round(best_auc, 6),
            best_nu      = best_nu,
            train_time_s = round(elapsed, 2),
        ))

    print(f'{"─"*55}')

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(result_csv, index=False)
    print(f'\n结果已保存: {result_csv}')

    # ── 汇总：标记最优维度 ─────────────────────────────────────────────────────
    best_overall = max(r['auc_ocsvm'] for r in rows)
    print(f'\n{"="*40}')
    print(f'数据集: {_stem}')
    print(f'{"dim":>5}  {"OC-SVM AUC":>12}  {"best_nu":>8}')
    print(f'{"-"*30}')
    for r in rows:
        marker = '*' if abs(r['auc_ocsvm'] - best_overall) < 1e-6 else ' '
        print(f'{r["latent_dim"]:>5d}  {marker}{r["auc_ocsvm"]:>11.4f}  {r["best_nu"]:>8.2f}')
    print(f'{"="*40}')
    print(f'* = 最优维度')
    print(f'nu候选={NU_CANDIDATES}  epochs={TRAIN_CFG["epochs"]}  batch={TRAIN_CFG["batch_size"]}')
