"""
Granular-CMK (GCMK)
将粒球生成与CMK的W训练相结合：
  1. 特征切半 → 两视图
  2. 共享粒球划分（在全特征空间生成）→ 各视图球心矩阵
  3. FCNet (W) 在球心上训练（加权对比核损失）
  4. 聚类标签通过球成员关系传播回原始样本
"""

import os, sys, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn
import scipy.io as scio
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GB import getGranularBall


# ─── 聚类评价指标 ─────────────────────────────────────────────────────────────

def accuracy_score(y_true, y_pred):
    cm = metrics.cluster.contingency_matrix(y_true, y_pred)
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum() / cm.sum()

def purity_score(y_true, y_pred):
    cm = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / cm.sum()

def nmi_score(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)

def cluster_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred), nmi_score(y_true, y_pred), purity_score(y_true, y_pred)


# ─── 视图构造 ──────────────────────────────────────────────────────────────────

def split_to_views(X):
    """将单视图特征矩阵 (N, D) 对半切分为两视图 [(N, D//2), (N, D-D//2)]。"""
    D = X.shape[1]
    return [X[:, :D // 2], X[:, D // 2:]]


# ─── 粒球生成与球心计算 ────────────────────────────────────────────────────────

def build_granular_balls(Xs_np):
    """
    在全特征空间（各视图L2归一化后拼接）生成共享粒球划分，
    再分别在各视图特征空间计算球心。

    参数:
        Xs_np : list of np.ndarray, 每个形状 (N, D_v)

    返回:
        centers_list : list of np.ndarray, 每个形状 (M, D_v) — 各视图球心
        weights      : np.ndarray (M,) — 每个球包含的样本数
        gb_index     : list of np.ndarray — 每个球的原始样本索引
    """
    # 构造参考空间：各视图L2归一化后拼接
    parts = []
    for X in Xs_np:
        norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
        parts.append((X / norms).astype(np.float64))
    ref = np.concatenate(parts, axis=1)  # (N, sum D_v)

    # 在参考空间生成粒球划分
    _, weights, gb_index = getGranularBall(ref)
    M = len(gb_index)

    # 各视图球心 = 该球内样本的均值
    centers_list = []
    for X in Xs_np:
        C = np.zeros((M, X.shape[1]), dtype=np.float32)
        for i, idx in enumerate(gb_index):
            C[i] = X[idx].mean(axis=0)
        centers_list.append(C)

    return centers_list, weights.astype(np.float32), gb_index


# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_data(args):
    """
    加载数据，构造视图，生成粒球。

    支持两种模式（由 args.view_mode 控制）：
      'split'   : 从单视图 .mat 中读取 X，对半切分为两视图（异常检测场景）
      'multiview': 从多视图 .mat 中读取 X（含多个视图的数组），直接使用

    返回:
        Cs       : list of Tensor (M, D_v) — 各视图球心（在 device 上）
        weights  : Tensor (M,) — 归一化球权重（在 device 上）
        gb_index : list of np.ndarray — 每球原始样本索引
        gt       : np.ndarray (N,) — 原始样本真实标签
        num_class: int
        feat_dims: list of int
    """
    data = scio.loadmat(os.path.join(args.data_dir, args.data_name + '.mat'))
    gt = data['gt'].squeeze()
    num_class = int(np.unique(gt).shape[0])

    if args.view_mode == 'split':
        # 单视图，按列对半切分
        X = data['X'].astype(np.float32)
        if X.ndim == 1:
            X = X.squeeze()
        # 支持 (N, D) 或 (D, N) 格式
        if X.shape[0] == gt.shape[0]:
            Xs_np = split_to_views(X)
        else:
            Xs_np = split_to_views(X.T)
    else:
        # 多视图格式，与 CMK 一致：data['X'] 是 object 数组，每元素 (D_v, N)
        Xs_raw = data['X'].squeeze().tolist()
        Xs_np = [np.asarray(v, dtype=np.float32).T for v in Xs_raw]  # 每个 (N, D_v)

    N = Xs_np[0].shape[0]
    print(f'  样本数 N={N}, 视图数={len(Xs_np)}, '
          f'特征维度={[X.shape[1] for X in Xs_np]}')

    # 粒球生成
    print(f'  正在生成粒球 ...')
    centers_list, weights, gb_index = build_granular_balls(Xs_np)
    M = centers_list[0].shape[0]
    print(f'  共生成 {M} 个粒球，压缩比 {N/M:.1f}x')

    # 转为 Tensor
    feat_dims = []
    Cs = []
    for C in centers_list:
        Cs.append(torch.tensor(C, dtype=torch.float32).to(args.device))
        feat_dims.append(C.shape[1])

    # 归一化权重（和为1）
    w = torch.tensor(weights / weights.sum(), dtype=torch.float32).to(args.device)

    # 同时返回原始样本 Tensor（用于 checkpoint 时在全量数据上评估，与 CMK 公平对比）
    Xs = [torch.tensor(X, dtype=torch.float32).to(args.device) for X in Xs_np]

    return Cs, w, gb_index, gt, num_class, feat_dims, Xs


# ─── 投影网络 FCNet ────────────────────────────────────────────────────────────

class FCNet(nn.Module):
    """各视图独立线性投影 W_v: D_v → latent_dim（无偏置）。"""

    def __init__(self, feat_dims, latent_dim=64, normalize=True):
        super().__init__()
        self.num_view = len(feat_dims)
        self.normalize = normalize
        for i, d in enumerate(feat_dims):
            setattr(self, f'fc_{i}', nn.Linear(d, latent_dim, bias=False))

    def forward(self, Cs):
        out = []
        for i in range(self.num_view):
            h = getattr(self, f'fc_{i}')(Cs[i])
            if self.normalize:
                h = F.normalize(h, dim=1)
            out.append(h)
        return out


# ─── 加权对比核损失（含聚类对齐项）────────────────────────────────────────────

class GranularConLoss(nn.Module):
    """
    在球心上计算加权 CMK 对比核损失。

    正样本对：不同视图中相同球索引的球心。
    每个球的损失按其大小（权重 w_i）加权。
    当 trade_off > 0 时，附加核 k-means 对齐损失（CMKKM 阶段）。
    """

    def __init__(self, kernel_options, weights, num_class, device):
        super().__init__()
        self.kernel_options = kernel_options
        self.num_class = num_class
        self.device = device
        self.w = weights  # (M,) 归一化权重，和为1

    # ── 核函数 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _eu_dist2(a, b):
        """逐对欧氏距离平方，数值稳定。"""
        aa = (a * a).sum(1, keepdim=True)
        bb = (b * b).sum(1, keepdim=True)
        return (aa + bb.T - 2 * a @ b.T).clamp(min=0)

    def _kernel(self, F_cat):
        opts = self.kernel_options
        ktype = opts['type']
        if ktype == 'Gaussian':
            D = self._eu_dist2(F_cat, F_cat)
            return torch.exp(-D / (2 * opts['t'] ** 2))
        elif ktype == 'Linear':
            return F_cat @ F_cat.T
        elif ktype == 'Polynomial':
            return (opts['a'] * (F_cat @ F_cat.T) + opts['b']) ** opts['d']
        elif ktype == 'Sigmoid':
            return torch.tanh(opts['d'] * (F_cat @ F_cat.T) + opts['c'])
        elif ktype == 'Cauchy':
            D = self._eu_dist2(F_cat, F_cat)
            return 1 / (D / opts['sigma'] + 1)
        else:
            raise NotImplementedError(f'未知核类型: {ktype}')

    # ── 前向 ────────────────────────────────────────────────────────────────

    def forward(self, features_balls, features_full=None):
        """
        参数:
            features_balls : list of Tensor (M, latent_dim) — 粒球球心嵌入，用于对比损失
            features_full  : list of Tensor (N, latent_dim) — 全量样本嵌入，用于对齐损失
                             CMK 阶段传 None，CMKKM 阶段传入以在 N 样本尺度计算 loss_extra
        返回:
            loss_con   — 加权对比损失（球心）
            loss_extra — 核 k-means 对齐损失（全量样本，CMK 阶段为 0）
            K_gb_np    — (V*M, V*M) 球心核矩阵 numpy
            H_gb       — (M, num_class) 球心嵌入特征向量
            H_full     — (N, num_class) 全量样本特征向量，CMK 阶段为 None
        """
        num_view = len(features_balls)
        M = features_balls[0].shape[0]
        F_ball_cat = torch.cat(features_balls, dim=0)  # (V*M, latent_dim)

        # ── 正样本掩码（球心维度）───────────────────────────────────────────
        mask = torch.eye(M, dtype=torch.float32, device=self.device).repeat(num_view, num_view)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(M * num_view, device=self.device).view(-1, 1), 0
        )
        mask = mask * logits_mask

        # ── 球心核矩阵（保留梯度，用于 loss_con 反向传播）──────────────────
        K_ball = self._kernel(F_ball_cat)  # (V*M, V*M)

        # ── 加权对比损失（在球心上计算）────────────────────────────────────
        logits = torch.exp(K_ball)
        log_prob = torch.log(logits) - torch.log((logits * logits_mask).sum(1, keepdim=True))
        per_sample = -(mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        per_ball = per_sample.view(num_view, M)
        loss_con = (per_ball * self.w.unsqueeze(0)).sum(1).mean()

        # ── 球心 H（监控用，不参与梯度）────────────────────────────────────
        with torch.no_grad():
            kernel_ball_mean = torch.stack(
                [K_ball[v * M:(v + 1) * M, v * M:(v + 1) * M] for v in range(num_view)], dim=2
            ).mean(2)
            _, evecs_ball = torch.linalg.eigh(kernel_ball_mean)
            H_gb = F.normalize(evecs_ball[:, -self.num_class:], dim=1).cpu().numpy()

        # ── 核 k-means 对齐损失（CMKKM 阶段：在全量 N 样本上计算）──────────
        if features_full is not None:
            N = features_full[0].shape[0]
            F_full_cat = torch.cat(features_full, dim=0)  # (V*N, latent_dim)
            K_full = self._kernel(F_full_cat)              # (V*N, V*N)，保留梯度

            kernel_full_mean = torch.stack(
                [K_full[v * N:(v + 1) * N, v * N:(v + 1) * N] for v in range(num_view)], dim=2
            ).mean(2)  # (N, N)

            # H_full 固定（detach），loss_extra 梯度通过 kernel_full_mean → W
            with torch.no_grad():
                _, evecs_full = torch.linalg.eigh(kernel_full_mean)
                H_full_t = evecs_full[:, -self.num_class:]  # (N, num_class)

            loss_extra = (torch.trace(kernel_full_mean) -
                          torch.trace(H_full_t.T @ kernel_full_mean @ H_full_t)) / N
            H_full = F.normalize(H_full_t, dim=1).cpu().numpy()
        else:
            loss_extra = torch.tensor(0.0, device=self.device)
            H_full = None

        return loss_con, loss_extra, K_ball.detach().cpu().numpy(), H_gb, H_full


# ─── 单步训练 ──────────────────────────────────────────────────────────────────

def train_step(Cs, Xs, model, criterion, optimizer, trade_off):
    """
    一个 epoch 的训练。
    CMK 阶段（trade_off=0）：仅用球心做对比损失；
    CMKKM 阶段（trade_off>0）：对比损失在球心上，对齐损失在全量 N 样本上。
    """
    model.train()
    features_balls = model(Cs)
    # CMKKM 阶段才计算全量前向（避免 CMK 阶段额外开销）
    features_full = model(Xs) if trade_off > 0 else None

    loss_con, loss_extra, K_gb_np, H_gb, H_full = criterion(features_balls, features_full)

    loss = loss_con + trade_off * loss_extra
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    num_view, M = len(Cs), Cs[0].shape[0]
    K_gb = np.stack(
        [K_gb_np[v * M:(v + 1) * M, v * M:(v + 1) * M] for v in range(num_view)], axis=2
    )  # (M, M, V)
    return loss_con.item(), loss_extra.item(), K_gb, H_gb, H_full


# ─── 学习率调度（余弦，与 CMK 一致）──────────────────────────────────────────

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch * 3 / args.epochs)) / 2
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ─── 标签传播 ──────────────────────────────────────────────────────────────────

def propagate_labels(ball_labels, gb_index, N):
    """将每个粒球的聚类标签传播至其包含的原始样本。"""
    sample_labels = np.zeros(N, dtype=int)
    for ball_id, idx in enumerate(gb_index):
        sample_labels[idx] = ball_labels[ball_id]
    return sample_labels


# ─── 全量样本评估（公平对比核心）────────────────────────────────────────────────

@torch.no_grad()
def eval_on_full_data(model, Xs, gt, num_class, kernel_options, seed):
    """
    将训练好的 W 作用于全部 N 个原始样本，
    计算 N×N 核矩阵（与 CMK 保存的 K 结构一致），
    并在其上跑 KMeans 评估。

    这是与 CMK 公平对比的关键：两者都在 N 个原始样本的嵌入核上评估。

    返回:
        acc, nmi, pur  — 在 N 个原始样本上的聚类指标
        K_full         — (N, N, V) 各视图 N×N 核矩阵（numpy，与 CMK 保存格式相同）
        H_full         — (N, num_class) 原始样本嵌入
    """
    model.eval()
    features = model(Xs)  # list of (N, latent_dim)

    num_view = len(features)
    N = features[0].shape[0]
    F_cat = torch.cat(features, dim=0)  # (V*N, latent_dim)

    # 复用 GranularConLoss 的核函数逻辑
    dummy = GranularConLoss.__new__(GranularConLoss)
    dummy.kernel_options = kernel_options
    K_big = dummy._kernel(F_cat).cpu().numpy()  # (V*N, V*N)

    # 各视图对角块 → (N, N, V)
    K_full = np.stack(
        [K_big[v * N:(v + 1) * N, v * N:(v + 1) * N] for v in range(num_view)], axis=2
    )

    # 均值核上做特征分解得 H_full
    kernel_mean = torch.tensor(K_full.mean(2), dtype=torch.float32)
    _, evecs = torch.linalg.eigh(kernel_mean)
    H_full = F.normalize(evecs[:, -num_class:], dim=1).numpy()  # (N, num_class)

    km = KMeans(n_clusters=num_class, n_init=10, random_state=seed).fit(H_full)
    acc, nmi, pur = cluster_metric(gt, km.labels_)
    return acc, nmi, pur, K_full, H_full


# ─── 主流程 ────────────────────────────────────────────────────────────────────

def main(args):
    seed = 42
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Cs, weights, gb_index, gt, num_class, feat_dims, Xs = load_data(args)
    N, M = gt.shape[0], Cs[0].shape[0]

    model = FCNet(feat_dims, args.latent_dim, args.normalize).to(args.device)
    criterion = GranularConLoss(
        args.kernel_options, weights, num_class, args.device
    ).to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    t_start = time.time()
    accs_gb, nmis_gb, purs_gb = [], [], []    # Track A：球心传播
    accs_full, nmis_full, purs_full = [], [], []  # Track B：全量样本（公平对比）

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        trade_off = 0 if epoch <= args.epochs // 3 else args.trade_off

        # CMK 阶段：loss_extra=0，H_full=None
        # CMKKM 阶段：loss_extra 在全量 N 样本核上计算，H_full 直接可用
        loss_con, loss_extra, K_gb, H_gb, H_full = train_step(
            Cs, Xs, model, criterion, optimizer, trade_off
        )

        # Track A：球心 KMeans → 标签传播到 N 样本
        km_gb = KMeans(n_clusters=num_class, n_init=10, random_state=seed).fit(H_gb)
        sample_labels = propagate_labels(km_gb.labels_, gb_index, N)
        acc_gb, nmi_gb, pur_gb = cluster_metric(gt, sample_labels)
        accs_gb.append(acc_gb); nmis_gb.append(nmi_gb); purs_gb.append(pur_gb)

        # Track B：CMKKM 阶段 H_full 由 train_step 直接返回，无需额外前向
        if H_full is not None:
            km_f = KMeans(n_clusters=num_class, n_init=10, random_state=seed).fit(H_full)
            acc_f, nmi_f, pur_f = cluster_metric(gt, km_f.labels_)
            accs_full.append(acc_f); nmis_full.append(nmi_f); purs_full.append(pur_f)
        else:
            acc_f = nmi_f = pur_f = float('nan')

        if epoch % args.print_freq == 0:
            full_str = f'[Full] ACC:{acc_f:.4f} NMI:{nmi_f:.4f}' if H_full is not None else '[CMK phase]'
            print(f'  . epoch {epoch:4d}, time:{time.time()-t_start:.2f}s, '
                  f'loss_con:{loss_con:.4f}, loss_extra:{loss_extra:.4f} | '
                  f'[GB] ACC:{acc_gb:.4f} | {full_str}')

        # 存档节点（与 CMK 三段策略一致）
        save_file = None
        ktype = args.kernel_options['type']
        if epoch == args.epochs // 3:
            save_file = os.path.join(args.save_path, f'{ktype}_cmk.mat')
        elif epoch == args.epochs * 2 // 3:
            save_file = os.path.join(args.save_path, f'{ktype}_cmkkm_mid.mat')
        elif epoch == args.epochs:
            save_file = os.path.join(args.save_path, f'{ktype}_cmkkm.mat')

        if save_file:
            # CMK 阶段的 checkpoint 需单独计算全量评估
            if H_full is None:
                acc_f, nmi_f, pur_f, K_full, H_full = eval_on_full_data(
                    model, Xs, gt, num_class, args.kernel_options, seed
                )
                accs_full.append(acc_f); nmis_full.append(nmi_f); purs_full.append(pur_f)
                km_f = KMeans(n_clusters=num_class, n_init=10, random_state=seed).fit(H_full)
            else:
                # CMKKM 阶段：K_full 由 eval_on_full_data 补充（用于保存，与 CMK 格式一致）
                _, _, _, K_full, _ = eval_on_full_data(
                    model, Xs, gt, num_class, args.kernel_options, seed
                )

            print(f'  [Checkpoint] [GB-propagate] ACC={acc_gb:.4f} NMI={nmi_gb:.4f} PUR={pur_gb:.4f}')
            print(f'  [Checkpoint] [Full-sample ] ACC={acc_f:.4f}  NMI={nmi_f:.4f}  PUR={pur_f:.4f}')

            scio.savemat(save_file, {
                'accs': accs_gb, 'nmis': nmis_gb, 'purs': purs_gb, 'gt': gt,
                'K': K_full,     # (N, N, V) — 与 CMK 格式相同，可直接用 test_bbcsport.m
                'H': H_full,     # (N, num_class)
                'K_gb': K_gb,    # (M, M, V) — 粒球核
                'H_gb': H_gb,    # (M, num_class)
                'gb_num': M, 'sample_num': N,
            })

    torch.cuda.empty_cache()
    print(f'\n  [GB-propagate] 最佳 ACC={max(accs_gb):.4f}  NMI={max(nmis_gb):.4f}  PUR={max(purs_gb):.4f}')
    if accs_full:
        print(f'  [Full-sample ] 最佳 ACC={max(accs_full):.4f}  NMI={max(nmis_full):.4f}  PUR={max(purs_full):.4f}')


# ─── 默认参数 ──────────────────────────────────────────────────────────────────

def default_args(data_name, normalize=True, latent_dim=128,
                 learning_rate=1.0, epochs=300,
                 view_mode='split', data_dir='./data'):
    """
    view_mode:
      'split'     — 单视图数据，对半切分特征为两视图（默认）
      'multiview' — 已有多视图数据（如 bbcsport_2view.mat）
    """
    args = argparse.ArgumentParser().parse_args([])

    args.kernel_options = {'type': 'Gaussian', 't': 1.0}
    args.normalize = normalize
    args.trade_off = 1
    args.latent_dim = latent_dim
    args.learning_rate = learning_rate
    args.momentum = 0.9
    args.weight_decay = 0
    args.epochs = epochs
    assert args.epochs % 3 == 0, 'epochs 必须能被3整除'
    args.cosine = True
    args.lr_decay_rate = 0.1
    args.lr_decay_epochs = [700, 800, 900]
    args.temperature = 1.0
    args.print_freq = 50
    args.view_mode = view_mode
    args.data_dir = data_dir
    args.data_name = data_name
    args.save_path = os.path.join(
        './save', data_name,
        f'norm_{normalize}', f'dim_{latent_dim}',
        f'lr_{learning_rate}', f'epochs_{epochs}'
    )
    os.makedirs(args.save_path, exist_ok=True)
    return args


if __name__ == '__main__':
    # 示例：在 bbcsport 多视图数据上运行
    args = default_args(
        data_name='bbcsport_2view',
        view_mode='multiview',
        data_dir='../CMK-code_release/data',
        epochs=300,
        learning_rate=1.0,
    )
    args.kernel_options = {'type': 'Gaussian', 't': 1.0}
    main(args)
