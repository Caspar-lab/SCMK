"""
CMK_OCSVM_scatter.py — CMK + 多核散度惩罚（CMKKM风格） + OC-SVM
================================================================

【改进动机】
  原始 CMK 的跨核 InfoNCE 损失只保证"跨核一致性"：
    同一样本 i 在不同投影头 k, l 的嵌入 h_k_i ≈ h_l_i
  但并不强制正常样本在各投影头的嵌入形成紧致簇。
  因此 OC-SVM 面对的正常嵌入可能散布范围较大，边界松弛。

【新增惩罚项：多核散度损失】
  参考 CMKKM（Centroid Multiple Kernel K-Means）单类目标：
    Tr(K · H · H^T) = (1/N²) Σ_{i,j} Σ_k η_k · K_k(h_{k,i}, h_{k,j})
  单类情形 H = (1/√N)·1，均匀权重 η_k = 1/K，使用线性（余弦）核：
    J_CMKKM = (1/K) Σ_k ||μ_k||²，   μ_k = (1/N) Σ_i h_{k,i}
  等价关系（L2 归一化嵌入 + 线性核）：
    (1/N²) Σ_{i,j} h_{k,i}·h_{k,j} = ||μ_k||²
  即：最大化嵌入均值模长 ≡ 最大化正常样本对的平均余弦相似度
      ≡ 最小化正常样本在各潜在核空间中的散度

【总损失】
  L_total = L_cross + λ · L_scatter
  L_cross  : 跨核 InfoNCE（同原 CMK）
  L_scatter: -(1/K) Σ_k ||μ_k||²  （对所有 batch 正常样本求均值中心，取负）

【防坍缩保证】
  L2 归一化将嵌入限制在单位超球面，μ_k 模长有上界 1。
  InfoNCE 在全部嵌入坍缩为同一点时梯度非零（loss = log 2N），
  两项损失的张力自然防止退化解。

【与 CMK_OCSVM.py 的关系】
  仅替换训练函数和配置，其余（数据加载、模型结构、OC-SVM 搜索）完全复用。
"""

import os, sys, time
import numpy as np
import torch
import torch.optim as optim
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CMK_OCSVM import (
    load_data, gauss_med_kernels, CMKNet,
    get_embeddings, cross_kernel_loss,
    LATENT_DIMS, NU_CANDIDATES,
    TRAIN_CFG as _BASE_CFG,
)


# ─── 超参数 ────────────────────────────────────────────────────────────────────

# 在基础配置之上增加 lambda_scatter
TRAIN_CFG = {
    **_BASE_CFG,
    'lambda_scatter': 0,   # 多核散度惩罚强度，0 退化为原始 CMK
}

_default_path = r'C:\OD\Shihao\5\dataset\mixed\nhanes_age_364.mat'
DATA_PATH  = sys.argv[1] if len(sys.argv) > 1 else _default_path
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')


# ─── 多核散度损失 ──────────────────────────────────────────────────────────────

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


# ─── 训练（带散度惩罚） ────────────────────────────────────────────────────────

def train_cmk_scatter(X, y, kernels, latent_dim, device, cfg):
    """
    在正常样本上用 L_total = L_cross + λ·L_scatter 训练 CMKNet。

    与原始 train_cmk 的唯一区别：
      loss = cross_kernel_loss(hs, kernels)
           + cfg['lambda_scatter'] * scatter_loss(hs)

    Args:
        X        : (N, D) float32 特征矩阵
        y        : (N,)   标签，0=正常，1=异常（训练只用 y==0）
        kernels  : gauss_med_kernels 输出的核配置列表
        latent_dim: 嵌入维度 d
        device   : torch.device
        cfg      : 训练配置字典，须含 lambda_scatter 键

    Returns:
        训练好的 CMKNet 模型
    """
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    X_train = X[y == 0]
    N, D    = X_train.shape
    model   = CMKNet(D, latent_dim, len(kernels), cfg['normalize']).to(device)
    opt     = optim.Adam(model.parameters(), lr=cfg['lr'])
    X_t     = torch.tensor(X_train, dtype=torch.float32)
    lam     = cfg.get('lambda_scatter', 0.0)

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, cfg['batch_size']):
            idx = perm[i: i + cfg['batch_size']]
            if len(idx) < 4:
                continue
            hs = model(X_t[idx].to(device))

            L_cross   = cross_kernel_loss(hs, kernels)
            L_scatter = scatter_loss(hs) if lam > 0 else 0.0
            loss      = L_cross + lam * L_scatter

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


# ─── OC-SVM 评估 ──────────────────────────────────────────────────────────────

def eval_ocsvm(model, X, y, device):
    """拼接嵌入 → 网格搜索 nu → 返回 (best_auc, best_nu)。"""
    H_all    = get_embeddings(model, X, device)           # (N, K*d)
    H_normal = H_all[y == 0]
    best_auc, best_nu = -1.0, None
    for nu in NU_CANDIDATES:
        clf  = OneClassSVM(kernel='linear', nu=nu).fit(H_normal)
        scores = -clf.decision_function(H_all)
        auc    = roc_auc_score(y, scores)
        if auc > best_auc:
            best_auc, best_nu = auc, nu
    return best_auc, best_nu


# ─── 核级混合检测器：通过权重 w 退化到原始 OCSVM ──────────────────────────────

def eval_ocsvm_blend(model, X, y, device, w_grid=None, nu_list=None, n_proj=None):
    """
    检测阶段异质核凸组合：K(w) = (1-w)·K_raw + w·K_emb

      K_raw = RBF(原始X)         —— 纯 OCSVM 的核（w=0 退化到此）
      K_emb = H_all·H_allᵀ / K   —— 线性核作用在 CMK 嵌入上（w=1 即现方法）

    半监督：仅用正常样本拟合 precomputed-OCSVM，对全量打分。
    端点精确：w=0 ⇒ 原始 RBF OCSVM；w=1 ⇒ 线性 OCSVM on H_all（常数缩放对 AUC 不变）。

    返回：results 字典 {(w, nu): auc}，外加便捷键
          'best'(最优w,nu,auc)、'w0_best'(纯OCSVM最优nu的auc)、'w1_best'(scatter)。
    """
    from sklearn.metrics.pairwise import rbf_kernel

    w_grid  = w_grid  if w_grid  is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
    nu_list = nu_list if nu_list is not None else NU_CANDIDATES

    H_all = get_embeddings(model, X, device)              # (N, K*d)
    K     = n_proj if n_proj is not None else len(model.projectors)
    nrm   = (y == 0)

    # 两个 Gram 矩阵各算一次
    var_x = float(X.var()) if float(X.var()) > 0 else 1.0
    gamma = 1.0 / (X.shape[1] * var_x)                    # sklearn 'scale'
    K_raw = rbf_kernel(X, X, gamma=gamma)                 # (N, N) ∈ (0,1]
    K_emb = (H_all @ H_all.T) / float(K)                  # (N, N) 线性核，尺度归一

    results = {}
    for w in w_grid:
        Kc = (1.0 - w) * K_raw + w * K_emb
        Kc_fit = Kc[np.ix_(nrm, nrm)]                     # 正常×正常（拟合）
        Kc_all = Kc[:, nrm]                               # 全量×正常（打分）
        for nu in nu_list:
            clf    = OneClassSVM(kernel='precomputed', nu=nu).fit(Kc_fit)
            scores = -clf.decision_function(Kc_all)
            results[(w, nu)] = roc_auc_score(y, scores)

    # 便捷汇总
    best_wn  = max(results, key=results.get)
    w0_best  = max(results[(0.0, nu)] for nu in nu_list) if 0.0 in w_grid else None
    w1_best  = max(results[(1.0, nu)] for nu in nu_list) if 1.0 in w_grid else None
    results['best']    = (best_wn[0], best_wn[1], results[best_wn])
    results['w0_best'] = w0_best     # 纯 OCSVM（原始 RBF）
    results['w1_best'] = w1_best     # 现 scatter 方法（线性 on H_all）
    return results


# ─── 单次运行（一个 latent_dim） ──────────────────────────────────────────────

def run_one_dim(X, y, kernels, latent_dim, device, cfg):
    t0    = time.time()
    model = train_cmk_scatter(X, y, kernels, latent_dim, device, cfg)
    auc, nu = eval_ocsvm(model, X, y, device)
    return auc, nu, time.time() - t0


# ─── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)
    stem       = os.path.splitext(os.path.basename(DATA_PATH))[0]
    result_csv = os.path.join(RESULT_DIR, f'{stem}_cmk_scatter.csv')

    print(f'加载: {os.path.basename(DATA_PATH)}')
    X, y, meta = load_data(DATA_PATH)
    N, D       = meta['N'], X.shape[1]
    print(f'N={N}  D={D}  异常率={meta["anomaly_rate"]*100:.1f}%  '
          f'正常样本={(y==0).sum()}')

    kernels = gauss_med_kernels(X[y == 0])
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lam     = TRAIN_CFG['lambda_scatter']
    print(f'设备={device}  K={len(kernels)}  λ_scatter={lam}\n{"─"*62}')
    print(f'{"dim":>5}  {"AUC(scatter)":>13}  {"best_nu":>8}  {"time":>7}')
    print(f'{"─"*62}')

    rows = []
    for latent_dim in LATENT_DIMS:
        auc, nu, elapsed = run_one_dim(X, y, kernels, latent_dim, device, TRAIN_CFG)
        print(f'  {latent_dim:>3d}       {auc:.4f}        {nu:.2f}    {elapsed:.1f}s')
        rows.append(dict(dataset=stem, latent_dim=latent_dim,
                         lambda_scatter=lam,
                         auc_scatter=round(auc, 6),
                         best_nu=nu, elapsed_s=round(elapsed, 2)))

    df = pd.DataFrame(rows)
    df.to_csv(result_csv, index=False)

    best = max(rows, key=lambda r: r['auc_scatter'])
    print(f'{"─"*62}')
    print(f'最优: dim={best["latent_dim"]}  AUC={best["auc_scatter"]:.4f}'
          f'  nu={best["best_nu"]:.2f}')
    print(f'保存: {result_csv}')

    # ── 与原始 CMK 对比（若结果文件已存在） ────────────────────────────────────
    baseline_csv = os.path.join(RESULT_DIR, f'{stem}_cmk_ocsvm.csv')
    if os.path.exists(baseline_csv):
        df_base = pd.read_csv(baseline_csv)
        base_best = df_base.loc[df_base['auc_ocsvm'].idxmax()] if 'auc_ocsvm' in df_base.columns else None
        if base_best is not None:
            delta = best['auc_scatter'] - base_best['auc_ocsvm']
            sign  = '+' if delta >= 0 else ''
            print(f'\n对比（vs. 原始 CMK）:')
            print(f'  CMK_OCSVM  AUC={base_best["auc_ocsvm"]:.4f}'
                  f'  dim={int(base_best["latent_dim"])}  nu={base_best["best_nu"]:.2f}')
            print(f'  CMK_scatter AUC={best["auc_scatter"]:.4f}'
                  f'  dim={best["latent_dim"]}  nu={best["best_nu"]:.2f}'
                  f'  Δ={sign}{delta:.4f}')
