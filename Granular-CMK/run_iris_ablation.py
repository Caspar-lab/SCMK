"""
run_iris_ablation.py — 针对 iris_Irisvirginica_11_variant1 数据集的核组合消融实验
===================================================================================

iris 数据集特点：N=111，D=4（极低维），异常率≈10%，n_normal≈100
当前问题：Gauss-5-med + dim=256 只达到 AUC=0.60

改进方向：
  1. latent_dim ≤ D（避免把 4 维数据投影到高维噪声空间）
  2. 异质核组合（不同核类型作为视图，多样性远大于 5 个同类高斯核）
  3. 同时测试线性和 RBF OC-SVM（iris 正常类未必线性可分）

消融因子（核组合 × latent_dim × OC-SVM 核类型）：
  核组合：
    Gauss-5-med   : 当前基线，5 个自适应带宽高斯核
    Hetero-5      : Gaussian + Linear + Polynomial(d=2) + Sigmoid + Cauchy（异质 5 核）
    Hetero-3      : Gaussian + Polynomial(d=3) + Cauchy（3 种差异最大的核，K=3）
    Poly-5        : 5 个不同次数的多项式核（d=1..5）
    GaussCauchy-5 : Gaussian(×3 尺度) + Cauchy(×2 尺度)（混合重尾核）
  latent_dim : [2, 4, 8, 16, 32]（重点测试小维度，因 D=4）
  OC-SVM 核  : linear、rbf（同时测试，取各自最优 AUC）
"""

import os, sys, time, random
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.svm import OneClassSVM
from CMK_OCSVM import (load_data, gauss_med_kernels,
                        CMKNet, cross_kernel_loss,
                        get_embeddings, TRAIN_CFG)
import torch.optim as optim

# ─── 路径 ─────────────────────────────────────────────────────────────────────
DATA_PATH  = r'D:\Microsoft\documents\博士课题\异常检测\论文\5\dataset\numerical\iris_Irisvirginica_11_variant1.mat'
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_CSV = os.path.join(RESULT_DIR, 'iris_kernel_ablation.csv')

# ─── 实验因子 ─────────────────────────────────────────────────────────────────
LATENT_DIMS   = [2, 4, 8, 16, 32]     # 重点测试 ≤D 的维度，D=4
NU_CANDIDATES = [0.01, 0.05, 0.1, 0.2]
OCSVM_KERNELS = ['linear', 'rbf']     # 同时测试两种 OC-SVM 核
TRAIN_CFG_IRIS = {**TRAIN_CFG, 'epochs': 200}  # 样本少，多训练几轮


# ─── 核组合定义 ────────────────────────────────────────────────────────────────

def build_kernel_configs(X_normal):
    """
    生成 5 种核组合，每种返回 list of (name, ktype, kopts)。

    【设计逻辑】
      Gauss-5-med   : 当前基线，仅作对比参考
      Hetero-5      : 5 种完全不同的核类型，跨核对比信号最丰富，
                      适合低维数据（每种核关注不同的几何特性）
      Hetero-3      : 保留差异最大的 3 种核（Gaussian/Polynomial/Cauchy），
                      减少参数量，适合 N=111 的极小样本
      Poly-5        : 5 个不同次数多项式核（d=1..5），
                      多项式核对低维数据的非线性结构敏感
      GaussCauchy-5 : 3 个高斯核（不同尺度）+ 2 个柯西核（不同尺度），
                      柯西核尾部更重，对远离正常中心的异常点更敏感
    """
    # 估计数据自适应带宽
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_normal), min(500, len(X_normal)), replace=False)
    med = np.median(pairwise_distances(X_normal[idx], metric='euclidean'))
    t = max(med, 1e-3)

    configs = {
        'Gauss-5-med': [
            (f'G-{t*r:.3g}', 'Gaussian', {'t': max(t * r, 1e-3)})
            for r in (0.1, 0.5, 1.0, 2.0, 5.0)
        ],
        'Hetero-5': [
            ('Gaussian',   'Gaussian',   {'t': t}),
            ('Linear',     'Linear',     {}),
            ('Poly-d2',    'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 2.0}),
            ('Sigmoid',    'Sigmoid',    {'d': 1.0, 'c': 0.0}),
            ('Cauchy',     'Cauchy',     {'sigma': t}),
        ],
        'Hetero-3': [
            ('Gaussian',   'Gaussian',   {'t': t}),
            ('Poly-d3',    'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 3.0}),
            ('Cauchy',     'Cauchy',     {'sigma': t}),
        ],
        'Poly-5': [
            (f'Poly-d{d}', 'Polynomial', {'a': 1.0, 'b': 1.0, 'd': float(d)})
            for d in (1, 2, 3, 4, 5)
        ],
        'GaussCauchy-5': [
            ('G-sm',    'Gaussian', {'t': max(t * 0.5, 1e-3)}),
            ('G-md',    'Gaussian', {'t': t}),
            ('G-lg',    'Gaussian', {'t': t * 2.0}),
            ('Cau-sm',  'Cauchy',   {'sigma': max(t * 0.5, 1e-3)}),
            ('Cau-lg',  'Cauchy',   {'sigma': t * 2.0}),
        ],
    }
    return configs


# ─── 训练函数（与 CMK_OCSVM.train_cmk 相同，内联以便直接调参）────────────────

def train_cmk(X, y, kernels, latent_dim, device, cfg):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    X_train = X[y == 0]
    N, D    = X_train.shape
    model   = CMKNet(D, latent_dim, len(kernels), cfg['normalize']).to(device)
    opt     = optim.Adam(model.parameters(), lr=cfg['lr'])
    X_t     = torch.tensor(X_train, dtype=torch.float32)
    for _ in range(cfg['epochs']):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, cfg['batch_size']):
            idx = perm[i: i + cfg['batch_size']]
            if len(idx) < 4:
                continue
            hs   = model(X_t[idx].to(device))
            loss = cross_kernel_loss(hs, kernels)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# ─── OC-SVM 评分（同时支持 linear 和 rbf 核）─────────────────────────────────

def ocsvm_best_auc(H_all, H_normal, y, svm_kernel, nu_list):
    """对指定 SVM 核类型搜索最优 nu，返回 (best_nu, best_auc)。"""
    best_nu, best_auc = nu_list[0], -1.0
    for nu in nu_list:
        try:
            clf = OneClassSVM(kernel=svm_kernel, nu=nu)
            clf.fit(H_normal)
            auc = roc_auc_score(y, -clf.decision_function(H_all))
            if auc > best_auc:
                best_auc, best_nu = auc, nu
        except Exception:
            pass
    return best_nu, best_auc


# ─── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'加载: {os.path.basename(DATA_PATH)}')
    X, y, meta = load_data(DATA_PATH)
    n_normal   = int((y == 0).sum())
    print(f'N={meta["N"]}  D={X.shape[1]}  异常率={meta["anomaly_rate"]*100:.1f}%'
          f'  正常样本={n_normal}\n')

    device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kernel_cfgs = build_kernel_configs(X[y == 0])

    rows = []
    for cfg_name, kernels in kernel_cfgs.items():
        print(f'─── 核组合: {cfg_name} (K={len(kernels)}) ───')
        for latent_dim in LATENT_DIMS:
            t0    = time.time()
            model = train_cmk(X, y, kernels, latent_dim, device, TRAIN_CFG_IRIS)
            H_all    = get_embeddings(model, X, device)
            H_normal = H_all[y == 0]

            results = {}
            for svm_k in OCSVM_KERNELS:
                best_nu, best_auc = ocsvm_best_auc(H_all, H_normal, y, svm_k, NU_CANDIDATES)
                results[svm_k] = (best_nu, best_auc)

            elapsed = time.time() - t0
            lin_nu, lin_auc = results['linear']
            rbf_nu, rbf_auc = results['rbf']
            print(f'  dim={latent_dim:>2d}  linear={lin_auc:.4f}(nu={lin_nu:.2f})'
                  f'  rbf={rbf_auc:.4f}(nu={rbf_nu:.2f})  ({elapsed:.1f}s)')

            rows.append(dict(
                kernel_cfg  = cfg_name,
                n_kernels   = len(kernels),
                latent_dim  = latent_dim,
                auc_linear  = round(lin_auc, 6),
                nu_linear   = lin_nu,
                auc_rbf     = round(rbf_auc, 6),
                nu_rbf      = rbf_nu,
                time_s      = round(elapsed, 2),
            ))
        print()

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False)
    print(f'结果已保存: {RESULT_CSV}')

    # ── 汇总：按核组合显示最优 (dim, linear_AUC, rbf_AUC) ───────────────────
    print(f'\n{"="*68}')
    print(f'iris 核消融汇总（每种核组合取各自最优 latent_dim）')
    print(f'{"核组合":<16} {"K":>2}  {"最优dim":>6}  {"Linear AUC":>11}  {"RBF AUC":>10}')
    print(f'{"-"*55}')
    for cfg_name in kernel_cfgs:
        sub = df[df['kernel_cfg'] == cfg_name]
        best_lin_row = sub.loc[sub['auc_linear'].idxmax()]
        best_rbf_row = sub.loc[sub['auc_rbf'].idxmax()]
        # 分别显示 linear 和 rbf 各自最优维度
        print(f'{cfg_name:<16} {len(kernel_cfgs[cfg_name]):>2}'
              f'  lin@{int(best_lin_row["latent_dim"]):<3d} = {best_lin_row["auc_linear"]:.4f}'
              f'    rbf@{int(best_rbf_row["latent_dim"]):<3d} = {best_rbf_row["auc_rbf"]:.4f}')
    print(f'{"="*68}')
    print(f'epochs={TRAIN_CFG_IRIS["epochs"]}  nu候选={NU_CANDIDATES}  latent_dims={LATENT_DIMS}')
