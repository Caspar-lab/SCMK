"""
核组合消融实验 — 验证"同核异参"是否优于"异核单参"

实验设计：
  控制变量：K=5（5 个投影头，模型参数量相同）、其余超参数全部固定
  自变量：核的组合方式（下列 7 种配置）
  因变量：CK-Incon AUC（跨核不一致性得分，不需要调 K 参数）
          CK-KNN  AUC（拼接嵌入 KNN，K∈[2,60] 取最优）

7 种核配置（均含 5 个核）：
  Hetero-5    : 5 种不同核族（当前默认，对照组）
  Gauss-5-lin : 5 个 Gaussian，t 线性间隔  [0.5, 1.0, 1.5, 2.0, 2.5]
  Gauss-5-log : 5 个 Gaussian，t 对数间隔  [0.1, 0.5, 1.0, 5.0, 10.0]
  Gauss-5-med : 5 个 Gaussian，t 基于数据中位距离自适应设置（×[0.1,0.5,1,2,5]）
  Gauss-2+het : 2 个 Gaussian（best t 线性）+ 3 种其他核（减少 Gaussian 数量）
  Cauchy-5    : 5 个 Cauchy，sigma 对数间隔 [0.1, 0.5, 1.0, 2.0, 5.0]
  Poly-5      : 5 个 Polynomial，degree∈[2,3,4,5,6]，保持 a=b=1

为什么用 CK-Incon 作主指标：
  该指标直接衡量各核嵌入的跨核一致性，不需要搜索额外的 KNN 超参数 K，
  结果更干净、计算更快，且理论上与算法设计目标直接对应。
"""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CKAD import load_data, train_ckad, get_embeddings, \
                 cross_kernel_inconsistency, best_knn_auc, roc_auc_score

# ─── 数据目录 ──────────────────────────────────────────────────────────────────
DATA_DIR = r'D:\Microsoft\documents\博士课题\异常检测\论文\5\dataset\mixed'
K_MIN, K_MAX = 2, 60

# ─── 训练超参数（固定，所有配置相同）─────────────────────────────────────────
TRAIN_CFG = dict(latent_dim=64, epochs=100, batch_size=512,
                 lr=0.01, normalize=True, seed=42)

# ─── 7 种核配置（K=5）─────────────────────────────────────────────────────────

def gauss_med_kernels(X):
    """基于数据中位距离自适应设置 Gaussian 带宽。"""
    from sklearn.metrics import pairwise_distances
    # 随机采样最多 500 个样本估计中位距离
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(500, len(X)), replace=False)
    med = np.median(pairwise_distances(X[idx], metric='euclidean'))
    med = max(med, 1e-3)
    ts = [med * r for r in [0.1, 0.5, 1.0, 2.0, 5.0]]
    return [(f'Gauss-{t:.3g}', 'Gaussian', {'t': t}) for t in ts]


STATIC_CONFIGS = {
    'Hetero-5': [
        ('Gaussian',    'Gaussian',   {'t': 1.0}),
        ('Linear',      'Linear',     {}),
        ('Polynomial',  'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 2.0}),
        ('Sigmoid',     'Sigmoid',    {'d': 2.0, 'c': 0.0}),
        ('Cauchy',      'Cauchy',     {'sigma': 1.0}),
    ],
    'Gauss-5-lin': [          # t 线性间隔
        ('G-0.5',  'Gaussian', {'t': 0.5}),
        ('G-1.0',  'Gaussian', {'t': 1.0}),
        ('G-1.5',  'Gaussian', {'t': 1.5}),
        ('G-2.0',  'Gaussian', {'t': 2.0}),
        ('G-2.5',  'Gaussian', {'t': 2.5}),
    ],
    'Gauss-5-log': [          # t 对数间隔
        ('G-0.1',  'Gaussian', {'t': 0.1}),
        ('G-0.5',  'Gaussian', {'t': 0.5}),
        ('G-1.0',  'Gaussian', {'t': 1.0}),
        ('G-5.0',  'Gaussian', {'t': 5.0}),
        ('G-10.0', 'Gaussian', {'t': 10.0}),
    ],
    'Gauss-2+het': [          # 2 个 Gaussian + 3 种异构核
        ('G-0.5',      'Gaussian',   {'t': 0.5}),
        ('G-2.0',      'Gaussian',   {'t': 2.0}),
        ('Linear',     'Linear',     {}),
        ('Polynomial', 'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 2.0}),
        ('Cauchy',     'Cauchy',     {'sigma': 1.0}),
    ],
    'Cauchy-5': [             # 同核异参：Cauchy
        ('C-0.1', 'Cauchy', {'sigma': 0.1}),
        ('C-0.5', 'Cauchy', {'sigma': 0.5}),
        ('C-1.0', 'Cauchy', {'sigma': 1.0}),
        ('C-2.0', 'Cauchy', {'sigma': 2.0}),
        ('C-5.0', 'Cauchy', {'sigma': 5.0}),
    ],
    'Poly-5': [               # 同核异参：Polynomial，不同 degree
        ('P-d2', 'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 2.0}),
        ('P-d3', 'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 3.0}),
        ('P-d4', 'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 4.0}),
        ('P-d5', 'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 5.0}),
        ('P-d6', 'Polynomial', {'a': 1.0, 'b': 1.0, 'd': 6.0}),
    ],
    # 'Gauss-5-med' 依赖数据，在运行时动态生成（见下方）
}

CONFIG_NAMES = ['Hetero-5', 'Gauss-5-lin', 'Gauss-5-log', 'Gauss-5-med',
                'Gauss-2+het', 'Cauchy-5', 'Poly-5']


# ─── 单数据集消融 ──────────────────────────────────────────────────────────────

def run_one_dataset(path):
    import torch
    fname = os.path.basename(path)
    print(f'\n{"="*70}\n>>> {fname}')

    X, y, meta = load_data(path)
    N, ar = meta['N'], meta['anomaly_rate']
    print(f'  N={N}  D={X.shape[1]}  异常率={ar*100:.1f}%  '
          f'训练正常样本={int((y==0).sum())}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 构造自适应 Gaussian 配置
    configs = dict(STATIC_CONFIGS)
    configs['Gauss-5-med'] = gauss_med_kernels(X)

    row = {'dataset': os.path.splitext(fname)[0], 'N': N, 'anomaly_rate': ar}

    for cfg_name in CONFIG_NAMES:
        kernels = configs[cfg_name]
        model = train_ckad(X, y=y, kernels=kernels, **TRAIN_CFG,
                           device=device, print_freq=TRAIN_CFG['epochs'] + 1)
        H_all, H_per = get_embeddings(model, X, device=device)

        # CK-Incon（直接用，无需调 K）
        incon = cross_kernel_inconsistency(H_per)
        auc_incon = roc_auc_score(y, incon)

        # CK-KNN（搜索最优 K）
        bk, auc_knn = best_knn_auc(H_all, y, K_MIN, K_MAX)

        row[f'{cfg_name}_incon'] = auc_incon
        row[f'{cfg_name}_knn']   = auc_knn
        row[f'{cfg_name}_bestk'] = bk
        print(f'  [{cfg_name:<12}]  Incon={auc_incon:.4f}  KNN={auc_knn:.4f}(k={bk})')

    return row


# ─── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    datasets = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.mat'))
    all_rows = []
    for fname in datasets:
        row = run_one_dataset(os.path.join(DATA_DIR, fname))
        all_rows.append(row)

    # ── 汇总表：CK-Incon ──────────────────────────────────────────────────────
    col_w = 11
    sep = '=' * (35 + col_w * len(CONFIG_NAMES) + 4)
    print(f'\n\n{sep}')
    print('CK-Incon AUC 汇总（越高越好；训练仅使用正常样本）')
    header = f'{"Dataset":<35}' + ''.join(f'{c:>{col_w}}' for c in CONFIG_NAMES)
    print(header)
    print('-' * len(sep))
    for r in all_rows:
        vals = [r[f'{c}_incon'] for c in CONFIG_NAMES]
        best = max(vals)
        name = r['dataset'][:34]
        line = f'{name:<35}'
        for v in vals:
            marker = '*' if abs(v - best) < 1e-6 else ' '
            line += f'{v:>{col_w-1}.4f}{marker}'
        print(line)

    avgs = [sum(r[f'{c}_incon'] for r in all_rows) / len(all_rows) for c in CONFIG_NAMES]
    best_avg = max(avgs)
    print('-' * len(sep))
    avg_line = f'{"Average":<35}'
    for v in avgs:
        marker = '*' if abs(v - best_avg) < 1e-6 else ' '
        avg_line += f'{v:>{col_w-1}.4f}{marker}'
    print(avg_line)
    print(sep)

    # ── 汇总表：CK-KNN ────────────────────────────────────────────────────────
    print(f'\n{sep}')
    print('CK-KNN AUC 汇总（K∈[2,60] 取最优；训练仅使用正常样本）')
    print(header)
    print('-' * len(sep))
    for r in all_rows:
        vals = [r[f'{c}_knn'] for c in CONFIG_NAMES]
        best = max(vals)
        name = r['dataset'][:34]
        line = f'{name:<35}'
        for v in vals:
            marker = '*' if abs(v - best) < 1e-6 else ' '
            line += f'{v:>{col_w-1}.4f}{marker}'
        print(line)

    avgs_knn = [sum(r[f'{c}_knn'] for r in all_rows) / len(all_rows) for c in CONFIG_NAMES]
    best_avg_knn = max(avgs_knn)
    print('-' * len(sep))
    avg_line2 = f'{"Average":<35}'
    for v in avgs_knn:
        marker = '*' if abs(v - best_avg_knn) < 1e-6 else ' '
        avg_line2 += f'{v:>{col_w-1}.4f}{marker}'
    print(avg_line2)
    print(sep)

    print(f'\n说明：* 表示该数据集上最优配置')
    print(f'latent_dim={TRAIN_CFG["latent_dim"]}  epochs={TRAIN_CFG["epochs"]}'
          f'  batch={TRAIN_CFG["batch_size"]}  lr={TRAIN_CFG["lr"]}'
          f'  K_search=[{K_MIN},{K_MAX}]')
