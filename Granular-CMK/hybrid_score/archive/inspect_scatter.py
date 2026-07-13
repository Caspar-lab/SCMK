"""
inspect_scatter.py — scatter 单数据集效果检验工具
==================================================

用途：在指定数据集上独立复核 scatter(max_ensemble) 的真实效果，回答
      "scatter 在这个数据集上是否真的那么好？"

透明度设计（不只报最优 AUC，还暴露以下信息）：
  1. 完整网格：dim × lambda 每个配置的 max_ensemble AUC（看是普遍好还是仅个别配置好）
  2. oracle 程度：网格 AUC 的 best / mean / median / std（best 远高于 mean ⇒ 依赖挑参）
  3. 信号归因：最优配置下 normalized(方向) / norm_rbf(幅值) / max_ensemble 各自 AUC
  4. 基线对照：原始特征 RBF OCSVM 的 AUC（scatter 要超过它才有意义）

【评估口径说明】
  AUC 均为在测试集上对 nu（及网格 dim/lambda）取最优 —— 这是 oracle 上界估计，
  与 hybrid_best.csv 口径一致。本工具帮助判断该上界是"普遍稳健"还是"挑参得来"。
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hybrid_score import (load_data, gauss_med_kernels, extract_components,
                              _best_ocsvm_scores, _minmax)
from CMK_OCSVM_scatter import train_cmk_scatter

# ════════════════════════════════════════════════════════════════════════════
#                                参 数 配 置
# ════════════════════════════════════════════════════════════════════════════

# 要检验的数据集：写数据集名（自动在下方 SEARCH_DIRS 里找 .mat），或直接写完整 .mat 路径
DATASETS = [
    'vertebral',
    # 'cardio',
    # 'autos_variant1',
    # r'C:\path\to\some.mat',
]

# 数据集 .mat 搜索目录（按顺序查找，命中即用）
SEARCH_DIRS = [
    r'C:\OD\Shihao\datasets',
]

# 搜索网格
LATENT_DIMS   = [16, 32, 64, 128, 256]
LAMBDAS       = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
NU_CANDIDATES = [0.01, 0.05, 0.1, 0.2]

# 训练超参数
EPOCHS     = 100
BATCH_SIZE = 512
LR         = 0.01
NORMALIZE  = True     # 训练时 L2 归一化嵌入（scatter 标准设置）
SEED       = 42

# 输出开关
SHOW_FULL_GRID  = True    # 打印完整 dim×lambda 网格
SHOW_BASELINE   = True    # 计算原始特征 RBF OCSVM 基线对照
SAVE_CSV        = True     # 保存逐配置明细 csv
CSV_DIR         = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════════════════


def locate(name):
    """把数据集名或路径解析为 .mat 绝对路径。"""
    if name.lower().endswith('.mat') and os.path.exists(name):
        return name
    for d in SEARCH_DIRS:
        p = os.path.join(d, name + '.mat')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'找不到数据集 {name}（搜索目录: {SEARCH_DIRS}）')


def eval_three_sources(model, X, y, device):
    """返回 normalized / norm_rbf / max_ensemble 三个分数源的 AUC（各自 nu 最优）。"""
    H_norm_per, H_norms = extract_components(model, X, device)
    H_dir = np.concatenate(H_norm_per, axis=1)
    a_dir, nu_dir, s_dir = _best_ocsvm_scores(H_dir,   y, H_dir[y == 0],   'linear', NU_CANDIDATES)
    a_nrm, nu_nrm, s_nrm = _best_ocsvm_scores(H_norms, y, H_norms[y == 0], 'rbf',    NU_CANDIDATES)
    if s_dir is None or s_nrm is None:
        a_ens = max(a_dir, a_nrm)
    else:
        a_ens = roc_auc_score(y, np.maximum(_minmax(s_dir), _minmax(s_nrm)))
    return dict(normalized=a_dir, norm_rbf=a_nrm, max_ensemble=a_ens,
                nu_dir=nu_dir, nu_nrm=nu_nrm)


def ocsvm_baseline(X, y):
    """原始特征上的 RBF 半监督 OCSVM（nu 网格取最优 AUC），作对照基线。"""
    best = -1.0
    for nu in NU_CANDIDATES:
        clf = OneClassSVM(kernel='rbf', gamma='scale', nu=nu).fit(X[y == 0])
        best = max(best, roc_auc_score(y, -clf.decision_function(X)))
    return best


def run_dataset(name, device):
    path = locate(name)
    stem = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    kernels = gauss_med_kernels(X[y == 0])

    print(f'\n{"="*74}')
    print(f'数据集: {stem}')
    print(f'  N={meta["N"]}  D={X.shape[1]}  异常率={meta["anomaly_rate"]*100:.1f}%  '
          f'正常={int((y==0).sum())}  核数K={len(kernels)}')
    print(f'  网格: dim={LATENT_DIMS}  lambda={LAMBDAS}  nu={NU_CANDIDATES}')
    print(f'  训练: epochs={EPOCHS} batch={BATCH_SIZE} lr={LR} seed={SEED}')

    base_auc = ocsvm_baseline(X, y) if SHOW_BASELINE else None
    if base_auc is not None:
        print(f'  基线 RBF-OCSVM(原始特征) AUC = {base_auc:.4f}')
    print(f'{"-"*74}')

    cfg_base = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                    normalize=NORMALIZE, seed=SEED)

    rows = []
    grid = np.full((len(LATENT_DIMS), len(LAMBDAS)), np.nan)  # max_ensemble AUC
    for i, dim in enumerate(LATENT_DIMS):
        for j, lam in enumerate(LAMBDAS):
            cfg = {**cfg_base, 'lambda_scatter': lam}
            t0 = time.time()
            model = train_cmk_scatter(X, y, kernels, dim, device, cfg)
            res = eval_three_sources(model, X, y, device)
            dt = time.time() - t0
            grid[i, j] = res['max_ensemble']
            rows.append(dict(dataset=stem, latent_dim=dim, lambda_scatter=lam,
                             auc_normalized=round(res['normalized'], 4),
                             auc_norm_rbf=round(res['norm_rbf'], 4),
                             auc_max_ensemble=round(res['max_ensemble'], 4),
                             elapsed_s=round(dt, 1)))

    df = pd.DataFrame(rows)

    # ── 完整网格 ──
    if SHOW_FULL_GRID:
        print('max_ensemble AUC 网格 (行=dim, 列=lambda):')
        hdr = '  dim\\lam ' + ''.join(f'{lam:>9.4g}' for lam in LAMBDAS)
        print(hdr)
        for i, dim in enumerate(LATENT_DIMS):
            print(f'  {dim:>6} ' + ''.join(f'{grid[i,j]:>9.4f}' for j in range(len(LAMBDAS))))
        print()

    # ── oracle 程度 ──
    flat = df['auc_max_ensemble'].values
    best_row = df.loc[df['auc_max_ensemble'].idxmax()]
    print(f'max_ensemble 网格统计:  best={flat.max():.4f}  mean={flat.mean():.4f}  '
          f'median={np.median(flat):.4f}  min={flat.min():.4f}  std={flat.std():.4f}')
    gap_bm = flat.max() - flat.mean()
    verdict = ('普遍稳健 (best≈mean)' if gap_bm < 0.03 else
               '中等依赖挑参' if gap_bm < 0.10 else '强依赖挑参 (best≫mean)')
    print(f'  best - mean = {gap_bm:+.4f}  →  {verdict}')

    # ── 信号归因（在最优配置下重训一次，看三源）──
    bd, bl = int(best_row['latent_dim']), float(best_row['lambda_scatter'])
    model = train_cmk_scatter(X, y, kernels, bd, device, {**cfg_base, 'lambda_scatter': bl})
    src = eval_three_sources(model, X, y, device)
    print(f'\n最优配置: dim={bd}  lambda={bl}  →  max_ensemble AUC={best_row["auc_max_ensemble"]:.4f}')
    print(f'  信号归因:  normalized(方向)={src["normalized"]:.4f}(nu={src["nu_dir"]})   '
          f'norm_rbf(幅值)={src["norm_rbf"]:.4f}(nu={src["nu_nrm"]})')
    dom = 'norm_rbf(幅值)' if src['norm_rbf'] > src['normalized'] else 'normalized(方向)'
    print(f'  主导信号: {dom}')
    if base_auc is not None:
        delta = best_row['auc_max_ensemble'] - base_auc
        print(f'  vs 原始RBF-OCSVM基线: {best_row["auc_max_ensemble"]:.4f} - {base_auc:.4f} '
              f'= {delta:+.4f}  ({"scatter更优" if delta>0 else "基线更优/持平"})')

    if SAVE_CSV:
        out = os.path.join(CSV_DIR, f'inspect_{stem}.csv')
        df.to_csv(out, index=False)
        print(f'  明细已存: {out}')

    return df


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  共检验 {len(DATASETS)} 个数据集')

    summary = []
    for name in DATASETS:
        try:
            df = run_dataset(name, device)
            br = df.loc[df['auc_max_ensemble'].idxmax()]
            summary.append((br['dataset'], br['latent_dim'], br['lambda_scatter'],
                            br['auc_max_ensemble'], df['auc_max_ensemble'].mean()))
        except Exception as e:
            print(f'[ERR] {name}: {e}')

    if len(summary) > 1:
        print(f'\n{"="*74}\n汇总:')
        print(f'  {"dataset":<36}{"dim":>5}{"lambda":>9}{"best":>9}{"mean":>9}')
        for d, dim, lam, best, mean in summary:
            print(f'  {d:<36}{int(dim):>5}{lam:>9.4g}{best:>9.4f}{mean:>9.4f}')
