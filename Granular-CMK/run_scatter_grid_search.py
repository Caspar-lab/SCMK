"""
run_scatter_grid_search.py — CMK_OCSVM_scatter 三维参数网格搜索
================================================================

在 run_lambda_search.py 基础上扩展：
  原脚本：固定 dim=32，仅搜索 lambda
  本脚本：联合搜索 latent_dim × lambda_scatter × nu 三维组合

搜索空间（每数据集独立，不跨数据集平均选参）：
  latent_dim     ∈ LATENT_DIMS   （5 个维度）
  lambda_scatter ∈ LAMBDAS       （6 个值，0.0 退化为纯 CMK 基线）
  nu             ∈ NU_CANDIDATES （4 个值）

训练次数 = len(LATENT_DIMS) × len(LAMBDAS)（per dataset）
nu 在训练后轻量搜索（不重新训练），总开销约 30 次训练 × N 数据集。
"""

import os, sys, time
import pandas as pd
import torch
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CMK_OCSVM import (load_data, gauss_med_kernels,
                        get_embeddings, NU_CANDIDATES, TRAIN_CFG as _BASE_CFG)
from CMK_OCSVM_scatter import train_cmk_scatter

# ─── 路径配置 ──────────────────────────────────────────────────────────────────

DATA_DIR   = r'C:\OD\Shihao\5\dataset\numerical'
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'result', 'scatter_grid_search')
os.makedirs(RESULT_DIR, exist_ok=True)

ALL_CSV  = os.path.join(RESULT_DIR, 'scatter_grid_all.csv')
BEST_CSV = os.path.join(RESULT_DIR, 'scatter_grid_best.csv')

# ─── 搜索空间 ──────────────────────────────────────────────────────────────────

LATENT_DIMS = [16, 32, 64, 128, 256]
LAMBDAS     = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
# NU_CANDIDATES 直接从 CMK_OCSVM 导入：[0.01, 0.05, 0.1, 0.2]


# ─── 单数据集三维搜索 ──────────────────────────────────────────────────────────

def run_one(path, device):
    """
    对单个数据集运行完整的 dim × lambda × nu 网格搜索。

    返回 rows：每行对应一组 (dim, lambda, nu)，记录 AUC 等信息。
    训练只在 (dim, lambda) 维度循环，nu 在每次训练后轻量搜索（无需重新训练）。
    """
    stem       = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    kernels    = gauss_med_kernels(X[y == 0])
    n_normal   = int((y == 0).sum())

    rows = []
    n_dim = len(LATENT_DIMS)
    n_lam = len(LAMBDAS)

    for i_dim, dim in enumerate(LATENT_DIMS):
        for i_lam, lam in enumerate(LAMBDAS):
            cfg = {**_BASE_CFG, 'lambda_scatter': lam}
            t0  = time.time()

            model = train_cmk_scatter(X, y, kernels, dim, device, cfg)

            # 提取全量嵌入（训练一次，nu 在此基础上轻量搜索）
            H_all    = get_embeddings(model, X, device)
            H_normal = H_all[y == 0]
            elapsed  = time.time() - t0

            # 搜索 nu：对每个候选值拟合 OC-SVM，记录各自 AUC
            best_auc_dim_lam = -1.0
            for nu in NU_CANDIDATES:
                try:
                    clf    = OneClassSVM(kernel='linear', nu=nu).fit(H_normal)
                    scores = -clf.decision_function(H_all)
                    auc    = roc_auc_score(y, scores)
                except Exception:
                    auc = float('nan')

                rows.append(dict(
                    dataset        = stem,
                    latent_dim     = dim,
                    lambda_scatter = lam,
                    nu             = nu,
                    auc            = round(auc, 6) if not np.isnan(auc) else float('nan'),
                    elapsed_s      = round(elapsed, 2),
                ))
                if not np.isnan(auc) and auc > best_auc_dim_lam:
                    best_auc_dim_lam = auc

            tag = '(base)' if lam == 0 else ''
            prog = f'[{i_dim+1}/{n_dim}][{i_lam+1}/{n_lam}]'
            print(f'  {prog} dim={dim:>3d}  λ={lam:>7.1f} {tag:<7}'
                  f'  best_AUC={best_auc_dim_lam:.4f}  ({elapsed:.1f}s)')

    return rows


# ─── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    datasets = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.mat'))
    device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n_train_per_ds = len(LATENT_DIMS) * len(LAMBDAS)
    print(f'共 {len(datasets)} 个数据集  设备: {device}')
    print(f'搜索空间: dim={LATENT_DIMS}')
    print(f'          lambda={LAMBDAS}')
    print(f'          nu={NU_CANDIDATES}')
    print(f'每数据集训练次数: {n_train_per_ds}  '
          f'(dim×λ={len(LATENT_DIMS)}×{len(LAMBDAS)}，nu 轻量搜索)')
    print(f'{"="*72}')

    all_rows, best_rows = [], []

    for fname in datasets:
        path = os.path.join(DATA_DIR, fname)
        stem = os.path.splitext(fname)[0]
        X, y, meta = load_data(path)
        print(f'\n[{stem}]  N={meta["N"]}  D={X.shape[1]}  '
              f'异常率={meta["anomaly_rate"]*100:.1f}%')

        t_ds  = time.time()
        rows  = run_one(path, device)
        all_rows.extend(rows)

        # ── 每数据集汇总 ──────────────────────────────────────────────────────
        df_ds    = pd.DataFrame([r for r in rows if not pd.isna(r['auc'])])
        best_row = df_ds.loc[df_ds['auc'].idxmax()].to_dict()

        # lambda=0（纯 CMK）中各 nu 的最高 AUC 作为基线
        base_df  = df_ds[df_ds['lambda_scatter'] == 0.0]
        base_auc = base_df['auc'].max() if len(base_df) > 0 else float('nan')
        base_nu  = base_df.loc[base_df['auc'].idxmax(), 'nu'] if len(base_df) > 0 else None

        gain = (best_row['auc'] - base_auc) if not np.isnan(base_auc) else float('nan')

        best_rows.append(dict(
            dataset        = stem,
            best_dim       = int(best_row['latent_dim']),
            best_lambda    = best_row['lambda_scatter'],
            best_nu        = best_row['nu'],
            best_auc       = round(best_row['auc'], 6),
            baseline_auc   = round(base_auc, 6) if not np.isnan(base_auc) else float('nan'),
            baseline_nu    = base_nu,
            gain_vs_base   = round(gain, 6) if not np.isnan(gain) else float('nan'),
        ))

        print(f'  → 最佳: dim={int(best_row["latent_dim"])}  '
              f'λ={best_row["lambda_scatter"]}  nu={best_row["nu"]}  '
              f'AUC={best_row["auc"]:.4f}  '
              f'(λ=0基线: {base_auc:.4f}, 增益{gain:+.4f})'
              f'  ({time.time()-t_ds:.1f}s)')

        # 增量保存（防止中途中断丢失）
        pd.DataFrame(all_rows).to_csv(ALL_CSV, index=False)
        pd.DataFrame(best_rows).to_csv(BEST_CSV, index=False)

    # ── 全局汇总表 ────────────────────────────────────────────────────────────
    print(f'\n{"="*80}')
    print(f'CMK_OCSVM_scatter 三维网格搜索汇总（每数据集独立最优参数）')
    print(f'{"Dataset":<38} {"dim":>4}  {"lambda":>8}  {"nu":>5}  '
          f'{"bestAUC":>8}  {"baseAUC":>8}  {"gain":>7}')
    print(f'{"-"*80}')

    aucs_best, aucs_base = [], []
    for r in best_rows:
        name = r['dataset'][:37]
        base_str = f'{r["baseline_auc"]:>8.4f}' if not pd.isna(r['baseline_auc']) else f'{"N/A":>8}'
        gain_str = f'{r["gain_vs_base"]:>+7.4f}' if not pd.isna(r['gain_vs_base']) else f'{"N/A":>7}'
        print(f'{name:<38} {r["best_dim"]:>4d}  {r["best_lambda"]:>8.1f}  '
              f'{r["best_nu"]:>5.2f}  {r["best_auc"]:>8.4f}  '
              f'{base_str}  {gain_str}')
        aucs_best.append(r['best_auc'])
        if not pd.isna(r['baseline_auc']):
            aucs_base.append(r['baseline_auc'])

    print(f'{"-"*80}')
    avg_best = sum(aucs_best) / len(aucs_best) if aucs_best else float('nan')
    avg_base = sum(aucs_base) / len(aucs_base) if aucs_base else float('nan')
    avg_gain = avg_best - avg_base if not np.isnan(avg_base) else float('nan')
    print(f'{"Average":<38} {"":>4}  {"":>8}  {"":>5}  '
          f'{avg_best:>8.4f}  {avg_base:>8.4f}  {avg_gain:>+7.4f}')
    print(f'{"="*80}')

    # ── 参数分布统计 ──────────────────────────────────────────────────────────
    df_best = pd.DataFrame(best_rows)
    print(f'\n最优参数分布：')
    print(f'  dim    : {dict(df_best["best_dim"].value_counts().sort_index())}')
    print(f'  lambda : {dict(df_best["best_lambda"].value_counts().sort_index())}')
    print(f'  nu     : {dict(df_best["best_nu"].value_counts().sort_index())}')

    print(f'\n明细: {ALL_CSV}')
    print(f'汇总: {BEST_CSV}')
