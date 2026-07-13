"""
run_blend_search.py — 检测阶段核混合权重 w 扫描
================================================

对每个数值数据集训练一次 CMK（dim 固定），然后在检测阶段扫描
核混合权重 w：K(w) = (1-w)·K_raw(RBF on X) + w·K_emb(linear on H_all)。

  w=0 → 纯 RBF OCSVM（原始特征）
  w=1 → 现 scatter 方法（线性 OCSVM on H_all）

同时报告 oracle-best-w（上界）与固定 w=0.5（无偷看的真实可部署值）。
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CMK_OCSVM import load_data, gauss_med_kernels, NU_CANDIDATES, TRAIN_CFG as _BASE_CFG
from CMK_OCSVM_scatter import train_cmk_scatter, eval_ocsvm_blend

DATA_DIR   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dataset', 'numerical')
RESULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'result', 'cmk_scatter_blend')
os.makedirs(RESULT_DIR, exist_ok=True)

LATENT_DIM     = 32
LAMBDA_SCATTER = 10.0                       # λ 搜索得到的稳健默认值
W_GRID         = [0.0, 0.25, 0.5, 0.75, 1.0]
FIXED_W        = 0.5

TRAIN_CFG  = {**_BASE_CFG, 'lambda_scatter': LAMBDA_SCATTER}
ALL_CSV    = os.path.join(RESULT_DIR, 'blend_all.csv')
SUMMARY    = os.path.join(RESULT_DIR, 'blend_summary.csv')


def run_one(path, device):
    stem       = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    kernels    = gauss_med_kernels(X[y == 0])

    t0    = time.time()
    model = train_cmk_scatter(X, y, kernels, LATENT_DIM, device, TRAIN_CFG)
    res   = eval_ocsvm_blend(model, X, y, device, w_grid=W_GRID, nu_list=NU_CANDIDATES)
    elapsed = time.time() - t0

    # 明细行
    rows = [dict(dataset=stem, w=w, nu=nu, auc=round(res[(w, nu)], 6))
            for w in W_GRID for nu in NU_CANDIDATES]

    # 各 w 的最优 nu AUC
    auc_by_w  = {w: max(res[(w, nu)] for nu in NU_CANDIDATES) for w in W_GRID}
    bw, bnu, bauc = res['best']
    auc_ocsvm = res['w0_best']
    auc_scat  = res['w1_best']
    auc_fixed = auc_by_w[FIXED_W]
    max_end   = max(auc_ocsvm, auc_scat)

    srow = dict(
        dataset        = stem,
        auc_ocsvm_w0   = round(auc_ocsvm, 6),
        auc_scatter_w1 = round(auc_scat, 6),
        auc_fixed_w05  = round(auc_fixed, 6),
        best_w         = bw,
        best_w_auc     = round(bauc, 6),
        best_nu        = bnu,
        gain_vs_maxend = round(bauc - max_end, 6),
        inter_beats_both = bool(0.0 < bw < 1.0 and bauc > max_end + 1e-9),
        elapsed_s      = round(elapsed, 2),
    )
    bar = '  '.join(f'w={w}:{auc_by_w[w]:.3f}' for w in W_GRID)
    print(f'[{stem}]  {bar}')
    print(f'    OCSVM(w0)={auc_ocsvm:.4f}  scatter(w1)={auc_scat:.4f}  '
          f'fixed(w.5)={auc_fixed:.4f}  bestw={bw}→{bauc:.4f}  '
          f'(Δvs.max端={srow["gain_vs_maxend"]:+.4f})  ({elapsed:.1f}s)')
    return rows, srow


if __name__ == '__main__':
    datasets = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.mat'))
    device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'共 {len(datasets)} 个数据集  dim={LATENT_DIM}  λ={LAMBDA_SCATTER}  '
          f'w={W_GRID}\n{"="*70}')

    all_rows, summary = [], []
    for fname in datasets:
        rows, srow = run_one(os.path.join(DATA_DIR, fname), device)
        all_rows.extend(rows)
        summary.append(srow)
        pd.DataFrame(all_rows).to_csv(ALL_CSV, index=False)
        pd.DataFrame(summary).to_csv(SUMMARY, index=False)

    df = pd.DataFrame(summary)
    print(f'\n{"="*72}')
    print(f'核混合权重 w 汇总（dim={LATENT_DIM}, λ={LAMBDA_SCATTER}）')
    print(f'{"-"*72}')
    # 各固定 w 的平均 AUC
    all_df = pd.DataFrame(all_rows)
    print('各固定 w 的平均 AUC（每数据集取最优 nu）：')
    for w in W_GRID:
        m = all_df[all_df['w'] == w].groupby('dataset')['auc'].max().mean()
        tag = ' (=纯OCSVM)' if w == 0 else (' (=scatter)' if w == 1 else '')
        print(f'  w={w:<4} mean AUC={m:.4f}{tag}')
    print(f'{"-"*72}')
    print(f'平均 OCSVM(w=0)     : {df["auc_ocsvm_w0"].mean():.4f}')
    print(f'平均 scatter(w=1)   : {df["auc_scatter_w1"].mean():.4f}')
    print(f'平均 fixed(w=0.5)   : {df["auc_fixed_w05"].mean():.4f}')
    print(f'平均 best-w(oracle) : {df["best_w_auc"].mean():.4f}')
    n_inter = int(df['inter_beats_both'].sum())
    print(f'中间 w∈(0,1) 严格超过两端点的数据集数：{n_inter} / {len(df)}')
    print(f'best-w ≥ max(两端点) 的数据集数：'
          f'{int((df["gain_vs_maxend"] >= -1e-9).sum())} / {len(df)}（应=全部）')
    print(f'{"="*72}')
    print(f'\n明细: {ALL_CSV}\n汇总: {SUMMARY}')
