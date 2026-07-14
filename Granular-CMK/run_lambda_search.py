"""
run_lambda_search.py — 为每个数值数据集单独搜索最合适的 lambda_scatter
========================================================================

为加快搜索，固定 latent_dim（默认 32），仅扫描 lambda 网格。
每个数据集独立选出使 AUC 最高的 lambda（不做跨数据集平均）。
lambda=0 即纯 CMK 基线（无散度惩罚）。
"""

import os, sys, time
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CMK_OCSVM import load_data, gauss_med_kernels, TRAIN_CFG as _BASE_CFG
from CMK_OCSVM_scatter import train_cmk_scatter, eval_ocsvm

DATA_DIR   = r'C:\OD\Shihao\5\dataset\numerical'
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result/lambda_search')
os.makedirs(RESULT_DIR, exist_ok=True)

LATENT_DIM = 32
LAMBDAS    = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]

ALL_CSV  = os.path.join(RESULT_DIR, 'numerical_lambda_search_all.csv')
BEST_CSV = os.path.join(RESULT_DIR, 'numerical_lambda_search_best.csv')


def run_one(path, device):
    stem       = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    kernels    = gauss_med_kernels(X[y == 0])
    rows       = []

    for lam in LAMBDAS:
        cfg     = {**_BASE_CFG, 'lambda_scatter': lam}
        t0      = time.time()
        model   = train_cmk_scatter(X, y, kernels, LATENT_DIM, device, cfg)
        auc, nu = eval_ocsvm(model, X, y, device)
        elapsed = time.time() - t0
        rows.append(dict(dataset=stem, latent_dim=LATENT_DIM,
                         lambda_scatter=lam, auc=round(auc, 6),
                         best_nu=nu, elapsed_s=round(elapsed, 2)))
        tag = '(baseline)' if lam == 0 else ''
        print(f'  λ={lam:>7.1f} {tag:<10} AUC={auc:.4f}  nu={nu:.2f}  ({elapsed:.1f}s)')

    return rows


if __name__ == '__main__':
    datasets = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.mat'))
    device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'共 {len(datasets)} 个数据集  设备: {device}  dim={LATENT_DIM}  '
          f'λ={LAMBDAS}\n{"="*64}')

    all_rows, best_rows = [], []
    for fname in datasets:
        path = os.path.join(DATA_DIR, fname)
        stem = os.path.splitext(fname)[0]
        X, y, meta = load_data(path)
        print(f'\n[{stem}]  N={meta["N"]}  D={X.shape[1]}  '
              f'异常率={meta["anomaly_rate"]*100:.1f}%')

        rows = run_one(path, device)
        all_rows.extend(rows)

        best = max(rows, key=lambda r: r['auc'])
        base = next(r for r in rows if r['lambda_scatter'] == 0.0)
        gain = best['auc'] - base['auc']
        best_rows.append(dict(dataset=stem, best_lambda=best['lambda_scatter'],
                              best_auc=best['auc'], best_nu=best['best_nu'],
                              baseline_auc=base['auc'], gain_vs_baseline=round(gain, 6)))
        print(f'  → 最佳: λ={best["lambda_scatter"]}  AUC={best["auc"]:.4f}'
              f'  (基线λ=0: {base["auc"]:.4f}, 增益{gain:+.4f})')

        # 增量保存
        pd.DataFrame(all_rows).to_csv(ALL_CSV, index=False)
        pd.DataFrame(best_rows).to_csv(BEST_CSV, index=False)

    # ── 汇总：每数据集最佳 λ ──────────────────────────────────────────────────
    print(f'\n{"="*74}')
    print(f'每数据集最佳 lambda（dim={LATENT_DIM} 固定）')
    print(f'{"Dataset":<38}{"bestλ":>8}{"bestAUC":>9}{"baseAUC":>9}{"gain":>8}')
    print(f'{"-"*74}')
    for r in best_rows:
        print(f'{r["dataset"][:37]:<38}{r["best_lambda"]:>8.1f}'
              f'{r["best_auc"]:>9.4f}{r["baseline_auc"]:>9.4f}'
              f'{r["gain_vs_baseline"]:>+8.4f}')
    print(f'{"="*74}')
    print(f'\n明细: {ALL_CSV}\n最佳: {BEST_CSV}')
