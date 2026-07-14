"""
run_numerical_scatter.py — 在 dataset/numerical/*.mat 上批量运行 CMK + 散度惩罚 + OC-SVM
==========================================================================================

对每个数据集扫描 latent_dim，固定 lambda_scatter，记录最优 AUC，
并与原始 CMK+OC-SVM 历史结果（若存在）做并排对比。
"""

import os, sys, time
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CMK_OCSVM import load_data, gauss_med_kernels, LATENT_DIMS, TRAIN_CFG as _BASE_CFG
from CMK_OCSVM_scatter import train_cmk_scatter, eval_ocsvm

DATA_DIR   = r'C:\OD\Shihao\5\dataset\numerical'
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result/cmk_scatter')
os.makedirs(RESULT_DIR, exist_ok=True)

LAMBDA_SCATTER = 1000.0   # 散度惩罚强度

TRAIN_CFG = {**_BASE_CFG, 'lambda_scatter': LAMBDA_SCATTER}

SUMMARY       = os.path.join(RESULT_DIR, 'numerical_cmk_scatter_summary.csv')
ALL_CSV       = os.path.join(RESULT_DIR, 'numerical_cmk_scatter_all.csv')
SUMMARY_OCSVM = os.path.join(RESULT_DIR, 'numerical_cmk_ocsvm_summary.csv')


def run_one(path, device):
    stem       = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    kernels    = gauss_med_kernels(X[y == 0])
    rows       = []

    for dim in LATENT_DIMS:
        t0       = time.time()
        model    = train_cmk_scatter(X, y, kernels, dim, device, TRAIN_CFG)
        auc, nu  = eval_ocsvm(model, X, y, device)
        elapsed  = time.time() - t0
        rows.append(dict(dataset=stem, latent_dim=dim,
                         lambda_scatter=LAMBDA_SCATTER,
                         auc_scatter=round(auc, 6), best_nu=nu,
                         elapsed_s=round(elapsed, 2)))
        print(f'  dim={dim:>3d}  AUC={auc:.4f}  nu={nu:.2f}  ({elapsed:.1f}s)')

    return rows


if __name__ == '__main__':
    datasets = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.mat'))
    device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ocsvm_lookup = {}
    if os.path.exists(SUMMARY_OCSVM):
        df_old = pd.read_csv(SUMMARY_OCSVM)
        ocsvm_lookup = dict(zip(df_old['dataset'], df_old['auc_ocsvm']))

    print(f'共 {len(datasets)} 个数据集  设备: {device}  λ_scatter={LAMBDA_SCATTER}')
    print(f'{"="*64}')

    all_rows, summary = [], []
    for fname in datasets:
        path = os.path.join(DATA_DIR, fname)
        stem = os.path.splitext(fname)[0]
        X, y, meta = load_data(path)
        print(f'\n[{stem}]  N={meta["N"]}  D={X.shape[1]}  '
              f'异常率={meta["anomaly_rate"]*100:.1f}%')
        rows = run_one(path, device)
        all_rows.extend(rows)
        best = max(rows, key=lambda r: r['auc_scatter'])
        summary.append(best)

        oc_str = ''
        if stem in ocsvm_lookup:
            delta = best['auc_scatter'] - ocsvm_lookup[stem]
            sign  = '+' if delta >= 0 else ''
            oc_str = f'  (OC-SVM={ocsvm_lookup[stem]:.4f}  Δ={sign}{delta:.4f})'
        print(f'  → 最优: dim={best["latent_dim"]}  AUC={best["auc_scatter"]:.4f}{oc_str}')

        pd.DataFrame(rows).to_csv(
            os.path.join(RESULT_DIR, f'{stem}_cmk_scatter.csv'), index=False)

    pd.DataFrame(all_rows).to_csv(ALL_CSV, index=False)
    pd.DataFrame(summary).to_csv(SUMMARY, index=False)

    # ── 汇总对比 ──────────────────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print(f'CMK+Scatter 汇总（每数据集取最优 latent_dim，λ={LAMBDA_SCATTER}）')
    if ocsvm_lookup:
        print(f'{"Dataset":<34} {"Scatter":>8}  {"OC-SVM":>8}  {"Δ":>8}  {"dim":>4}')
        print(f'{"-"*72}')
    else:
        print(f'{"Dataset":<34} {"Scatter":>8}  {"dim":>4}')
        print(f'{"-"*52}')

    aucs_s, aucs_o = [], []
    for r in summary:
        aucs_s.append(r['auc_scatter'])
        if ocsvm_lookup and r['dataset'] in ocsvm_lookup:
            oc = ocsvm_lookup[r['dataset']]
            aucs_o.append(oc)
            d = r['auc_scatter'] - oc
            print(f'{r["dataset"]:<34} {r["auc_scatter"]:>8.4f}  {oc:>8.4f}  '
                  f'{d:>+8.4f}  {r["latent_dim"]:>4d}')
        else:
            print(f'{r["dataset"]:<34} {r["auc_scatter"]:>8.4f}  {r["latent_dim"]:>4d}')

    print(f'{"-"*72}')
    avg_s = sum(aucs_s) / len(aucs_s)
    if aucs_o:
        avg_o = sum(aucs_o) / len(aucs_o)
        print(f'{"Average":<34} {avg_s:>8.4f}  {avg_o:>8.4f}  {avg_s-avg_o:>+8.4f}')
    else:
        print(f'{"Average":<34} {avg_s:>8.4f}')
    print(f'{"="*72}')
    print(f'\n明细: {ALL_CSV}\n汇总: {SUMMARY}')
