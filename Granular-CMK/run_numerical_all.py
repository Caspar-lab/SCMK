"""
run_numerical_all.py — 对 dataset/numerical 中所有数据集批量运行 CMK + OC-SVM
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CMK_OCSVM import (load_data, gauss_med_kernels, train_cmk,
                        get_embeddings, best_nu_ocsvm,
                        LATENT_DIMS, NU_CANDIDATES, TRAIN_CFG)

DATA_DIR   = r'D:\Microsoft\documents\博士课题\异常检测\论文\5\dataset\numerical'
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(RESULT_DIR, 'numerical_cmk_ocsvm_summary.csv')


def run_one(path, device):
    """对单个数据集运行完整的 latent_dim 扫描，返回各维度结果列表。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    n_normal   = int((y == 0).sum())
    kernels    = gauss_med_kernels(X[y == 0])

    rows = []
    for latent_dim in LATENT_DIMS:
        t0    = time.time()
        model = train_cmk(X, y, kernels, latent_dim, device, TRAIN_CFG)
        H_all = get_embeddings(model, X, device)
        H_normal_emb = H_all[y == 0]
        best_nu, best_auc = best_nu_ocsvm(H_all, H_normal_emb, y, NU_CANDIDATES)
        elapsed = time.time() - t0

        rows.append(dict(
            dataset      = stem,
            N            = meta['N'],
            anomaly_rate = round(meta['anomaly_rate'], 4),
            n_train      = n_normal,
            latent_dim   = latent_dim,
            auc_ocsvm    = round(best_auc, 6),
            best_nu      = best_nu,
            train_time_s = round(elapsed, 2),
        ))
        print(f'  dim={latent_dim:>3d}  AUC={best_auc:.4f}  nu={best_nu:.2f}  ({elapsed:.1f}s)')

    return rows


if __name__ == '__main__':
    datasets = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.mat'))
    device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    all_rows    = []
    summary     = []   # 每个数据集只记录最优 latent_dim 那行，用于汇总表

    sep = '─' * 65
    print(f'共 {len(datasets)} 个数据集，设备: {device}\n{sep}')

    for fname in datasets:
        path = os.path.join(DATA_DIR, fname)
        stem = os.path.splitext(fname)[0]
        X, y, meta = load_data(path)
        print(f'\n[{stem}]  N={meta["N"]}  D={X.shape[1]}  '
              f'异常率={meta["anomaly_rate"]*100:.1f}%')

        t_ds = time.time()
        rows = run_one(path, device)
        all_rows.extend(rows)

        # 取该数据集最优维度的行写入汇总
        best_row = max(rows, key=lambda r: r['auc_ocsvm'])
        summary.append(best_row)
        print(f'  → 最优: dim={best_row["latent_dim"]}  AUC={best_row["auc_ocsvm"]:.4f}'
              f'  ({time.time()-t_ds:.1f}s 总计)')

        # 每跑完一个数据集立即保存当前数据集的结果（只保存 rows，不含其他数据集）
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULT_DIR, f'{stem}_cmk_ocsvm.csv'), index=False)

    # ── 保存全量明细和汇总 ──────────────────────────────────────────────────────
    pd.DataFrame(all_rows).to_csv(
        os.path.join(RESULT_DIR, 'numerical_cmk_ocsvm_all.csv'), index=False)
    pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)

    # ── 打印汇总表 ────────────────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print(f'CMK + OC-SVM  汇总（每数据集取最优 latent_dim）')
    print(f'{"Dataset":<42} {"N":>6} {"Anom%":>6}  {"AUC":>7}  {"dim":>4}  {"nu":>5}')
    print(f'{"-"*72}')
    aucs = []
    for r in summary:
        name = r['dataset'][:41]
        print(f'{name:<42} {r["N"]:>6} {r["anomaly_rate"]*100:>5.1f}%  '
              f'{r["auc_ocsvm"]:>7.4f}  {r["latent_dim"]:>4d}  {r["best_nu"]:>5.2f}')
        aucs.append(r['auc_ocsvm'])
    print(f'{"-"*72}')
    print(f'{"Average":<42} {"":>6} {"":>6}  {sum(aucs)/len(aucs):>7.4f}')
    print(f'{"="*72}')
    print(f'\n明细: {os.path.join(RESULT_DIR, "numerical_cmk_ocsvm_all.csv")}')
    print(f'汇总: {SUMMARY_CSV}')
