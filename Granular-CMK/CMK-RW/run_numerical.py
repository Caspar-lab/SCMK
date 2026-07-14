"""
run_numerical.py — 在 dataset/numerical/*.mat 上批量运行 CMK-RW
================================================================

同时读取已有的 CMK+OC-SVM 结果做并排对比，方便直观评估改进效果。
"""

import os, sys, time
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CMK_OCSVM import load_data, gauss_med_kernels, TRAIN_CFG
from cmk_rw import cmk_rw_score, LATENT_DIMS, ETA_CFG, RW_CFG

DATA_DIR   = r'D:\Microsoft\documents\博士课题\异常检测\论文\5\dataset\numerical'
RESULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

SUMMARY_OCSVM = os.path.join(RESULT_DIR, 'numerical_cmk_ocsvm_summary.csv')
SUMMARY_RW    = os.path.join(RESULT_DIR, 'numerical_cmk_rw_summary.csv')
ALL_RW        = os.path.join(RESULT_DIR, 'numerical_cmk_rw_all.csv')


def run_one(path, device):
    stem       = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    kernels    = gauss_med_kernels(X[y == 0])
    rows       = []

    for dim in LATENT_DIMS:
        auc, _, eta, elapsed = cmk_rw_score(
            X, y, kernels, dim, device,
            train_cfg=TRAIN_CFG, eta_cfg=ETA_CFG, rw_cfg=RW_CFG)
        eta_mean = eta.mean(axis=0)
        print(f'  dim={dim:>3d}  AUC={auc:.4f}  ({elapsed:.1f}s)'
              f'  eta=[{", ".join(f"{v:.2f}" for v in eta_mean)}]')
        rows.append(dict(
            dataset      = stem,
            N            = meta['N'],
            D_raw        = X.shape[1],
            anomaly_rate = round(meta['anomaly_rate'], 4),
            latent_dim   = dim,
            auc_rw       = round(auc, 6),
            train_time_s = round(elapsed, 2),
        ))

    return rows


if __name__ == '__main__':
    datasets = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.mat'))
    device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 读取 OC-SVM 历史结果（若存在）
    ocsvm_lookup = {}
    if os.path.exists(SUMMARY_OCSVM):
        df_old = pd.read_csv(SUMMARY_OCSVM)
        ocsvm_lookup = dict(zip(df_old['dataset'], df_old['auc_ocsvm']))

    print(f'共 {len(datasets)} 个数据集  设备: {device}')
    print(f'latent_dims={LATENT_DIMS}  eta_iters={ETA_CFG["n_iters"]}  '
          f'rw_k={RW_CFG["k"]}  rw_damping={RW_CFG["damping"]}\n{"─"*70}')

    all_rows, summary = [], []

    for fname in datasets:
        path = os.path.join(DATA_DIR, fname)
        stem = os.path.splitext(fname)[0]
        X, y, meta = load_data(path)
        print(f'\n[{stem}]  N={meta["N"]}  D={X.shape[1]}  '
              f'异常率={meta["anomaly_rate"]*100:.1f}%')

        t_ds = time.time()
        rows = run_one(path, device)
        all_rows.extend(rows)

        best = max(rows, key=lambda r: r['auc_rw'])
        summary.append(best)

        ocsvm_str = ''
        if stem in ocsvm_lookup:
            delta = best['auc_rw'] - ocsvm_lookup[stem]
            sign  = '+' if delta >= 0 else ''
            ocsvm_str = f'  (OC-SVM={ocsvm_lookup[stem]:.4f}  Δ={sign}{delta:.4f})'
        print(f'  → 最优: dim={best["latent_dim"]}  AUC={best["auc_rw"]:.4f}'
              f'{ocsvm_str}  ({time.time()-t_ds:.1f}s 总计)')

        # 每个数据集完成后立即保存
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULT_DIR, f'{stem}_cmk_rw.csv'), index=False)

    pd.DataFrame(all_rows).to_csv(ALL_RW, index=False)
    pd.DataFrame(summary).to_csv(SUMMARY_RW, index=False)

    # ── 汇总对比表 ────────────────────────────────────────────────────────────
    print(f'\n{"="*75}')
    print(f'CMK-RW 汇总（每数据集取最优 latent_dim）')
    if ocsvm_lookup:
        print(f'{"Dataset":<38} {"N":>6} {"Anom%":>6}  {"RW-AUC":>8}  '
              f'{"OC-SVM":>8}  {"Δ":>7}  {"dim":>4}')
        print(f'{"-"*75}')
    else:
        print(f'{"Dataset":<38} {"N":>6} {"Anom%":>6}  {"RW-AUC":>8}  {"dim":>4}')
        print(f'{"-"*60}')

    aucs_rw, aucs_oc = [], []
    for r in summary:
        name     = r['dataset'][:37]
        auc_oc   = ocsvm_lookup.get(r['dataset'], None)
        aucs_rw.append(r['auc_rw'])
        if ocsvm_lookup:
            delta_s = f'{r["auc_rw"] - auc_oc:+.4f}' if auc_oc else '   N/A'
            oc_s    = f'{auc_oc:.4f}' if auc_oc else '   N/A'
            aucs_oc.append(auc_oc)
            print(f'{name:<38} {r["N"]:>6} {r["anomaly_rate"]*100:>5.1f}%  '
                  f'{r["auc_rw"]:>8.4f}  {oc_s:>8}  {delta_s:>7}  {r["latent_dim"]:>4d}')
        else:
            print(f'{name:<38} {r["N"]:>6} {r["anomaly_rate"]*100:>5.1f}%  '
                  f'{r["auc_rw"]:>8.4f}  {r["latent_dim"]:>4d}')

    print(f'{"-"*75}')
    avg_rw = sum(aucs_rw) / len(aucs_rw)
    if aucs_oc:
        avg_oc    = sum(aucs_oc) / len(aucs_oc)
        avg_delta = avg_rw - avg_oc
        sign      = '+' if avg_delta >= 0 else ''
        print(f'{"Average":<38} {"":>6} {"":>6}  {avg_rw:>8.4f}  '
              f'{avg_oc:>8.4f}  {sign}{avg_delta:.4f}')
    else:
        print(f'{"Average":<38} {avg_rw:>8.4f}')
    print(f'{"="*75}')
    print(f'\n明细: {ALL_RW}')
    print(f'汇总: {SUMMARY_RW}')
