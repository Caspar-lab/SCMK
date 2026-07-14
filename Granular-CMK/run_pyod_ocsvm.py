"""
run_pyod_ocsvm.py — pyod 官方 OCSVM 在实验数据集上的 nu 扫描基线
=================================================================

协议（由 TRAIN_ON_NORMAL 切换）：
  TRAIN_ON_NORMAL=True  半监督：仅用正常样本 X[y==0] 拟合，再对全量打分
                        （与 CMK_OCSVM_scatter.eval_ocsvm 一致）→ result/pyod_ocsvm_semi/
  TRAIN_ON_NORMAL=False 无监督：全量 X 拟合，pyod/ADBench 标准用法
                        → result/pyod_ocsvm/
预处理：复用 CMK_OCSVM.load_data（标称列 OneHot + 数值列 MinMax），
        与 CMK 管线完全一致，结果可直接对比。
跳过：N>10000 的大数据集（RBF OCSVM 在大 N 上 O(N^2~3) 极慢），
      以及 *_datalists_outlier.mat（仅为数据集名列表，非数据）。
"""

import os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pyod.models.ocsvm import OCSVM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CMK_OCSVM import load_data

TRAIN_ON_NORMAL = True   # True=半监督(仅正常样本拟合)  False=无监督(全量拟合)

DATA_DIR   = r'D:\Microsoft\documents\博士课题\异常检测\实验\datasets'
_sub       = 'pyod_ocsvm_semi' if TRAIN_ON_NORMAL else 'pyod_ocsvm'
RESULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result', _sub)
os.makedirs(RESULT_DIR, exist_ok=True)

NU_GRID  = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
MAX_N    = 10000   # 跳过 N 超过此值的数据集
SKIP_KEY = 'datalists_outlier'

ALL_CSV     = os.path.join(RESULT_DIR, 'pyod_ocsvm_all.csv')
SUMMARY_CSV = os.path.join(RESULT_DIR, 'pyod_ocsvm_summary.csv')


def run_one(path):
    """对单个数据集扫描 nu，返回 (rows, summary_row)。N 过大返回 (None, None)。"""
    stem       = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    N          = meta['N']
    if N > MAX_N:
        print(f'[skip] {stem}  N={N} > {MAX_N}')
        return None, None

    X_fit = X[y == 0] if TRAIN_ON_NORMAL else X
    print(f'\n[{stem}]  N={N}  D={X.shape[1]}  异常率={meta["anomaly_rate"]*100:.1f}%'
          f'  拟合样本={len(X_fit)}')
    rows = []
    for nu in NU_GRID:
        t0  = time.time()
        clf = OCSVM(nu=nu)                 # pyod 默认 kernel=rbf, gamma=auto
        clf.fit(X_fit)                     # 半监督=仅正常样本 / 无监督=全量
        # 半监督：对全量打分（含异常）；无监督：训练集即全量
        scores = clf.decision_function(X) if TRAIN_ON_NORMAL else clf.decision_scores_
        auc = roc_auc_score(y, scores)
        n_sv = int(clf.detector_.support_.shape[0]) if hasattr(clf, 'detector_') else -1
        elapsed = time.time() - t0
        rows.append(dict(dataset=stem, nu=nu, auc=round(auc, 6),
                         n_support=n_sv, fit_time_s=round(elapsed, 2)))
        print(f'  nu={nu:>4}  AUC={auc:.4f}  n_sv={n_sv:<5}  ({elapsed:.1f}s)')

    best = max(rows, key=lambda r: r['auc'])
    summary = dict(dataset=stem, N=N, D=X.shape[1],
                   anomaly_rate=round(meta['anomaly_rate'], 4),
                   best_nu=best['nu'], best_auc=best['auc'])
    for r in rows:                          # 附 nu 敏感性列
        summary[f'auc@{r["nu"]}'] = r['auc']
    return rows, summary


if __name__ == '__main__':
    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.endswith('.mat') and SKIP_KEY not in f)
    print(f'共 {len(files)} 个 .mat 候选  nu={NU_GRID}  跳过 N>{MAX_N}\n{"="*64}')

    all_rows, summary = [], []
    for fname in files:
        try:
            rows, srow = run_one(os.path.join(DATA_DIR, fname))
        except Exception as e:
            print(f'[ERR] {fname}: {e}')
            continue
        if rows is None:
            continue
        all_rows.extend(rows)
        summary.append(srow)
        stem = os.path.splitext(fname)[0]
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULT_DIR, f'{stem}_ocsvm.csv'), index=False)
        pd.DataFrame(all_rows).to_csv(ALL_CSV, index=False)
        pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)

    # ── nu 推荐 ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(summary)
    print(f'\n{"="*70}')
    print(f'完成 {len(df)} 个数据集。')
    print(f'\n各 nu 在全部数据集上的平均 AUC：')
    mean_by_nu = {nu: df[f'auc@{nu}'].mean() for nu in NU_GRID}
    for nu, m in mean_by_nu.items():
        print(f'  nu={nu:>4}  mean AUC={m:.4f}')
    best_mean_nu = max(mean_by_nu, key=mean_by_nu.get)

    print(f'\n各 nu 成为"逐数据集最优"的次数：')
    mode = df['best_nu'].value_counts().sort_index()
    for nu in NU_GRID:
        print(f'  nu={nu:>4}  best on {int(mode.get(nu, 0))} datasets')
    mode_nu = df['best_nu'].value_counts().idxmax()

    print(f'\n>>> 推荐 nu：')
    print(f'    平均 AUC 最高 → nu={best_mean_nu}  (mean AUC={mean_by_nu[best_mean_nu]:.4f})')
    print(f'    逐数据集最优众数 → nu={mode_nu}')
    print(f'{"="*70}')
    print(f'\n明细: {ALL_CSV}\n汇总: {SUMMARY_CSV}')
