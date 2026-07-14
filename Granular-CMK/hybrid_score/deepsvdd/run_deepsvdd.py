"""
run_deepsvdd.py — PyOD DeepSVDD 在 24 个对比数据集上的统一实验
==============================================================

为与 SCMK 口径一致：
  - 预处理复用 CMK_OCSVM.load_data（标称 OneHot + 数值 MinMax）
  - 半监督协议：仅正常样本 X[y==0] 训练，对全量 X 打分
  - 超参网格搜索取最优 AUC（oracle，与论文其它对比算法一致）

统一用 PyOD 重跑全部 24 个（而非混用他人实现），保证 DeepSVDD 整列可比，
并补全原 Experimental_results 中缺失的 4 个数据集。

输出（本目录）：
  deepsvdd_best.csv : dataset, N, D, anomaly_rate, best_auc, hidden_neurons, epochs, fit_s
  deepsvdd_all.csv  : dataset, hidden_neurons, epochs, auc, fit_s   （逐配置明细）
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

_DSV = os.path.dirname(os.path.abspath(__file__))
_HS  = os.path.dirname(_DSV)                       # hybrid_score
_GCMK = os.path.dirname(_HS)                       # Granular-CMK
sys.path.insert(0, _GCMK)
from CMK_OCSVM import load_data
from pyod.models.deep_svdd import DeepSVDD

# ─── 配置 ──────────────────────────────────────────────────────────────────────
DATA_DIR = r'C:\OD\Shihao\datasets'
SEL_CSV  = os.path.join(_HS, 'selection.csv')      # 24 个数据集来源
BEST_CSV = os.path.join(_DSV, 'deepsvdd_best.csv')
ALL_CSV  = os.path.join(_DSV, 'deepsvdd_all.csv')

# 超参网格（结构 × 训练轮数）
HIDDEN_NEURONS = [[64, 32], [128, 64], [32, 16]]
EPOCHS         = [50, 100]
BATCH_SIZE     = 64
DROPOUT        = 0.2
SEED           = 42


def run_one(stem):
    path = os.path.join(DATA_DIR, stem + '.mat')
    if not os.path.exists(path):
        print(f'[skip] {stem}: 找不到 {path}')
        return None, None
    X, y, meta = load_data(path)
    X = X.astype(np.float32)
    Xn = X[y == 0]                                  # 半监督：仅正常样本训练
    nfeat = X.shape[1]

    rows, best = [], (-1.0, None, None, 0.0)
    for hn in HIDDEN_NEURONS:
        for ep in EPOCHS:
            t0 = time.time()
            try:
                clf = DeepSVDD(n_features=nfeat, hidden_neurons=hn, epochs=ep,
                               batch_size=BATCH_SIZE, dropout_rate=DROPOUT,
                               random_state=SEED, verbose=0)
                clf.fit(Xn)
                scores = clf.decision_function(X)   # 越大越异常
                auc = roc_auc_score(y, scores)
            except Exception as e:
                print(f'    [err] {stem} hn={hn} ep={ep}: {e}')
                auc = float('nan')
            dt = time.time() - t0
            rows.append(dict(dataset=stem, hidden_neurons=str(hn), epochs=ep,
                             auc=round(auc, 6), fit_s=round(dt, 1)))
            if not np.isnan(auc) and auc > best[0]:
                best = (auc, hn, ep, dt)
            print(f'    hn={str(hn):<10} ep={ep:<4} AUC={auc:.4f}  ({dt:.1f}s)')

    brow = dict(dataset=stem, N=meta['N'], D=nfeat,
                anomaly_rate=round(meta['anomaly_rate'], 4),
                best_auc=round(best[0], 6), hidden_neurons=str(best[1]),
                epochs=best[2], fit_s=round(best[3], 1))
    print(f'  → {stem:<34} best AUC={best[0]:.4f}  (hn={best[1]}, ep={best[2]})')
    return rows, brow


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass

    datasets = list(pd.read_csv(SEL_CSV)['dataset'])
    print(f'DeepSVDD (PyOD) 实验：{len(datasets)} 个数据集  '
          f'网格 hidden={HIDDEN_NEURONS} × epochs={EPOCHS}')
    print('=' * 72)

    all_rows, best_rows = [], []
    for stem in datasets:
        print(f'\n[{stem}]')
        rows, brow = run_one(stem)
        if rows is None:
            continue
        all_rows.extend(rows)
        best_rows.append(brow)
        pd.DataFrame(all_rows).to_csv(ALL_CSV, index=False)
        pd.DataFrame(best_rows).to_csv(BEST_CSV, index=False)

    df = pd.DataFrame(best_rows)
    print(f'\n{"="*72}\n完成 {len(df)} 个数据集，平均 best AUC = {df["best_auc"].mean():.4f}')
    print(f'明细: {ALL_CSV}\n汇总: {BEST_CSV}')
