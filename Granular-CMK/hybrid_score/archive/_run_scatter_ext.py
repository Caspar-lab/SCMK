"""在 datasets 文件夹的扩展数据集上跑 scatter 网格，补充候选池。"""
import os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r'C:\OD\Shihao\5\Granular-CMK\hybrid_score')
from run_hybrid_score import (load_data, gauss_med_kernels, score_config,
                              LATENT_DIMS, LAMBDAS)

DS      = r'C:\OD\Shihao\datasets'
SURVEY  = r'C:\OD\Shihao\5\Granular-CMK\hybrid_score\survey.csv'
OUT     = r'C:\OD\Shihao\5\Granular-CMK\hybrid_score\scatter_ext.csv'

# 候选筛选：中小规模 + 未跑scatter + 对比算法覆盖足够
MAX_N   = 4000
MIN_ALG = 10

if __name__ == '__main__':
    sv = pd.read_csv(SURVEY)
    cand = sv[(sv['N'] <= MAX_N) & (~sv['done']) & (sv['n_alg'] >= MIN_ALG)]
    cand = cand.sort_values('opp_max').reset_index(drop=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'扩展候选 {len(cand)} 个 (N<={MAX_N}, 未跑, n_alg>={MIN_ALG}):  device={device}')
    print(cand[['dataset', 'N', 'D', 'opp_max']].to_string(index=False))
    print('=' * 70)

    rows = []
    for _, r in cand.iterrows():
        stem = r['dataset']
        path = os.path.join(DS, stem + '.mat')
        X, y, meta = load_data(path)
        kernels = gauss_med_kernels(X[y == 0])
        t0 = time.time()
        best = (-1.0, None, None)
        for dim in LATENT_DIMS:
            for lam in LAMBDAS:
                auc, _ = score_config(X, y, kernels, dim, lam, device)
                if auc > best[0]:
                    best = (auc, dim, lam)
        dt = time.time() - t0
        rows.append(dict(dataset=stem, N=meta['N'], D=X.shape[1],
                         best_dim=best[1], best_lambda=best[2],
                         scatter_auc=round(best[0], 4),
                         opp_max=r['opp_max'], elapsed_s=round(dt, 1)))
        gap = best[0] - r['opp_max']
        print(f'{stem:<34} scatter={best[0]:.4f}  opp_max={r["opp_max"]:.4f}  '
              f'gap_vs_max={gap:+.4f}  dim={best[1]} lam={best[2]}  ({dt:.0f}s)')
        pd.DataFrame(rows).to_csv(OUT, index=False)

    print(f'\n存: {OUT}')
