"""把扩展数据集 scatter 结果并入 result/hybrid_score 的 hybrid_best.csv 与 hybrid_all.csv。

扩展数据集只跑了网格最优（scatter_ext.csv 仅 best 配置），故：
  - hybrid_best.csv：追加每个扩展数据集的 best 行（group='extended'）
  - hybrid_all.csv ：追加每个扩展数据集的 best 配置行（group='extended'，
                     代表该数据集的最优网格点；非全网格明细）
已存在同名 dataset 的不重复追加。
"""
import pandas as pd

EXT      = r'C:\OD\Shihao\5\Granular-CMK\hybrid_score\scatter_ext.csv'
BEST_CSV = r'C:\OD\Shihao\5\result\hybrid_score\hybrid_best.csv'
ALL_CSV  = r'C:\OD\Shihao\5\result\hybrid_score\hybrid_all.csv'
GROUP    = 'extended'

ext  = pd.read_csv(EXT)
best = pd.read_csv(BEST_CSV)
alld = pd.read_csv(ALL_CSV)

exist_best = set(best['dataset'])
exist_all  = set(alld['dataset'])

# ── hybrid_best.csv ──
new_best = [
    dict(dataset=r['dataset'], group=GROUP,
         best_dim=int(r['best_dim']), best_lambda=r['best_lambda'],
         best_auc=round(float(r['scatter_auc']), 6))
    for _, r in ext.iterrows() if r['dataset'] not in exist_best
]
best2 = pd.concat([best, pd.DataFrame(new_best)], ignore_index=True)
best2 = best2.sort_values(['group', 'dataset']).reset_index(drop=True)
best2.to_csv(BEST_CSV, index=False)

# ── hybrid_all.csv ──
new_all = [
    dict(dataset=r['dataset'], group=GROUP,
         latent_dim=int(r['best_dim']), lambda_scatter=r['best_lambda'],
         auc=round(float(r['scatter_auc']), 6),
         train_s=r.get('elapsed_s', float('nan')))
    for _, r in ext.iterrows() if r['dataset'] not in exist_all
]
all2 = pd.concat([alld, pd.DataFrame(new_all)], ignore_index=True)
all2.to_csv(ALL_CSV, index=False)

print(f'hybrid_best.csv: +{len(new_best)} 行 -> 共 {len(best2)} 行')
print(f'hybrid_all.csv : +{len(new_all)} 行 -> 共 {len(all2)} 行')
print('\n各组数据集数 (best):')
print(best2.groupby('group')['dataset'].nunique().to_string())
print(f'\n全部数据集平均 best_auc = {best2["best_auc"].mean():.4f}')
