"""Update DFNO_mean in master_compare_v2.csv to full-data transductive values, back up
the old file, then print the new comparison averages and pairwise-win totals."""
import os, shutil
import numpy as np, pandas as pd

MASTER = 'C:/OD/Shihao/5/result/hybrid_score_semi/master_compare_v2.csv'
DFNO_FULL = {  # stem -> full-data DFNO AUC (opt_out_scores[:,0] over all objects)
    'vertebral': 0.37270, 'thyroid': 0.95295, 'wbc_malignant_39_variant1': 0.99711,
    'glass': 0.85745, 'ecoli': 0.90520, 'pageblocks_1_258_variant1': 0.97469,
    'wine': 0.95714, 'cardio': 0.91419, 'cardiotocography_2and3_33_variant1': 0.80948,
    'tic_tac_toe_negative_12_variant1': 1.00000, 'tic_tac_toe_negative_69_variant1': 0.99882,
    'wpbc_variant1': 0.55629, 'ionosphere_b_24_variant1': 1.00000, 'zoo_variant1': 0.93417,
    'sick_sick_72_variant1': 0.83272, 'autos_variant1': 0.62267, 'annealing_variant1': 0.75898,
    'lymphography': 0.99765, 'bands_band_6_variant1': 0.90919, 'audiology_variant1': 0.90283}

if not os.path.exists(MASTER + '.bak_seed2dfno'):
    shutil.copy(MASTER, MASTER + '.bak_seed2dfno')
df = pd.read_csv(MASTER)
df['DFNO_mean'] = df['stem'].map(DFNO_FULL)
assert df['DFNO_mean'].notna().all()
df.to_csv(MASTER, index=False)

ALGS = ['KFGOD', 'Disent', 'DeepSVDD', 'DFNO', 'LMKAD', 'ICL', 'NeuTraLAD']
print('=== new comparison means ===')
for a in ALGS + ['SCMK']:
    print(f'  {a:<10} {df[a+"_mean"].mean():.4f}')
scmk = df['SCMK_mean'].values
tot = 0
print('=== SCMK pairwise wins per baseline (SCMK_mean > alg_mean) ===')
for a in ALGS:
    w = int((scmk > df[a + '_mean'].values).sum()); tot += w
    print(f'  {a:<10} {w}/20')
print(f'  TOTAL wins = {tot}/140')
print('=== SCMK best-mean count ===')
cols = [a + '_mean' for a in ALGS] + ['SCMK_mean']
M = df[cols].values
best = (M.argmax(axis=1) == len(ALGS)).sum()
# ties where SCMK equals row max
tie = sum(1 for i in range(len(M)) if M[i, -1] == M[i].max())
print(f'  SCMK strict-best rows = {best}; rows where SCMK == row max (incl ties) = {tie}')
