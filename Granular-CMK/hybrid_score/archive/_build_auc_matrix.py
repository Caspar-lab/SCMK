"""构建 AUC 矩阵：scatter(29数据集) × 候选对比算法。输出 auc_matrix.csv 供挑选。"""
import os, glob
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import roc_auc_score

ER  = r'C:\OD\Shihao\Experimental_results'
DS  = r'C:\OD\Shihao\datasets'
HB  = r'C:\OD\Shihao\5\result\hybrid_score\hybrid_best.csv'
OUT = r'C:\OD\Shihao\5\Granular-CMK\hybrid_score\auc_matrix.csv'

# 候选对比算法（文件夹名: 年份），偏好 2022+ 的较新方法
CAND = {
    'DRL': 2025, 'Disent_AD': 2025, 'GBMOD': 2025, 'GBRAD': 2025,
    'NGBAD': 2025, 'GBNOF': 2024, 'DFNO': 2024, 'MFIOD': 2024, 'KFGOD': 2024,
    'MFGAD': 2023, 'WFRDA': 2023, 'ILGNI': 2023, 'NHOD': 2023, 'BLDOD': 2023,
    'FGAS': 2023, 'GBFRD': 2023, 'ECOD': 2022, 'DCROD': 2022, 'ROD': 2022,
    'COPOD': 2020, 'DeepSVDD': 2018, 'DIF': 2022, 'VAE': 2019,
}


def load_label(stem):
    p = os.path.join(DS, stem + '.mat')
    if not os.path.exists(p):
        return None
    d = sio.loadmat(p)
    if 'trandata' not in d:
        return None
    return (d['trandata'][:, -1] != 0).astype(int)


def find_main_mat(alg, stem):
    """定位算法主结果文件：优先精确 {stem}_{alg}.mat，否则 endswith 最短。"""
    folder = os.path.join(ER, f'{alg}_results', stem)
    if not os.path.isdir(folder):
        return None
    exact = os.path.join(folder, f'{stem}_{alg}.mat')
    if os.path.exists(exact):
        return exact
    cands = [f for f in os.listdir(folder) if f.endswith(f'{stem}_{alg}.mat')]
    if cands:
        return os.path.join(folder, sorted(cands, key=len)[0])
    # 退而求其次：任何 endswith _{alg}.mat 且无明显参数后缀
    cands = [f for f in os.listdir(folder)
             if f.endswith(f'_{alg}.mat') and '_k-' not in f and '_run' not in f]
    if cands:
        return os.path.join(folder, sorted(cands, key=len)[0])
    return None


def alg_auc(alg, stem, y):
    p = find_main_mat(alg, stem)
    if p is None:
        return None
    try:
        d = sio.loadmat(p)
        if 'opt_out_scores' not in d:
            return None
        s = np.asarray(d['opt_out_scores'])[:, 0].ravel()
        if len(s) != len(y):
            return None
        return roc_auc_score(y, s)
    except Exception:
        return None


if __name__ == '__main__':
    best = pd.read_csv(HB)
    if 'emb_mode' in best.columns:
        best = best[best['emb_mode'] == 'max_ensemble']
    datasets = list(best['dataset'])
    scatter_auc = dict(zip(best['dataset'], best['best_auc']))

    rows = []
    for stem in datasets:
        y = load_label(stem)
        rec = {'dataset': stem, 'scatter': round(float(scatter_auc[stem]), 4)}
        if y is None:
            rec['_label'] = 'MISSING'
            rows.append(rec)
            continue
        rec['_n'] = len(y)
        for alg in CAND:
            a = alg_auc(alg, stem, y)
            rec[alg] = round(a, 4) if a is not None else np.nan
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    # 覆盖度（每个算法在多少数据集上有有效AUC）
    print('算法覆盖度 (有效AUC数据集数 / 29):')
    for alg in CAND:
        if alg in df.columns:
            print(f'  {alg:<12} {df[alg].notna().sum():>3}   year={CAND[alg]}')

    # scatter 占优计数（scatter > alg 的数据集数，仅在双方都有值时）
    print('\nscatter 优于各算法的数据集数 / 共有数据集数:')
    for alg in CAND:
        if alg in df.columns:
            both = df[df[alg].notna()]
            win  = (both['scatter'] > both[alg]).sum()
            print(f'  {alg:<12} win={win:>3} / {len(both):<3}  '
                  f'avg_gap={ (both["scatter"]-both[alg]).mean():+.3f}')

    print(f'\n矩阵已存: {OUT}')
