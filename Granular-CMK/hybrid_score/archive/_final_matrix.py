"""合并 scatter(29 + 扩展) 与全部候选算法 AUC，输出 full_matrix.csv。"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import roc_auc_score

ER  = r'C:\OD\Shihao\Experimental_results'
DS  = r'C:\OD\Shihao\datasets'
HB  = r'C:\OD\Shihao\5\result\hybrid_score\hybrid_best.csv'
EXT = r'C:\OD\Shihao\5\Granular-CMK\hybrid_score\scatter_ext.csv'
OUT = r'C:\OD\Shihao\5\Granular-CMK\hybrid_score\full_matrix.csv'

POOL = ['Disent_AD', 'GBRAD', 'NGBAD', 'GBMOD', 'GBNOF', 'MFIOD', 'DFNO', 'KFGOD',
        'MFGAD', 'WFRDA', 'ILGNI', 'NHOD', 'BLDOD', 'FGAS', 'GBFRD',
        'ECOD', 'DCROD', 'ROD', 'COPOD', 'DeepSVDD', 'DIF', 'VAE']


def load_label(stem):
    p = os.path.join(DS, stem + '.mat')
    if not os.path.exists(p):
        return None
    d = sio.loadmat(p)
    return (d['trandata'][:, -1] != 0).astype(int) if 'trandata' in d else None


def find_main_mat(alg, stem):
    folder = os.path.join(ER, f'{alg}_results', stem)
    if not os.path.isdir(folder):
        return None
    exact = os.path.join(folder, f'{stem}_{alg}.mat')
    if os.path.exists(exact):
        return exact
    c = [f for f in os.listdir(folder) if f.endswith(f'{stem}_{alg}.mat')]
    if c:
        return os.path.join(folder, sorted(c, key=len)[0])
    c = [f for f in os.listdir(folder)
         if f.endswith(f'_{alg}.mat') and '_k-' not in f and '_run' not in f]
    return os.path.join(folder, sorted(c, key=len)[0]) if c else None


def alg_auc(alg, stem, y):
    p = find_main_mat(alg, stem)
    if p is None:
        return None
    try:
        s = np.asarray(sio.loadmat(p)['opt_out_scores'])[:, 0].ravel()
        return roc_auc_score(y, s) if len(s) == len(y) else None
    except Exception:
        return None


if __name__ == '__main__':
    best = pd.read_csv(HB)
    if 'emb_mode' in best.columns:
        best = best[best['emb_mode'] == 'max_ensemble']
    scatter = {r['dataset']: r['best_auc'] for _, r in best.iterrows()}

    if os.path.exists(EXT):
        ext = pd.read_csv(EXT)
        for _, r in ext.iterrows():
            scatter[r['dataset']] = r['scatter_auc']

    rows = []
    for stem, sc in scatter.items():
        y = load_label(stem)
        if y is None:
            continue
        rec = {'dataset': stem, 'N': len(y), 'anom': round(y.mean(), 3),
               'scatter': round(float(sc), 4)}
        for alg in POOL:
            a = alg_auc(alg, stem, y)
            rec[alg] = round(a, 4) if a is not None else np.nan
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f'full_matrix: {len(df)} datasets x {len(POOL)} algs  -> {OUT}')
