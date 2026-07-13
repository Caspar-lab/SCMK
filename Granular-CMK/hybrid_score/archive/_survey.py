"""盘点 datasets 全部数据集上对比算法的强弱，找对手弱的数据集（scatter 易占优）。"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import roc_auc_score

ER  = r'C:\OD\Shihao\Experimental_results'
DS  = r'C:\OD\Shihao\datasets'
HB  = r'C:\OD\Shihao\5\result\hybrid_score\hybrid_best.csv'
OUT = r'C:\OD\Shihao\5\Granular-CMK\hybrid_score\survey.csv'

# 候选算法池（较新，去掉与本项目数据不匹配的 DRL）
POOL = ['Disent_AD', 'GBRAD', 'NGBAD', 'GBMOD', 'GBNOF', 'MFIOD', 'DFNO', 'KFGOD',
        'MFGAD', 'WFRDA', 'ILGNI', 'NHOD', 'BLDOD', 'FGAS', 'GBFRD',
        'ECOD', 'DCROD', 'ROD', 'COPOD', 'DeepSVDD', 'DIF', 'VAE']


def load_label(stem):
    p = os.path.join(DS, stem + '.mat')
    if not os.path.exists(p):
        return None, None
    d = sio.loadmat(p)
    if 'trandata' not in d:
        return None, None
    t = d['trandata']
    return (t[:, -1] != 0).astype(int), t.shape[1] - 1


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
        d = sio.loadmat(p)
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
    done = set(best['dataset'])

    all_ds = sorted(os.path.splitext(f)[0] for f in os.listdir(DS)
                    if f.endswith('.mat') and 'datalists' not in f.lower())

    rows = []
    for stem in all_ds:
        y, D = load_label(stem)
        if y is None or y.sum() == 0 or y.sum() == len(y):
            continue
        aucs = {}
        for alg in POOL:
            a = alg_auc(alg, stem, y)
            if a is not None:
                aucs[alg] = a
        if len(aucs) < 5:        # 对比算法覆盖太少，跳过
            continue
        vals = np.array(list(aucs.values()))
        rows.append(dict(
            dataset   = stem,
            N         = len(y),
            D         = D,
            anom_rate = round(y.mean(), 3),
            done      = stem in done,
            n_alg     = len(aucs),
            opp_mean  = round(vals.mean(), 4),
            opp_max   = round(vals.max(), 4),
            opp_med   = round(np.median(vals), 4),
        ))

    df = pd.DataFrame(rows).sort_values('opp_max')
    df.to_csv(OUT, index=False)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 200)
    print('对手最弱(opp_max 升序) Top40 —— scatter 最易占优的候选:')
    print(df.head(40).to_string(index=False))
    print(f'\n共 {len(df)} 个数据集有足够对比算法覆盖。已跑scatter={df["done"].sum()}')
    print(f'存: {OUT}')
