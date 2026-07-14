"""复刻 plot_roc_compare.ipynb 的逐数据集 ROC 出图（LMKAD 用 python 源），
生成 result/hybrid_score/roc_compare/{stem}_ROC.pdf。供 torch311 python 直接运行。"""
import os, sys
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd, scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import torch

NB_DIR = 'C:/OD/Shihao/5/Granular-CMK/hybrid_score'
sys.path.insert(0, NB_DIR)
from run_hybrid_score import gauss_med_kernels, score_config, load_data

DATA_ROOT  = 'C:/OD/Shihao/datasets'
ER         = 'C:/OD/Shihao/Experimental_results'
RESULT_DIR = 'C:/OD/Shihao/5/result/hybrid_score'
SEL_CSV    = os.path.join(NB_DIR, 'selection.csv')
BEST_CSV   = os.path.join(RESULT_DIR, 'hybrid_best.csv')
KFGOD_DIR  = 'C:/OD/Shihao/KFGOD-main/results'
LMKAD_PY   = 'C:/OD/Shihao/Experimental_results/LMKAD_results_python'

plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'Times New Roman'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

ALG_YEAR = {'KFGOD': 2024, 'LMKAD': 2021, 'GBRAD': 2025, 'Disent_AD': 2025,
            'MFIOD': 2024, 'ILGNI': 2023, 'WFRDA': 2023, 'DIF': 2022, 'ECOD': 2022}
METHODS = list(ALG_YEAR.keys()) + ['scatter']

sel = pd.read_csv(SEL_CSV); DATASETS = list(sel['dataset'])
best = pd.read_csv(BEST_CSV)
if 'emb_mode' in best.columns:
    best = best[best['emb_mode'] == 'max_ensemble']
BEST_CFG = {r['dataset']: (int(r['best_dim']), float(r['best_lambda']))
            for _, r in best.iterrows()}


def load_xy(stem):
    d = sio.loadmat(os.path.join(DATA_ROOT, stem + '.mat'))
    t = d['trandata'].astype(float)
    return t, (t[:, -1] != 0).astype(int)


def get_scatter_scores(stem, y):
    X, yy, _ = load_data(os.path.join(DATA_ROOT, stem + '.mat'))
    dim, lam = BEST_CFG[stem]
    kernels = gauss_med_kernels(X[yy == 0])
    _, s = score_config(X, yy, kernels, dim, lam, device)
    return s


def _find_mat(alg, stem):
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


def get_alg_scores(alg, stem, y):
    if alg == 'KFGOD':
        p = os.path.join(KFGOD_DIR, stem, f'{stem}_KFGOD.mat')
        if not os.path.exists(p):
            return None
        s = np.asarray(sio.loadmat(p)['opt_out_scores'])[:, 0].ravel()
        return s if len(s) == len(y) else None
    if alg == 'LMKAD':
        p = os.path.join(LMKAD_PY, f'{stem}_LMKAD.mat')
        if not os.path.exists(p):
            return None
        s = np.asarray(sio.loadmat(p)['res_single'][0, 0]['opt_scores'], float).ravel()
        return s if len(s) == len(y) else None
    p = _find_mat(alg, stem)
    if p is None:
        return None
    s = np.asarray(sio.loadmat(p)['opt_out_scores'])[:, 0].ravel()
    return s if len(s) == len(y) else None


RESULTS = {}
for stem in DATASETS:
    _, y = load_xy(stem); RESULTS[stem] = {}
    for m in METHODS:
        try:
            sc = get_scatter_scores(stem, y) if m == 'scatter' else get_alg_scores(m, stem, y)
            if sc is None:
                continue
            fpr, tpr, _ = roc_curve(y, sc)
            RESULTS[stem][m] = (fpr, tpr, roc_auc_score(y, sc))
        except Exception as e:
            print('ERR', m, stem, ':', e)
    print(f'{stem:<36} ' + '  '.join(
        f'{m}={RESULTS[stem][m][2]:.3f}' for m in METHODS if m in RESULTS[stem]))

PALETTE = ['#FF8C00', '#00BFFF', '#BA55D3', '#20B2AA', '#6B8E23',
           '#6A5ACD', '#FF69B4', '#1E90FF', '#8B4513']
MARKERS = ['x', 'v', '^', '<', '>', 's', 'p', 'D', 'P']
STYLE = {a: dict(color=PALETTE[i], marker=MARKERS[i], lw=1.1)
         for i, a in enumerate(ALG_YEAR)}
STYLE['scatter'] = dict(color='#DC143C', marker='o', lw=1.5)

out_dir = os.path.join(RESULT_DIR, 'roc_compare')
os.makedirs(out_dir, exist_ok=True)
n = 0
for stem in DATASETS:
    res = RESULTS.get(stem, {})
    if not res:
        continue
    fig = plt.figure(figsize=(4, 3), dpi=150)
    plt.plot([0, 1], [0, 1], color='gray', lw=0.5, linestyle=(0, (8, 8)))
    for m in METHODS:
        if m not in res:
            continue
        fpr, tpr, auc_v = res[m]; st = STYLE[m]
        plt.plot(fpr, tpr, label=f'{m} ({auc_v:.3f})',
                 color=st['color'], marker=st['marker'],
                 markevery=max(len(fpr) // 8, 1), markersize=3, lw=st['lw'])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 20, 40, 60, 80, 100], fontsize=7)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 20, 40, 60, 80, 100], fontsize=7)
    plt.xlim(-0.05, 1.02); plt.ylim(-0.05, 1.02); plt.grid(True)
    plt.title(stem[:30], fontsize=9)
    plt.legend(prop={'size': 5}, ncol=2, loc='lower right')
    fig.patch.set_facecolor('white')
    plt.savefig(os.path.join(out_dir, stem + '_ROC.pdf'),
                bbox_inches='tight', pad_inches=0.02)
    plt.close()
    n += 1
print(f'saved {n} ROC pdfs to {out_dir}')
