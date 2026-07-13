"""
run_scatter_unsup.py — scatter(max_ensemble) 纯无监督版
======================================================
与半监督版(run_hybrid_score)的区别:
  · 带宽:gauss_med_kernels(X) 用【全量】数据估中位距离(不知正常子集)
  · 训练:CMK 在【全量】X 上对比训练(传全0标签使 X[y==0]=全量)
  · 评分:OC-SVM 在【全量】嵌入上拟合(H_normal=H_all),非仅正常样本
  · AUC :仍用真实标签评估;dim/lambda/nu 网格取最优(oracle,与对比算法同口径)

输出(result/scatter_unsup/):
  scatter_unsup_best.csv : dataset,N,D,anomaly,best_dim,best_lambda,best_auc,nu_dir,nu_mag,elapsed_s
  scatter_unsup_all.csv  : dataset,latent_dim,lambda_scatter,auc,elapsed_s
"""
import os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hybrid_score import (load_data, gauss_med_kernels, extract_components,
                              _best_ocsvm_scores, _minmax, NU_CANDIDATES,
                              LATENT_DIMS, LAMBDAS)
from CMK_OCSVM import TRAIN_CFG
from CMK_OCSVM_scatter import train_cmk_scatter

DATA_DIR = r'C:\OD\Shihao\datasets'
OUTDIR   = r'C:\OD\Shihao\5\result\scatter_unsup'
os.makedirs(OUTDIR, exist_ok=True)
BEST_CSV = os.path.join(OUTDIR, 'scatter_unsup_best.csv')
ALL_CSV  = os.path.join(OUTDIR, 'scatter_unsup_all.csv')

MAX_N = 8000                       # 跳过超大数据集(全量 RBF-OCSVM O(N^2~3) 过慢)
FOCUS = sys.argv[1:]               # 命令行传数据集名则只跑这些;否则跑全部


def score_unsup(X, y, kernels, dim, lam, dev):
    """无监督:全量训练 + 全量拟合 OC-SVM；AUC 用真实 y。返回 (auc, nu_dir, nu_mag)。"""
    yz = np.zeros(len(X), dtype=int)                      # 全0 -> 训练用全量
    model = train_cmk_scatter(X, yz, kernels, dim, dev,
                              {**TRAIN_CFG, 'lambda_scatter': lam})
    Hn, Hnorms = extract_components(model, X, dev)
    Hdir = np.concatenate(Hn, axis=1)
    # H_normal 传【全量】(无监督),nu 网格取使真实 AUC 最优的分数
    ad, nud, sd = _best_ocsvm_scores(Hdir,   y, Hdir,   'linear', NU_CANDIDATES)
    am, num, sm = _best_ocsvm_scores(Hnorms, y, Hnorms, 'rbf',    NU_CANDIDATES)
    if sd is None and sm is None:
        return float('nan'), None, None
    if sd is None:
        return am, None, num
    if sm is None:
        return ad, nud, None
    auc = roc_auc_score(y, np.maximum(_minmax(sd), _minmax(sm)))
    return auc, nud, num


def run_one(path, dev):
    stem = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    kernels = gauss_med_kernels(X)                        # 全量估带宽
    best = (-1.0, None, None, None, None)                 # auc,dim,lam,nud,num
    rows = []
    for dim in LATENT_DIMS:
        for lam in LAMBDAS:
            t0 = time.time()
            auc, nud, num = score_unsup(X, y, kernels, dim, lam, dev)
            dt = time.time() - t0
            rows.append(dict(dataset=stem, latent_dim=dim, lambda_scatter=lam,
                             auc=round(auc, 6) if not np.isnan(auc) else np.nan,
                             elapsed_s=round(dt, 1)))
            if not np.isnan(auc) and auc > best[0]:
                best = (auc, dim, lam, nud, num)
    brow = dict(dataset=stem, N=meta['N'], D=X.shape[1],
                anomaly=round(meta['anomaly_rate'], 4),
                best_dim=best[1], best_lambda=best[2],
                best_auc=round(best[0], 6), nu_dir=best[3], nu_mag=best[4],
                elapsed_s=round(sum(r['elapsed_s'] for r in rows), 1))
    print(f'  -> {stem:<34} best_AUC={best[0]:.4f}  dim={best[1]} lam={best[2]}')
    return rows, brow


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.endswith('.mat') and 'datalist' not in f.lower()
                   and not f.startswith('Ori_'))
    print(f'device={dev}  候选 .mat={len(files)}  MAX_N={MAX_N}')

    all_rows, best_rows = [], []
    for fn in files:
        stem = os.path.splitext(fn)[0]
        if FOCUS and stem not in FOCUS:
            continue
        path = os.path.join(DATA_DIR, fn)
        try:
            X, y, meta = load_data(path)
        except Exception as e:
            print(f'  [skip] {stem}: load 失败 {e}'); continue
        if meta['N'] > MAX_N:
            print(f'  [skip] {stem}: N={meta["N"]} > {MAX_N}'); continue
        if y.sum() == 0 or y.sum() == len(y):
            print(f'  [skip] {stem}: 单一类别'); continue
        rows, brow = run_one(path, dev)
        all_rows.extend(rows); best_rows.append(brow)
        pd.DataFrame(all_rows).to_csv(ALL_CSV, index=False)
        pd.DataFrame(best_rows).to_csv(BEST_CSV, index=False)

    df = pd.DataFrame(best_rows)
    print(f'\n完成 {len(df)} 个数据集  平均 best_AUC={df["best_auc"].mean():.4f}')
    print(f'明细:{ALL_CSV}\n汇总:{BEST_CSV}')
