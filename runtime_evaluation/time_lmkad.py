"""
time_lmkad.py — LMKAD (多尺度高斯核, 单类) 训练/推理耗时（秒），seed=2 划分。
================================================================================
复用 LMKAD 官方 python 端口（experiment_gauss + experiment）。单一 best_C（取自
LMKAD_gauss_seed2_summary.csv）：train=create_model（学核权重+边界），infer=test_model。

手动运行：
  C:/anaconda3/envs/torch311/python.exe time_lmkad.py
输出：results/lmkad_timing.csv  (dataset, method, n_train, n_test, D, train_s, infer_s)
"""
import os, sys, time, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import scipy.io as sio

LMK = r'C:/OD/Shihao/LMKAD-master/python'
sys.path.insert(0, LMK)
from experiment_gauss import load_presplit, compute_med, gauss_kernels
import experiment as _E
from experiment import set_backend
try:
    set_backend('gpu')
except Exception:
    try: set_backend('cpu')
    except Exception: pass

ROOT = r'C:/OD/Shihao/5'
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results'); os.makedirs(OUTDIR, exist_ok=True)
SPLIT = r'C:/OD/Shihao/split_datasets/datasets_split_seed2'
ER = r'C:/OD/Shihao/Experimental_results'
REPEAT = 3

sel = pd.read_csv(ROOT + '/result/hybrid_score_semi/selection_scatter_semi_seed2.csv')
DATASETS = list(sel['dataset'])
LMK_MAT = ER + '/LMKAD_gauss_split_seed2'


def best_C_for(name):
    """从每数据集 .mat 的 res_single.best_C 读取（summary CSV 不全）；缺失回退 1.0。"""
    p = os.path.join(LMK_MAT, name + '_LMKAD.mat')
    try:
        return float(np.asarray(sio.loadmat(p)['res_single'][0, 0]['best_C']).ravel()[0])
    except Exception:
        return 1.0


def run(name):
    Xtr, Xte, te_lbls, te_y01, oi = load_presplit(SPLIT, name)
    kernels = gauss_kernels(compute_med(Xtr))
    train_lbls = np.ones(Xtr.shape[0])
    C = best_C_for(name)
    # warm-up (backend / JIT)
    m1, m2 = _E.create_model(Xtr, train_lbls, kernels, C)
    _ = _E.test_model(Xte, te_lbls, m1, m2, kernels, C)
    # ---- train ----
    trs = []
    for _ in range(REPEAT):
        t0 = time.perf_counter(); m1, m2 = _E.create_model(Xtr, train_lbls, kernels, C)
        trs.append(time.perf_counter() - t0)
    # ---- infer ----
    ins = []
    for _ in range(REPEAT):
        t0 = time.perf_counter(); _E.test_model(Xte, te_lbls, m1, m2, kernels, C)
        ins.append(time.perf_counter() - t0)
    return dict(dataset=name, method='LMKAD', n_train=Xtr.shape[0], n_test=Xte.shape[0],
                D=Xtr.shape[1], train_s=round(float(np.median(trs)), 4),
                infer_s=round(float(np.median(ins)), 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default=None); args = ap.parse_args()
    names = [n for n in DATASETS if not args.only or n in set(args.only.split(','))]
    print(f'LMKAD timing  datasets={len(names)}', flush=True)
    rows, out = [], os.path.join(OUTDIR, 'lmkad_timing.csv')
    for i, nm in enumerate(names, 1):
        try:
            r = run(nm); rows.append(r)
            print(f"[{i}/{len(names)}] {nm:<34} train={r['train_s']:.3f}s infer={r['infer_s']:.3f}s (C={best_C_for(nm)})", flush=True)
        except Exception as e:
            print(f"[{i}/{len(names)}] {nm} ERROR {e}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, flush=True)
