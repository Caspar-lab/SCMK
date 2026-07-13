"""
time_dfno.py — DFNO 运行耗时（秒），seed=2 划分，每数据集单一 best_k。
================================================================================
DFNO 是直推式(transductive)模糊-粗糙方法：在 train+test 合并矩阵上一次性 O(n^2)
计算，无独立的训练/推理阶段。故 train_s 记为 0（无训练），infer_s = 整体计算耗时。
复用 DFNO 官方 repo 的向量化实现（已数值对齐原作）。

手动运行：
  C:/anaconda3/envs/torch311/python.exe time_dfno.py
输出：results/dfno_timing.csv  (dataset, method, n_train, n_test, D, train_s, infer_s)
"""
import os, sys, time, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd

DFNO_DIR = r'C:/OD/Shihao/DFNO-main/DFNO-main/Code'
sys.path.insert(0, DFNO_DIR)
from run_dfno_split import load_combined, preprocess, dfno_scores_over_k

ROOT = r'C:/OD/Shihao/5'
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results'); os.makedirs(OUTDIR, exist_ok=True)
SPLIT_ROOT = r'C:/OD/Shihao/split_datasets/datasets_split_seed2'
REPEAT = 3

sel = pd.read_csv(ROOT + '/result/hybrid_score_semi/selection_scatter_semi_seed2.csv')
DATASETS = list(sel['dataset'])
dsum = pd.read_csv(DFNO_DIR + '/results_split/DFNO_summary.csv')
BEST_K = dict(zip(dsum['dataset'], dsum['best_k'].astype(int)))


def run(name):
    X, test_mask, y, orig_idx, n_train = load_combined(name, SPLIT_ROOT)
    X = preprocess(X)
    k = int(BEST_K.get(name, 20))
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        _ = dfno_scores_over_k(X, [k])       # transductive 一次性计算
        times.append(time.perf_counter() - t0)
    return dict(dataset=name, method='DFNO', n_train=n_train,
                n_test=int(test_mask.sum()), D=X.shape[1],
                train_s=0.0, infer_s=round(float(np.median(times)), 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default=None); args = ap.parse_args()
    names = [n for n in DATASETS if not args.only or n in set(args.only.split(','))]
    print(f'DFNO timing  datasets={len(names)}  repeat={REPEAT}', flush=True)
    rows, out = [], os.path.join(OUTDIR, 'dfno_timing.csv')
    for i, nm in enumerate(names, 1):
        try:
            r = run(nm); rows.append(r)
            print(f"[{i}/{len(names)}] {nm:<34} infer={r['infer_s']:.3f}s (k={BEST_K.get(nm)})", flush=True)
        except Exception as e:
            print(f"[{i}/{len(names)}] {nm} ERROR {e}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, flush=True)
