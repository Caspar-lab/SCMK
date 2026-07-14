"""
time_kfgod.py — KFGOD (核化模糊-粗糙, 粒球) 运行耗时（秒），seed=2 划分。
================================================================================
KFGOD 直推式(无监督)：粒球生成 get_GB + 在球心上核化模糊-粗糙打分 KFGOD(centers, delta)，
再映回样本。无独立训练/推理阶段 → train_s=0，infer_s = 整体计算耗时（含 get_GB）。
在与其它方法相同的 train+test 合并矩阵（= 全体样本）上计时。delta 取代表值（时间由 n 主导）。
复用 KFGOD 官方 python 实现。

手动运行：
  C:/anaconda3/envs/torch311/python.exe time_kfgod.py
输出：results/kfgod_timing.csv  (dataset, method, n_train, n_test, D, train_s, infer_s)
"""
import os, sys, time, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler

KF = r'C:/OD/Shihao/KFGOD-main/code'
sys.path.insert(0, KF)
from KFGOD import KFGOD
from GB_generation_with_idx import get_GB

ROOT = r'C:/OD/Shihao/5'
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results'); os.makedirs(OUTDIR, exist_ok=True)
SPLIT = r'C:/OD/Shihao/split_datasets/datasets_split_seed2'
DELTA = 0.30           # 代表性 delta（原扫描 range(2,80,2)/100，时间与 delta 基本无关）
REPEAT = 3

sel = pd.read_csv(ROOT + '/result/hybrid_score_semi/selection_scatter_semi_seed2.csv')
DATASETS = list(sel['dataset'])


def load_combined(name):
    tr = np.asarray(sio.loadmat(os.path.join(SPLIT, 'train', name + '.mat'))['trandata'], float)
    te = np.asarray(sio.loadmat(os.path.join(SPLIT, 'test', name + '.mat'))['trandata'], float)
    X = np.vstack([tr[:, :-1], te[:, :-1]])
    return X, tr.shape[0], te.shape[0]


def kfgod_full(X):
    """完整 KFGOD 打分流程（get_GB + 球心打分 + 映射），返回样本级异常分。"""
    X = X.copy()
    ID = (X >= 1).all(axis=0) & (X.max(axis=0) != X.min(axis=0))
    if ID.any():
        X[:, ID] = MinMaxScaler().fit_transform(X[:, ID])
    n, m = X.shape
    GBs = get_GB(X)
    centers = np.zeros((len(GBs), m))
    for idx, gb in enumerate(GBs):
        centers[idx] = np.mean(gb[:, :-1], axis=0)
    OD_gb = KFGOD(centers, DELTA)
    OD = np.zeros(n)
    for idx, gb in enumerate(GBs):
        OD[gb[:, -1].astype(int)] = OD_gb[idx]
    return OD


def run(name):
    X, n_tr, n_te = load_combined(name)
    _ = kfgod_full(X)                      # warm-up
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter(); _ = kfgod_full(X); times.append(time.perf_counter() - t0)
    return dict(dataset=name, method='KFGOD', n_train=n_tr, n_test=n_te, D=X.shape[1],
                train_s=0.0, infer_s=round(float(np.median(times)), 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default=None); args = ap.parse_args()
    names = [n for n in DATASETS if not args.only or n in set(args.only.split(','))]
    print(f'KFGOD timing  datasets={len(names)}  delta={DELTA}', flush=True)
    rows, out = [], os.path.join(OUTDIR, 'kfgod_timing.csv')
    for i, nm in enumerate(names, 1):
        try:
            r = run(nm); rows.append(r)
            print(f"[{i}/{len(names)}] {nm:<34} infer={r['infer_s']:.3f}s", flush=True)
        except Exception as e:
            print(f"[{i}/{len(names)}] {nm} ERROR {e}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, flush=True)
