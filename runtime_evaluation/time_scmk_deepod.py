"""
time_scmk_deepod.py — 统一计时：SCMK 与深度单类基线的 训练/推理 耗时（秒）
================================================================================
同一台机器、同一 seed=2 半监督划分、单一固定配置，分别测量：
  train_s : 在训练集(仅正常)上拟合模型
  infer_s : 对测试集打分（推理）

覆盖（本进程内直接调用，最可比）：
  SCMK      : K=5 多核（train_cmk_scatter + 提取训练嵌入 + 拟合两路 OC-SVM）
  SCMK-1K   : K=1 单核（同 d,λ，仅核数不同 → 直接量化 C(K,2) 亲和矩阵的开销）
  DeepSVDD  : deepod.models.dsvdd.DeepSVDD
  ICL       : deepod.models.icl.ICL
  NeuTraLAD : deepod.models.neutral.NeuTraL

计时口径：CUDA 预热一次后测量；train 计 1 次，infer 取 REPEAT 次中位数。
默认在论文的 20 个数据集上（--all 跑清单全部）。手动运行：
  C:/anaconda3/envs/torch311/python.exe time_scmk_deepod.py
输出：results/scmk_deepod_timing.csv  （dataset, method, n_train, n_test, D, train_s, infer_s）
"""
import os, sys, time, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch

HS = r'C:\OD\Shihao\5\Granular-CMK\hybrid_score'
sys.path.insert(0, HS)
from run_hybrid_score_semi import (split_indices, extract_components, _minmax,
                                   gauss_med_kernels, load_data, _BASE_CFG)
from CMK_OCSVM_scatter import train_cmk_scatter
from sklearn.svm import OneClassSVM
from deepod.models.dsvdd import DeepSVDD
from deepod.models.icl import ICL
from deepod.models.neutral import NeuTraL

ROOT = r'C:/OD/Shihao/5'
HERE = os.path.dirname(os.path.abspath(__file__))
DR = 'C:/OD/Shihao/datasets'
OUTDIR = os.path.join(HERE, 'results'); os.makedirs(OUTDIR, exist_ok=True)
SEED = 2
NU = 0.1               # OC-SVM 固定 nu（网络训练主导耗时，nu 搜索仅评分细节）
REPEAT = 5             # 推理重复次数取中位数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sel = pd.read_csv(ROOT + '/result/hybrid_score_semi/selection_scatter_semi_seed2.csv')
DATASETS = list(sel['dataset'])
_all = pd.read_csv(ROOT + '/result/hybrid_score_semi/hybrid_semi_all.csv')
_s2 = _all[_all['split_seed'] == SEED]
BEST_CFG = {}
for ds in DATASETS:
    sub = _s2[_s2['dataset'] == ds]
    br = sub.loc[sub['auc'].idxmax()]
    BEST_CFG[ds] = (int(br['latent_dim']), float(br['lambda_scatter']))


def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()


def time_scmk(Xtr, Xte, dim, lam, ratios):
    """返回 (train_s, infer_s)。train=网络+两路OC-SVM拟合；infer=测试嵌入+打分。"""
    ker = gauss_med_kernels(Xtr, ratios=ratios)
    cfg = {**_BASE_CFG, 'lambda_scatter': lam}
    # ---- train ----
    sync(); t0 = time.perf_counter()
    model = train_cmk_scatter(Xtr, np.zeros(len(Xtr), int), ker, dim, device, cfg)
    Hnp_tr, Hn_tr = extract_components(model, Xtr, device)
    Hd_tr = np.concatenate(Hnp_tr, axis=1)
    ocs_dir = OneClassSVM(kernel='linear', nu=NU).fit(Hd_tr)
    ocs_mag = OneClassSVM(kernel='rbf', nu=NU).fit(Hn_tr)
    sync(); train_s = time.perf_counter() - t0
    # ---- infer (median of REPEAT) ----
    times = []
    for _ in range(REPEAT):
        sync(); t1 = time.perf_counter()
        Hnp_te, Hn_te = extract_components(model, Xte, device)
        Hd_te = np.concatenate(Hnp_te, axis=1)
        sd = -ocs_dir.decision_function(Hd_te)
        sn = -ocs_mag.decision_function(Hn_te)
        _ = np.maximum(_minmax(sd), _minmax(sn))
        sync(); times.append(time.perf_counter() - t1)
    return train_s, float(np.median(times))


def time_deepod(cls, Xtr, Xte):
    clf = cls(device=str(device), random_state=42, verbose=0)
    sync(); t0 = time.perf_counter(); clf.fit(Xtr); sync(); train_s = time.perf_counter() - t0
    times = []
    for _ in range(REPEAT):
        sync(); t1 = time.perf_counter(); _ = clf.decision_function(Xte); sync()
        times.append(time.perf_counter() - t1)
    return train_s, float(np.median(times))


def run(name):
    X, y, _ = load_data(os.path.join(DR, name + '.mat'))
    tr, te = split_indices(y, SEED)
    Xtr, Xte = X[tr].astype(np.float64), X[te].astype(np.float64)
    dim, lam = BEST_CFG[name]
    rows = []
    tr5, in5 = time_scmk(Xtr, Xte, dim, lam, (0.1, 0.5, 1.0, 2.0, 5.0))
    rows.append(('SCMK', tr5, in5))
    # 单核无跨核对，λ=0 会导致零损失(无训练信号)；用 λ>0 保证有 scatter 损失可训练。
    # 计时与 λ 值无关，只与是否存在有效损失有关。
    lam1 = lam if lam > 0 else 100.0
    tr1, in1 = time_scmk(Xtr, Xte, dim, lam1, (1.0,))
    rows.append(('SCMK-1K', tr1, in1))
    for nm, cls in [('DeepSVDD', DeepSVDD), ('ICL', ICL), ('NeuTraLAD', NeuTraL)]:
        try:
            a, b = time_deepod(cls, Xtr, Xte); rows.append((nm, a, b))
        except Exception as e:
            print('  ERR', nm, name, e); rows.append((nm, np.nan, np.nan))
    return [dict(dataset=name, method=m, n_train=len(Xtr), n_test=len(Xte),
                 D=X.shape[1], train_s=round(a, 4), infer_s=round(b, 4)) for m, a, b in rows]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--all', action='store_true')
    ap.add_argument('--only', default=None); args = ap.parse_args()
    names = DATASETS
    if args.only: names = [n for n in names if n in set(args.only.split(','))]
    print(f'device={device}  datasets={len(names)}  seed={SEED}  repeat={REPEAT}', flush=True)
    # warm-up CUDA once
    if names:
        _w = torch.zeros(8, 8, device=device); del _w; sync()
    allrows, out = [], os.path.join(OUTDIR, 'scmk_deepod_timing.csv')
    for i, nm in enumerate(names, 1):
        t0 = time.time()
        try:
            rr = run(nm); allrows.extend(rr)
            s = '  '.join(f"{r['method']}:{r['train_s']:.2f}/{r['infer_s']:.3f}" for r in rr)
            print(f"[{i}/{len(names)}] {nm:<34} {s}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{i}/{len(names)}] {nm} ERROR {e}", flush=True)
        pd.DataFrame(allrows).to_csv(out, index=False)  # incremental
    print('saved', out, flush=True)
