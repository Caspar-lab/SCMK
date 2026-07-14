"""
time_disentad.py — Disent-AD 训练/推理耗时（秒），seed=2 划分，每数据集 best_hidden_dim。
================================================================================
复用 Disent-AD 官方 repo 的 DisNet / DatasetBuilder。为隔离训练耗时，本脚本用
固定 hidden_dim 训练 EPOCHS 轮（不在每轮打分），单独测一次推理前向。

前置：需要 Disent-AD-main 在 sys.path（其包 data_preprocess / network 可导入）。
手动运行：
  cd C:/OD/Shihao/Disent-AD-main
  C:/anaconda3/envs/torch311/python.exe C:/OD/Shihao/5/runtime_evaluation/time_disentad.py
输出：results/disentad_timing.csv  (dataset, method, n_train, n_test, D, train_s, infer_s)
"""
import os, sys, time, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

DISENT_DIR = r'C:/OD/Shihao/Disent-AD-main'
sys.path.insert(0, DISENT_DIR)
from data_preprocess.load_dataset import DatasetBuilder
from network.dis_net import DisNet

ROOT = r'C:/OD/Shihao/5'
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results'); os.makedirs(OUTDIR, exist_ok=True)
SPLIT = r'C:/OD/Shihao/split_datasets/datasets_split_seed2'
ER = r'C:/OD/Shihao/Experimental_results'
SEED = 2
EPOCHS, PATCH_SIZE, OVERLAP, NORM, LR = 100, 1, 0, 'std', 1e-4
REPEAT = 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sel = pd.read_csv(ROOT + '/result/hybrid_score_semi/selection_scatter_semi_seed2.csv')
DATASETS = list(sel['dataset'])
dsum = pd.read_csv(ER + '/Disent_AD_split_seed2/Disent_AD_seed2_summary.csv')
BEST_HD = dict(zip(dsum['dataset'], dsum['best_hidden_dim'].astype(int)))


def batch_size_for(n):
    return min(256, max(16, 1 << int(np.floor(np.log2(max(n, 16))))))


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def sync():
    if device.type == 'cuda': torch.cuda.synchronize()


def load(name):
    tr = scipy.io.loadmat(os.path.join(SPLIT, 'train', name + '.mat'))['trandata']
    te = scipy.io.loadmat(os.path.join(SPLIT, 'test', name + '.mat'))['trandata']
    Xtr = torch.as_tensor(np.asarray(tr[:, :-1], np.float64), dtype=torch.float)
    Xte = torch.as_tensor(np.asarray(te[:, :-1], np.float64), dtype=torch.float)
    yte = (np.asarray(te[:, -1]) != 0).astype(np.float64)
    return Xtr, Xte, yte


def run(name):
    Xtr, Xte, yte = load(name); hd = int(BEST_HD.get(name, 256)); set_seed(SEED)
    bs = batch_size_for(Xtr.shape[0])
    tr_ds = DatasetBuilder(Xtr, patch_size=PATCH_SIZE, overlap=OVERLAP, norm=NORM)
    te_ds = DatasetBuilder(Xte, label=torch.as_tensor(yte.reshape(-1)),
                           patch_size=PATCH_SIZE, overlap=OVERLAP, norm=NORM)
    g = torch.Generator(); g.manual_seed(SEED)
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, generator=g)
    te_loader = DataLoader(te_ds, batch_size=bs, shuffle=False)
    model = DisNet(dim=hd, att_dim=PATCH_SIZE).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    # ---- train (EPOCHS, no per-epoch scoring) ----
    sync(); t0 = time.perf_counter()
    model.train()
    for _ in range(EPOCHS):
        for sample in tr_loader:
            rec, dis = model(sample['data'].to(device))
            loss = rec + dis
            opt.zero_grad(); loss.backward(); opt.step()
    sync(); train_s = time.perf_counter() - t0
    # ---- infer ----
    times = []
    for _ in range(REPEAT):
        sync(); t1 = time.perf_counter()
        model.eval()
        with torch.no_grad():
            for sample in te_loader:
                _ = model(sample['data'].to(device))
        sync(); times.append(time.perf_counter() - t1)
    return dict(dataset=name, method='Disent-AD', n_train=Xtr.shape[0],
                n_test=Xte.shape[0], D=Xtr.shape[1],
                train_s=round(train_s, 4), infer_s=round(float(np.median(times)), 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default=None); args = ap.parse_args()
    names = [n for n in DATASETS if not args.only or n in set(args.only.split(','))]
    print(f'device={device}  Disent-AD timing  datasets={len(names)}', flush=True)
    if names:
        _w = torch.zeros(8, 8, device=device); del _w; sync()
    rows, out = [], os.path.join(OUTDIR, 'disentad_timing.csv')
    for i, nm in enumerate(names, 1):
        try:
            r = run(nm); rows.append(r)
            print(f"[{i}/{len(names)}] {nm:<34} train={r['train_s']:.2f}s infer={r['infer_s']:.3f}s (hd={BEST_HD.get(nm)})", flush=True)
        except Exception as e:
            print(f"[{i}/{len(names)}] {nm} ERROR {e}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, flush=True)
