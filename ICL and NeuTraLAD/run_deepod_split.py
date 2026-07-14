# Batch AUC of DeepOD's ICL and NeuTraLAD on the seed-{0,1,2} semi-supervised splits.
#
# Protocol (deep one-class methods — fit/predict, no transductive combine):
#   * fit() on the train set (normal-only) of datasets_split_seed{seed};
#   * decision_function() on the test set (held-out 50% normals + all anomalies);
#   * AUC computed on the test rows. Matches scatter / Disent-AD semi-supervised setup.
#   * DeepOD default hyper-parameters; method random_state fixed to 42, so the only
#     source of variation across seeds is the data split (→ mean±std over seeds).
#   * Only datasets with N < 6000.
#
# Preprocessing matches the lab KFGOD/DFNO convention for cross-method parity:
#   MinMax-scale only numeric columns (all values >=1, non-constant), fit on the
#   combined train+test matrix; the model is still fit on train rows only.
#
# Outputs (under results_split/):
#   <ALGO>_summary_seed<seed>.csv   per dataset: best_auc, sizes, time, status
#   scores/seed<seed>/<ALGO>/<name>_scores.csv   orig_idx,label,anomaly_score
#   ICL_NeuTraL_meanstd.csv         per dataset mean±std over seeds (built at the end)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # before numpy/torch

import argparse
import time
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
import torch
from deepod.models.icl import ICL
from deepod.models.neutral import NeuTraL

ALGOS = {"ICL": ICL, "NeuTraL": NeuTraL}
SEEDS = [0, 1, 2]
MAX_SAMPLES = 6000   # 仅测试 N<6000 的数据集
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results_split")


def split_root(seed):
    return rf"C:/OD/Shihao/datasets_split_seed{seed}"


def preprocess_combined(X):
    """MinMax-scale numeric cols (all >=1, non-constant), fit on combined matrix."""
    X = X.copy()
    numeric = (X >= 1).all(axis=0) & (X.max(axis=0) != X.min(axis=0))
    if numeric.any():
        X[:, numeric] = MinMaxScaler().fit_transform(X[:, numeric])
    return X


def load_split(seed, name):
    r = split_root(seed)
    tr = np.asarray(loadmat(os.path.join(r, "train", name + ".mat"))["trandata"], dtype=np.float64)
    te_mat = loadmat(os.path.join(r, "test", name + ".mat"))
    te = np.asarray(te_mat["trandata"], dtype=np.float64)
    X = np.vstack([tr[:, :-1], te[:, :-1]])
    n_train = tr.shape[0]
    y_test = (te[:, -1] != 0).astype(int)
    orig_idx = np.asarray(te_mat["orig_idx"]).ravel().astype(int)
    return X, n_train, y_test, orig_idx


def run_one(algo_cls, seed, name):
    X, n_train, y_test, orig_idx = load_split(seed, name)
    if len(np.unique(y_test)) < 2:
        return {"dataset": name, "status": "skipped_one_class"}, None
    Xs = preprocess_combined(X)
    Xtr, Xte = Xs[:n_train], Xs[n_train:]
    t0 = time.time()
    clf = algo_cls(device=DEVICE, random_state=42, verbose=0)
    clf.fit(Xtr)                       # train on normal-only
    scores = clf.decision_function(Xte)  # higher = more anomalous
    elapsed = time.time() - t0
    auc = roc_auc_score(y_test, scores)
    summ = {"dataset": name, "status": "ok", "best_auc": round(float(auc), 6),
            "n_train": n_train, "n_test": len(Xte), "n_total": len(X),
            "n_features": X.shape[1], "time_seconds": round(elapsed, 2)}
    scores_df = pd.DataFrame({"orig_idx": orig_idx, "label": y_test,
                              "anomaly_score": np.asarray(scores, dtype=float)})
    return summ, scores_df


def list_datasets(seed, max_samples):
    man = pd.read_csv(os.path.join(split_root(seed), "split_manifest.csv"))
    names = man.loc[man["N"] < max_samples, "dataset"].tolist()
    have = {f[:-4] for f in os.listdir(os.path.join(split_root(seed), "train")) if f.endswith(".mat")}
    return sorted(n for n in names if n in have)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default="ICL,NeuTraL")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    algos = [a for a in args.algos.split(",") if a in ALGOS]
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"device={DEVICE}  algos={algos}  seeds={seeds}  max_N<{args.max_samples}", flush=True)

    for algo in algos:
        for seed in seeds:
            names = [args.dataset] if args.dataset else list_datasets(seed, args.max_samples)
            sc_dir = os.path.join(RESULTS_DIR, "scores", f"seed{seed}", algo)
            os.makedirs(sc_dir, exist_ok=True)
            summ_path = os.path.join(RESULTS_DIR, f"{algo}_summary_seed{seed}.csv")
            summaries = []
            print(f"\n=== {algo} | seed{seed} | {len(names)} datasets ===", flush=True)
            for i, name in enumerate(names, 1):
                try:
                    summ, scores_df = run_one(ALGOS[algo], seed, name)
                    if scores_df is not None:
                        scores_df.to_csv(os.path.join(sc_dir, f"{name}_scores.csv"), index=False)
                except Exception as exc:
                    summ = {"dataset": name, "status": f"error: {exc}"}
                summaries.append(summ)
                tag = (f"auc={summ.get('best_auc'):.4f} ({summ.get('time_seconds')}s)"
                       if summ.get("status") == "ok" else summ.get("status"))
                print(f"  [{i}/{len(names)}] {name:<36} {tag}", flush=True)
                pd.DataFrame(summaries).to_csv(summ_path, index=False)  # incremental
            ok = [s for s in summaries if s.get("status") == "ok"]
            if ok:
                print(f"  {algo} seed{seed}: mean AUC = {np.mean([s['best_auc'] for s in ok]):.4f}", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
