# Aggregate ICL / NeuTraLAD over seeds {0,1,2} -> per-dataset mean±std, and compare
# against the existing algorithms on the 20 manuscript datasets.
#
# Inputs : results_split/<ALGO>_summary_seed{0,1,2}.csv  (best_auc per dataset per seed)
# Outputs: results_split/ICL_NeuTraL_meanstd.csv          (all under-10k datasets)
#          results_split/compare_manuscript20.csv         (20 datasets, vs scatter/DFNO)
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_split")
ROOT = r"C:/OD/Shihao/5"
SEEDS = [0, 1, 2]
ALGOS = ["ICL", "NeuTraL"]


def load_seed_aucs(algo):
    """dataset -> {seed: best_auc} from the per-seed summary CSVs."""
    d = {}
    for s in SEEDS:
        p = os.path.join(RES, f"{algo}_summary_seed{s}.csv")
        df = pd.read_csv(p)
        df = df[df["status"] == "ok"]
        for _, r in df.iterrows():
            d.setdefault(r["dataset"], {})[s] = float(r["best_auc"])
    return d


def meanstd_table():
    rows = {}
    algo_data = {a: load_seed_aucs(a) for a in ALGOS}
    datasets = sorted(set().union(*[set(algo_data[a]) for a in ALGOS]))
    out = []
    for ds in datasets:
        row = {"dataset": ds}
        for a in ALGOS:
            vals = [algo_data[a][ds][s] for s in SEEDS if s in algo_data[a].get(ds, {})]
            if len(vals) == len(SEEDS):
                arr = np.array(vals, float)
                for s, v in zip(SEEDS, vals):
                    row[f"{a}_s{s}"] = round(v, 4)
                row[f"{a}_mean"] = round(arr.mean(), 4)
                row[f"{a}_std"] = round(arr.std(ddof=1), 4)
        out.append(row)
    return pd.DataFrame(out)


def main():
    df = meanstd_table()
    df.to_csv(os.path.join(RES, "ICL_NeuTraL_meanstd.csv"), index=False)
    print(f"all under-10k datasets: {len(df)}")
    for a in ALGOS:
        if f"{a}_mean" in df:
            print(f"  {a:<8} mean over datasets = {df[f'{a}_mean'].mean():.4f} "
                  f"(avg std {df[f'{a}_std'].mean():.4f})")

    # focused table on the 20 manuscript datasets, next to scatter (seed012) + DFNO
    sel = pd.read_csv(ROOT + "/result/hybrid_score_semi/selection_scatter_semi_seed2.csv")
    ds20 = list(sel["dataset"])
    scat = pd.read_csv(ROOT + "/result/hybrid_score_semi/scatter_seed012_meanstd.csv")  # name,mean,std
    namemap = dict(zip(sel["dataset"], range(len(sel))))  # keep order
    sub = df[df["dataset"].isin(ds20)].copy()
    sub = sub.merge(sel[["dataset", "scatter_semi_seed2", "DFNO"]], on="dataset", how="right")
    sub.to_csv(os.path.join(RES, "compare_manuscript20.csv"), index=False)
    cols = ["dataset"] + [c for c in ["ICL_mean", "ICL_std", "NeuTraL_mean", "NeuTraL_std",
                                      "scatter_semi_seed2", "DFNO"] if c in sub.columns]
    print("\n20 manuscript datasets:")
    print(sub[cols].to_string(index=False))
    miss = [d for d in ds20 if d not in set(df["dataset"])]
    if miss:
        print("\nNOT in ICL/NeuTraL results (check size cap / errors):", miss)


if __name__ == "__main__":
    main()
