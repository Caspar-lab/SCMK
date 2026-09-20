"""Check the archived score outputs paired with projection checkpoints."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils.config import ROOT, load_experiment_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", help="dataset names; default: all")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    configs = load_experiment_configs()
    datasets = args.datasets or list(configs)
    rows = []
    for dataset in datasets:
        aucs = []
        for seed in (0, 1, 2):
            run_dir = args.checkpoint_root / dataset / f"seed{seed}"
            score_file = run_dir / "scores.csv"
            weight_file = run_dir / "projection_heads.pt"
            if not score_file.exists() or not weight_file.exists():
                raise FileNotFoundError(f"Incomplete checkpoint run: {run_dir}")
            frame = pd.read_csv(score_file)
            aucs.append(float(roc_auc_score(frame["label"], frame["anomaly_score"])))
        rows.append({
            "dataset": dataset,
            "seed0": aucs[0], "seed1": aucs[1], "seed2": aucs[2],
            "mean": float(np.mean(aucs)), "std": float(np.std(aucs, ddof=1)),
        })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nMean AUC across datasets: {result['mean'].mean():.12f}")


if __name__ == "__main__":
    main()
