from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc_from_file(path: Path):
    frame = pd.read_csv(path)
    return float(roc_auc_score(frame["label"], frame["anomaly_score"]))


def summarize_reference_scores(root: Path, configs: dict):
    records = []
    for dataset, config in configs.items():
        values = [
            auc_from_file(root / "reference_scores" / dataset / f"seed{seed}" / "SCMK.csv")
            for seed in (0, 1, 2)
        ]
        records.append({
            "dataset": dataset,
            "seed0": values[0],
            "seed1": values[1],
            "seed2": values[2],
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "table_mean": float(config["table_mean"]),
            "table_std": float(config["table_std"]),
        })
    return pd.DataFrame(records)
