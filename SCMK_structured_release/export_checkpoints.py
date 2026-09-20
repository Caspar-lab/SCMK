"""Retrain final manuscript configurations and export projection checkpoints."""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils.config import ROOT, load_defaults, load_experiment_configs
from utils.runner import compute_scores, resolve_device, save_run


def write_summary(configs, checkpoint_root):
    rows = []
    for dataset in configs:
        values = {}
        for seed in (0, 1, 2):
            path = checkpoint_root / dataset / f"seed{seed}" / "scores.csv"
            if path.exists():
                frame = pd.read_csv(path)
                values[seed] = float(roc_auc_score(frame["label"], frame["anomaly_score"]))
        rows.append({
            "dataset": dataset,
            "completed_seeds": len(values),
            "seed0": values.get(0, np.nan),
            "seed1": values.get(1, np.nan),
            "seed2": values.get(2, np.nan),
            "mean_auc": float(np.mean(list(values.values()))) if values else np.nan,
            "std_auc": float(np.std(list(values.values()), ddof=1)) if len(values) == 3 else np.nan,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(checkpoint_root / "summary.csv", index=False)
    complete = summary[summary["completed_seeds"] == 3]
    (checkpoint_root / "summary.json").write_text(json.dumps({
        "completed_datasets": int(len(complete)),
        "total_datasets": int(len(summary)),
        "mean_auc_over_completed_datasets": (
            float(complete["mean_auc"].mean()) if len(complete) else None
        ),
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", help="dataset names; default: all")
    parser.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    defaults = load_defaults()
    configs = load_experiment_configs()
    datasets = args.datasets or list(configs)
    unknown = [name for name in datasets if name not in configs]
    if unknown:
        parser.error(f"Unknown datasets: {', '.join(unknown)}")
    checkpoint_root = ROOT / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device or defaults["runtime"]["device"])

    for dataset in datasets:
        for seed in args.seeds:
            run_dir = checkpoint_root / dataset / f"seed{seed}"
            weights = run_dir / "projection_heads.pt"
            scores = run_dir / "scores.csv"
            if weights.exists() and scores.exists() and not args.force:
                print(f"SKIP existing {dataset} seed={seed}", flush=True)
                continue
            params = configs[dataset]["seeds"][str(seed)]
            print(f"RUN {dataset} seed={seed} params={params}", flush=True)
            frame = compute_scores(
                ROOT, dataset, seed, configs[dataset], defaults, device,
                checkpoint_path=weights,
            )
            reference = ROOT / "reference_scores" / dataset / f"seed{seed}" / "SCMK.csv"
            auc, reference_auc = save_run(
                frame, scores, dataset, seed, configs[dataset], reference
            )
            print(
                f"  AUC={auc:.12f} archived={reference_auc:.12f} "
                f"delta={auc-reference_auc:+.6f}", flush=True
            )
            write_summary(configs, checkpoint_root)

    write_summary(configs, checkpoint_root)
    print(f"Checkpoint export complete: {checkpoint_root}", flush=True)


if __name__ == "__main__":
    main()
