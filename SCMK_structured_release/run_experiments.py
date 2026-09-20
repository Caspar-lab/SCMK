"""Run SCMK on multiple manuscript datasets and random splits."""
import argparse
from pathlib import Path

from utils.config import ROOT, load_defaults, load_experiment_configs
from utils.runner import compute_scores, resolve_device, save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", help="dataset names; default: all")
    parser.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    defaults = load_defaults()
    configs = load_experiment_configs()
    datasets = args.datasets or list(configs)
    unknown = [name for name in datasets if name not in configs]
    if unknown:
        parser.error(f"Unknown datasets: {', '.join(unknown)}")
    device = resolve_device(args.device or defaults["runtime"]["device"])
    output_root = (args.output or ROOT / defaults["paths"]["output_dir"]).resolve()

    for dataset in datasets:
        for seed in args.seeds:
            target = output_root / dataset / f"seed{seed}" / "SCMK.csv"
            if target.exists() and not args.force:
                print(f"SKIP existing {dataset} seed={seed}")
                continue
            params = configs[dataset]["seeds"][str(seed)]
            print(f"RUN {dataset} seed={seed} mode={configs[dataset]['mode']} params={params}")
            frame = compute_scores(ROOT, dataset, seed, configs[dataset], defaults, device)
            reference = ROOT / "reference_scores" / dataset / f"seed{seed}" / "SCMK.csv"
            auc, reference_auc = save_run(
                frame, target, dataset, seed, configs[dataset], reference
            )
            print(f"  AUC={auc:.12f} reference={reference_auc:.12f} delta={auc-reference_auc:+.3g}")


if __name__ == "__main__":
    main()
