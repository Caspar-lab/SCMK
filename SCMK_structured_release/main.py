"""Run SCMK for one dataset using the manuscript configuration."""
import argparse
from pathlib import Path

from utils.config import ROOT, load_defaults, load_experiment_configs
from utils.runner import compute_scores, resolve_device, save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    defaults = load_defaults()
    configs = load_experiment_configs()
    if args.dataset not in configs:
        parser.error(f"Unknown dataset: {args.dataset}. Available: {', '.join(configs)}")
    device = resolve_device(args.device or defaults["runtime"]["device"])
    output_root = (args.output or ROOT / defaults["paths"]["output_dir"]).resolve()
    target = output_root / args.dataset / f"seed{args.seed}" / "SCMK.csv"

    frame = compute_scores(
        ROOT, args.dataset, args.seed, configs[args.dataset], defaults, device
    )
    reference = ROOT / "reference_scores" / args.dataset / f"seed{args.seed}" / "SCMK.csv"
    auc, reference_auc = save_run(
        frame, target, args.dataset, args.seed, configs[args.dataset], reference
    )
    print(f"saved: {target}")
    print(f"AUC={auc:.12f} reference={reference_auc:.12f} delta={auc-reference_auc:+.3g}")


if __name__ == "__main__":
    main()
