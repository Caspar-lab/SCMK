"""Run end-to-end inference from a small-dataset checkpoint without training."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import pairwise_distances, roc_auc_score

from model import legacy
from utils.config import ROOT


EXAMPLES = {
    "glass": "glass",
    "ecoli": "ecoli",
    "wbc": "wbc_malignant_39_variant1",
}
EXAMPLE_ROOT = ROOT / "examples" / "pretrained_pipeline"


def load_projection(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = legacy.CMKNet(
        checkpoint["input_dim"], checkpoint["latent_dim"],
        checkpoint["num_heads"], checkpoint["normalize"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate(alias, seed, device):
    dataset = EXAMPLES[alias]
    run_dir = EXAMPLE_ROOT / alias / f"seed{seed}"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    state = np.load(run_dir / "inference_state.npz")
    detectors = joblib.load(run_dir / "detectors.joblib")
    X, y, _ = legacy.load_data(str(ROOT / "data" / "raw" / f"{dataset}.mat"))
    y = (y != 0).astype(int)
    test_idx = state["test_indices"]

    model = load_projection(run_dir / "projection_heads.pt", device)
    test_embeddings, test_norms = legacy.extract_components(model, X[test_idx], device)
    train_embeddings = state["directional_training_embeddings"]
    widths = state["kernel_bandwidths"]
    kernel_test = np.zeros((len(test_idx), train_embeddings.shape[1]), dtype=np.float64)
    for head, width in enumerate(widths):
        distances = pairwise_distances(
            test_embeddings[head], train_embeddings[head], metric="sqeuclidean"
        )
        kernel_test += np.exp(-distances / (2.0 * width * width))
    kernel_test /= len(widths)

    directional_score = -detectors["directional"].decision_function(kernel_test)
    magnitude_score = -detectors["magnitude"].decision_function(test_norms)
    anomaly_score = np.maximum(
        legacy._minmax(directional_score), legacy._minmax(magnitude_score)
    )
    auc = float(roc_auc_score(y[test_idx], anomaly_score))
    return auc, float(metadata["expected_auc"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=EXAMPLES, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    auc, expected = evaluate(args.dataset, args.seed, torch.device(args.device))
    print(f"dataset={args.dataset} seed={args.seed}")
    print(f"AUC={auc:.12f}")
    print(f"expected={expected:.12f}")
    print(f"absolute_difference={abs(auc-expected):.3g}")


if __name__ == "__main__":
    main()
