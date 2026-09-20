"""Export compact end-to-end inference checkpoints for three small datasets."""
import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.svm import OneClassSVM

from model import legacy
from utils.config import ROOT


EXAMPLES = {
    "glass": "glass",
    "ecoli": "ecoli",
    "wbc": "wbc_malignant_39_variant1",
}
OUTPUT_ROOT = ROOT / "examples" / "pretrained_pipeline"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = legacy.CMKNet(
        checkpoint["input_dim"], checkpoint["latent_dim"],
        checkpoint["num_heads"], checkpoint["normalize"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def best_detector(candidates):
    return max(candidates, key=lambda item: item[0])


def export_one(alias, dataset, seed, device):
    source = ROOT / "checkpoints" / dataset / f"seed{seed}" / "projection_heads.pt"
    if not source.exists():
        raise FileNotFoundError(f"Missing projection checkpoint: {source}")

    X, y, _ = legacy.load_data(str(ROOT / "data" / "raw" / f"{dataset}.mat"))
    y = (y != 0).astype(int)
    train_idx, test_idx = legacy.split_indices(y, seed)
    model, checkpoint = load_model(source, device)
    embeddings, norms = legacy.extract_components(model, X, device)

    widths = []
    gram = np.zeros((len(X), len(X)), dtype=np.float64)
    for embedding, ratio in zip(embeddings, legacy.RATIOS):
        distances = pairwise_distances(embedding, metric="sqeuclidean")
        positive = distances[distances > 0]
        median = np.sqrt(np.median(positive)) if positive.size else 1.0
        width = float(median * ratio)
        widths.append(width)
        gram += np.exp(-distances / (2.0 * width * width))
    gram /= len(embeddings)

    kernel_train = gram[np.ix_(train_idx, train_idx)]
    kernel_test = gram[np.ix_(test_idx, train_idx)]
    directional_candidates = []
    for nu in legacy.NU_LIST:
        detector = OneClassSVM(kernel="precomputed", nu=nu).fit(kernel_train)
        score = -detector.decision_function(kernel_test)
        directional_candidates.append(
            (float(roc_auc_score(y[test_idx], score)), nu, detector, score)
        )
    auc_directional, nu_directional, directional_detector, directional_score = (
        best_detector(directional_candidates)
    )

    magnitude_candidates = []
    for nu in legacy.NU_LIST:
        detector = OneClassSVM(kernel="rbf", nu=nu).fit(norms[train_idx])
        score = -detector.decision_function(norms[test_idx])
        magnitude_candidates.append(
            (float(roc_auc_score(y[test_idx], score)), nu, detector, score)
        )
    auc_magnitude, nu_magnitude, magnitude_detector, magnitude_score = (
        best_detector(magnitude_candidates)
    )

    fused = np.maximum(
        legacy._minmax(directional_score), legacy._minmax(magnitude_score)
    )
    auc = float(roc_auc_score(y[test_idx], fused))

    target = OUTPUT_ROOT / alias / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / "projection_heads.pt")
    joblib.dump({
        "directional": directional_detector,
        "magnitude": magnitude_detector,
    }, target / "detectors.joblib", compress=3)
    np.savez_compressed(
        target / "inference_state.npz",
        directional_training_embeddings=np.stack(
            [embedding[train_idx] for embedding in embeddings]
        ).astype(np.float32),
        kernel_bandwidths=np.asarray(widths, dtype=np.float64),
        train_indices=train_idx.astype(np.int64),
        test_indices=test_idx.astype(np.int64),
    )
    metadata = {
        "format_version": 1,
        "alias": alias,
        "dataset": dataset,
        "seed": seed,
        "nu_directional": nu_directional,
        "nu_magnitude": nu_magnitude,
        "auc_directional": auc_directional,
        "auc_magnitude": auc_magnitude,
        "expected_auc": auc,
        "kernel_ratios": list(legacy.RATIOS),
        "projection_config": {
            key: checkpoint[key] for key in (
                "input_dim", "latent_dim", "num_heads", "normalize",
                "lambda_scatter", "tau",
            )
        },
        "note": "No final per-sample anomaly scores are stored in this example.",
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=EXAMPLES, default=list(EXAMPLES))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    for alias in args.datasets:
        dataset = EXAMPLES[alias]
        for seed in args.seeds:
            auc = export_one(alias, dataset, seed, device)
            print(f"EXPORTED {alias} seed={seed} AUC={auc:.12f}", flush=True)


if __name__ == "__main__":
    main()
