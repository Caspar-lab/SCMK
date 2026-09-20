from pathlib import Path
from types import SimpleNamespace
import json
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.metrics import roc_auc_score

from model import bounded, legacy, scmk
from .data import load_fixed_split


def environment():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def resolve_device(name):
    return scmk.default_device() if name == "auto" else torch.device(name)


def _save_projection_checkpoint(model, path, dataset, seed, mode, input_dim,
                                dim, lam, tau):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "dataset": dataset,
        "split_seed": int(seed),
        "mode": mode,
        "input_dim": int(input_dim),
        "latent_dim": int(dim),
        "num_heads": len(model.projectors),
        "normalize": True,
        "lambda_scatter": float(lam),
        "tau": float(tau),
        "state_dict": model.state_dict(),
        "environment": environment(),
    }, path)


def compute_scores(root: Path, dataset: str, seed: int, config: dict,
                   defaults: dict, device, checkpoint_path=None):
    params = config["seeds"][str(seed)]
    dim = int(params["dim"])
    lam = float(params["lambda"])
    tau = float(params["tau"])
    mode = config["mode"]

    if mode == "legacy":
        X, y, _ = legacy.load_data(str(root / "data" / "raw" / f"{dataset}.mat"))
        y = (y != 0).astype(int)
        train_idx, test_idx = legacy.split_indices(y, seed)
        raw_bw = [k[2]["t"] for k in legacy.gauss_med_kernels(X[train_idx])]
        model, _ = legacy.train_variant(
            X[train_idx], legacy.RATIOS, raw_bw, dim, device, lam, tau,
            "embedding", legacy.TRAIN_CFG, False,
        )
        embeddings, norms = legacy.extract_components(model, X, device)
        _save_projection_checkpoint(
            model, checkpoint_path, dataset, seed, mode, X.shape[1], dim, lam, tau
        )
        _, directional = legacy._precom_mk_scores(
            embeddings, legacy.RATIOS, train_idx, test_idx, y[test_idx], legacy.NU_LIST
        )
        _, magnitude = legacy._ocsvm_best(
            norms[test_idx], y[test_idx], norms[train_idx], "rbf", legacy.NU_LIST
        )
        if directional is None or magnitude is None:
            raise RuntimeError("OC-SVM branch fitting failed")
        scores = np.maximum(legacy._minmax(directional), legacy._minmax(magnitude))
        original_idx = test_idx
    else:
        X, y, train_idx, test_idx, original_idx = load_fixed_split(
            root / "data", dataset, seed
        )
        if mode == "bounded":
            bounded_cfg = defaults["bounded"]
            args = SimpleNamespace(
                max_train=int(bounded_cfg["max_train"]),
                train_batch_size=int(bounded_cfg["train_batch_size"]),
                score_block_size=int(bounded_cfg["score_block_size"]),
                bandwidth_sample=int(bounded_cfg["bandwidth_sample"]),
            )
            scores = bounded.run_one(
                X, y, train_idx, test_idx, dim, lam, tau, seed, device, args,
                checkpoint_path=checkpoint_path, dataset=dataset,
            )[0]
        elif mode == "split":
            result = scmk.run_scmk(
                X, y, train_idx, test_idx, dim, lam, tau, device=device,
                return_model=checkpoint_path is not None,
            )
            scores = result["score"]
            if checkpoint_path is not None:
                _save_projection_checkpoint(
                    result["model"], checkpoint_path, dataset, seed, mode,
                    X.shape[1], dim, lam, tau,
                )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    return pd.DataFrame({
        "sample_index": np.asarray(original_idx, dtype=int),
        "label": y[test_idx].astype(int),
        "anomaly_score": np.asarray(scores, dtype=float),
    })


def save_run(frame, target: Path, dataset: str, seed: int, config: dict,
             reference_file: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    auc = float(roc_auc_score(frame["label"], frame["anomaly_score"]))
    reference = pd.read_csv(reference_file)
    reference_auc = float(roc_auc_score(reference["label"], reference["anomaly_score"]))
    metadata = {
        "dataset": dataset,
        "seed": seed,
        "config": config,
        "auc": auc,
        "reference_auc": reference_auc,
        "auc_difference": auc - reference_auc,
        "environment": environment(),
    }
    target.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return auc, reference_auc
