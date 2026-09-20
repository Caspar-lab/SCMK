from pathlib import Path
import numpy as np
import scipy.io as sio

from model import scmk


def load_fixed_split(data_root: Path, dataset: str, seed: int):
    """Load a stored train/test split while retaining original test indices."""
    split_root = data_root / "splits" / f"seed{seed}"
    train_file = sio.loadmat(split_root / "train" / f"{dataset}.mat")
    test_file = sio.loadmat(split_root / "test" / f"{dataset}.mat")
    train = train_file["trandata"].astype(np.float64)
    test = test_file["trandata"].astype(np.float64)
    data = np.vstack([train, test])
    X = scmk._preprocess(data[:, :-1])
    y = (data[:, -1] != 0).astype(int)
    train_idx = np.arange(len(train))
    test_idx = np.arange(len(train), len(data))
    original_idx = test_file["orig_idx"].ravel().astype(int)
    return X, y, train_idx, test_idx, original_idx
