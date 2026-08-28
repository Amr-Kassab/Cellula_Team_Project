"""Read-only loading and reproducibility helpers for modeling experiments."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "processed_data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def set_global_seed(seed: int = SEED) -> None:
    """Set seeds for reproducible Python, NumPy, and PyTorch experiments."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_result_dirs() -> None:
    for name in ("metrics", "figures", "models", "predictions"):
        (RESULTS_DIR / name).mkdir(parents=True, exist_ok=True)


def load_processed_data(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Load untouched preprocessing outputs and validate their expected schema."""
    filename = "X_clean_mubeta.npy" if kind == "mubeta" else "X_clean.npy"
    x_path, y_path = DATA_DIR / filename, DATA_DIR / "y_clean.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Expected processed files at {x_path} and {y_path}")
    X = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.int64, copy=False)
    if X.ndim != 3 or X.shape[1:] != (4, 751):
        raise ValueError(f"Expected X shape (n_trials, 4, 751); got {X.shape}")
    if y.ndim != 1 or len(y) != len(X) or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Labels must be binary 0/1 and match trial count.")
    return X, y

