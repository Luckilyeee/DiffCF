import os
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import utils as legacy_utils
except Exception:
    legacy_utils = None


def _zscore_per_channel(x, eps=1e-8):
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    return (x - mean) / (std + eps), mean, std


def _apply_zscore(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)


def _read_ucr(root, name):
    train_path = os.path.join(root, name, name + "_TRAIN.tsv")
    test_path = os.path.join(root, name, name + "_TEST.tsv")
    train = np.loadtxt(train_path, delimiter="\t")
    test = np.loadtxt(test_path, delimiter="\t")
    x_train, y_train = train[:, 1:], train[:, 0]
    x_test, y_test = test[:, 1:], test[:, 0]
    classes = sorted(list(set(y_train.tolist())))
    mapping = {c: i for i, c in enumerate(classes)}
    y_train = np.array([mapping[c] for c in y_train])
    y_test = np.array([mapping[c] for c in y_test])
    x_train = x_train[:, None, :]
    x_test = x_test[:, None, :]
    return x_train, y_train, x_test, y_test


def _read_uea(root, name):
    try:
        from sktime.datasets import load_from_tsfile
    except Exception as exc:  # pragma: no cover
        raise ImportError("UEA loading requires sktime. Install sktime or use synthetic/UCR.") from exc
    train_path = os.path.join(root, name, name + "_TRAIN.ts")
    test_path = os.path.join(root, name, name + "_TEST.ts")
    x_train, y_train = load_from_tsfile(train_path, return_data_type="numpy3d")
    x_test, y_test = load_from_tsfile(test_path, return_data_type="numpy3d")
    classes = sorted(list(set(y_train.tolist())))
    mapping = {c: i for i, c in enumerate(classes)}
    y_train = np.array([mapping[c] for c in y_train])
    y_test = np.array([mapping[c] for c in y_test])
    return x_train, y_train, x_test, y_test


def _make_synthetic(num_samples, channels, length, num_classes, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros((num_samples, channels, length), dtype=np.float32)
    y = rng.integers(0, num_classes, size=(num_samples,))
    t = np.linspace(0, 1, length)[None, None, :]
    for i in range(num_samples):
        cls = y[i]
        freq = 1.0 + cls * 0.5
        phase = rng.uniform(0, 2 * np.pi)
        amp = 1.0 + 0.1 * rng.normal()
        signal = amp * np.sin(2 * np.pi * freq * t + phase)
        noise = 0.05 * rng.normal(size=(channels, length))
        x[i] = signal + noise
    return x.astype(np.float32), y.astype(np.int64)


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        assert self.x.ndim == 3, "Expected [N, C, T]"

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_datasets(cfg):
    kind = cfg["dataset"]["kind"]
    root = cfg["dataset"]["root"]
    name = cfg["dataset"]["name"]
    channels = cfg["dataset"]["channels"]
    length = cfg["dataset"]["length"]
    num_classes = cfg["dataset"]["num_classes"]
    normalize = cfg["dataset"]["normalize"]

    if kind == "synthetic":
        x, y = _make_synthetic(600, channels, length, num_classes, seed=cfg["seed"])
        split = int(0.8 * len(x))
        x_train, y_train = x[:split], y[:split]
        x_test, y_test = x[split:], y[split:]
    elif kind == "ucr":
        if legacy_utils is not None and hasattr(legacy_utils, "readUCR"):
            x_train, y_train, x_test, y_test = legacy_utils.readUCR(name, root=root)
            x_train = x_train[:, None, :] if x_train.ndim == 2 else x_train
            x_test = x_test[:, None, :] if x_test.ndim == 2 else x_test
        else:
            x_train, y_train, x_test, y_test = _read_ucr(root, name)
    elif kind == "uea":
        if legacy_utils is not None and hasattr(legacy_utils, "readUEA"):
            x_train, y_train, x_test, y_test = legacy_utils.readUEA(name, root=root)
        else:
            x_train, y_train, x_test, y_test = _read_uea(root, name)
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

    if normalize == "zscore":
        x_train, mean, std = _zscore_per_channel(x_train)
        x_test = _apply_zscore(x_test, mean, std)
    else:
        mean, std = None, None

    train_ds = TimeSeriesDataset(x_train.astype(np.float32), y_train.astype(np.int64))
    test_ds = TimeSeriesDataset(x_test.astype(np.float32), y_test.astype(np.int64))

    meta = {
        "num_classes": int(np.max(y_train)) + 1,
        "channels": x_train.shape[1],
        "length": x_train.shape[2],
        "mean": mean,
        "std": std,
    }
    return train_ds, test_ds, meta

