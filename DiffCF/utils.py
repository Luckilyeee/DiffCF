import os
import numpy as np

def z_score_normalize(x, eps=1e-8):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def label_encoder(y_train, y_test):
    classes = sorted(list(set(y_train.tolist() if isinstance(y_train, np.ndarray) else y_train)))
    mapping = {c: i for i, c in enumerate(classes)}
    y_train_enc = np.array([mapping[c] for c in y_train])
    y_test_enc = np.array([mapping[c] for c in y_test])
    return y_train_enc, y_test_enc


def readUCR(ds_name, root=None):
    path = root or "/UCRArchive_2018/"
    train_data = np.loadtxt(os.path.join(path, ds_name, ds_name + '_TRAIN.tsv'), delimiter='\t')
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]

    test_data = np.loadtxt(os.path.join(path, ds_name, ds_name + '_TEST.tsv'), delimiter='\t')
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    y_train, y_test = label_encoder(y_train, y_test)
    x_train = z_score_normalize(x_train)
    x_test = z_score_normalize(x_test)

    return x_train, y_train, x_test, y_test


def readUEA(name, root=None):
    DATA_PATH = root or "/MTS_DATA"
    try:
        from sktime.datasets import load_from_tsfile
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("UEA loading requires sktime. Install sktime or use synthetic/UCR.") from exc

    X_train, y_train = load_from_tsfile(os.path.join(DATA_PATH, name, name + '_TRAIN.ts'),
                                        return_data_type='numpy3d')
    X_test, y_test = load_from_tsfile(os.path.join(DATA_PATH, name, name + '_TEST.ts'),
                                      return_data_type='numpy3d')
    if name in ["BasicMotions", "RacketSports"]:
        print("Applying z-score normalization for dataset ", name)
        X_train = z_score_normalize(X_train)
        X_test = z_score_normalize(X_test)

    y_train, y_test = label_encoder(y_train, y_test)
    return X_train, y_train, X_test, y_test
