import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SIGNAL_NAMES = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z"
]

ACTIVITY_NAMES = [
    "walking", "walking_upstairs", "walking_downstairs",
    "sitting", "standing", "laying"
]

DEFAULT_DATA_DIR = os.path.join("data", "UCI HAR Dataset")
DEFAULT_VAL_SUBJECTS = (27, 28, 29, 30)


def load_signals(data_dir: str, split: str) -> np.ndarray:
    """Returns [N, 128, 9] float32 windows."""
    signals = []
    signals_dir = os.path.join(data_dir, split, "Inertial Signals")

    if not os.path.exists(signals_dir):
        raise FileNotFoundError(
            f"Directory not found: {signals_dir}. Run `python -m src.download_data` first."
        )

    for sig in SIGNAL_NAMES:
        filepath = os.path.join(signals_dir, f"{sig}_{split}.txt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing expected inertial signal file: {filepath}")
        signals.append(np.loadtxt(filepath))

    stacked = np.stack(signals, axis=0)

    # DATA VALIDATION: Ensure we have exactly 9 channels and a 3D array
    assert stacked.ndim == 3, f"Expected 3D stacked signals, got {stacked.ndim}D"
    assert stacked.shape[0] == 9, f"Expected 9 channels, got {stacked.shape[0]}"

    return np.transpose(stacked, (1, 2, 0)).astype(np.float32)


def load_labels_and_subjects(data_dir: str, split: str):
    y_path = os.path.join(data_dir, split, f"y_{split}.txt")
    subj_path = os.path.join(data_dir, split, f"subject_{split}.txt")

    if not os.path.exists(y_path) or not os.path.exists(subj_path):
        raise FileNotFoundError(f"Missing label or subject files in {data_dir}/{split}")

    y = np.loadtxt(y_path, dtype=int) - 1  # Shift labels from 1-6 to 0-5
    subj = np.loadtxt(subj_path, dtype=int)

    # DATA VALIDATION: Ensure every label has a corresponding subject
    assert len(y) == len(subj), f"Mismatch: {len(y)} labels but {len(subj)} subjects"

    return y, subj


class ChannelStandardizer:
    """
    Per-channel z-scoring. Fit on the training split only, so no statistics from
    validation or test subjects leak into training. Gravity-dominated total_acc
    channels sit near ±1 g while body_acc is ~0.1 g; without this the encoder's
    first layer is dominated by the total_acc scale.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "ChannelStandardizer":
        self.mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = X.std(axis=(0, 1), keepdims=True).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("ChannelStandardizer.transform called before fit")
        return ((X - self.mean) / (self.std + self.eps)).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.ravel().tolist(), "std": self.std.ravel().tolist()}


class HARDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # DATA VALIDATION: Ensure X and y lengths match before creating tensors
        assert len(X) == len(y), f"Features length ({len(X)}) does not match labels length ({len(y)})"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def split_by_subject(subjects: np.ndarray, val_subjects) -> tuple:
    """Boolean (train_mask, val_mask) with zero subject overlap."""
    val_mask = np.isin(subjects, val_subjects)
    train_mask = ~val_mask
    overlap = set(subjects[train_mask]) & set(subjects[val_mask])
    assert not overlap, f"DATA LEAKAGE DETECTED! Overlapping subjects: {overlap}"
    return train_mask, val_mask


def get_dataloaders(
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = 64,
    val_subjects: tuple = DEFAULT_VAL_SUBJECTS,
    standardize: bool = True,
    seed: int = 42,
):
    X_train_raw = load_signals(data_dir, "train")
    y_train_raw, subj_train = load_labels_and_subjects(data_dir, "train")

    X_test_raw = load_signals(data_dir, "test")
    y_test_raw, subj_test = load_labels_and_subjects(data_dir, "test")

    train_mask, val_mask = split_by_subject(subj_train, val_subjects)
    # The official UCI test subjects must also be disjoint from everything we train on.
    assert not (set(subj_train) & set(subj_test)), "Train/test subject overlap in UCI split"

    X_tr, X_va, X_te = X_train_raw[train_mask], X_train_raw[val_mask], X_test_raw
    scaler = None
    if standardize:
        scaler = ChannelStandardizer().fit(X_tr)
        X_tr, X_va, X_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    train_dataset = HARDataset(X_tr, y_train_raw[train_mask])
    val_dataset   = HARDataset(X_va, y_train_raw[val_mask])
    test_dataset  = HARDataset(X_te, y_test_raw)

    # Seeded generator: shuffling order is reproducible per seed.
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
