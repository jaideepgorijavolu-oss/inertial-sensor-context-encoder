import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SIGNAL_NAMES = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z"
]

def load_signals(data_dir: str, split: str) -> np.ndarray:
    signals = []
    signals_dir = os.path.join(data_dir, split, "Inertial Signals")
    
    if not os.path.exists(signals_dir):
        raise FileNotFoundError(f"Directory not found: {signals_dir}. Did you extract the dataset correctly?")

    for sig in SIGNAL_NAMES:
        filepath = os.path.join(signals_dir, f"{sig}_{split}.txt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing expected inertial signal file: {filepath}")
        
        data = np.loadtxt(filepath)
        signals.append(data)
    
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

def get_dataloaders(
    data_dir: str = "data/UCI HAR Dataset",
    batch_size: int = 64,
    val_subjects: tuple = (27, 28, 29, 30)
):
    X_train_raw = load_signals(data_dir, "train")
    y_train_raw, subj_train = load_labels_and_subjects(data_dir, "train")

    X_test_raw = load_signals(data_dir, "test")
    y_test_raw, _ = load_labels_and_subjects(data_dir, "test")

    val_mask = np.isin(subj_train, val_subjects)
    train_mask = ~val_mask

    # DATA VALIDATION: Ensure no data leakage between train and val
    train_subjects_unique = set(subj_train[train_mask])
    val_subjects_unique = set(subj_train[val_mask])
    overlap = train_subjects_unique.intersection(val_subjects_unique)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping subjects: {overlap}"

    train_dataset = HARDataset(X_train_raw[train_mask], y_train_raw[train_mask])
    val_dataset   = HARDataset(X_train_raw[val_mask], y_train_raw[val_mask])
    test_dataset  = HARDataset(X_test_raw, y_test_raw)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader