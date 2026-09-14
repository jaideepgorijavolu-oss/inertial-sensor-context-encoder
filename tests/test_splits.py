import os
import numpy as np
import pytest
from src.dataset import load_labels_and_subjects

def test_subject_split_isolation():
    """Verify strict subject-level separation to guarantee zero data leakage."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data", "UCI HAR Dataset")
    
    if not os.path.exists(data_dir):
        pytest.skip("UCI HAR Dataset not found at expected path; skipping data split verification.")

    val_subjects = (27, 28, 29, 30)
    
    _, subj_train = load_labels_and_subjects(data_dir, "train")
    _, subj_test = load_labels_and_subjects(data_dir, "test")
    
    val_mask = np.isin(subj_train, val_subjects)
    train_mask = ~val_mask
    
    train_subjs = set(subj_train[train_mask])
    val_subjs = set(subj_train[val_mask])
    test_subjs = set(subj_test)
    
    # Assert zero overlap across splits
    assert len(train_subjs.intersection(val_subjs)) == 0, "Leakage detected between train and val splits!"
    assert len(train_subjs.intersection(test_subjs)) == 0, "Leakage detected between train and test splits!"
    assert len(val_subjs.intersection(test_subjs)) == 0, "Leakage detected between val and test splits!"