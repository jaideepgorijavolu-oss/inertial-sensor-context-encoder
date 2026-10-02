import numpy as np
import pytest

from src.dataset import ChannelStandardizer, split_by_subject


def test_standardizer_uses_train_stats_only():
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 2.0, size=(200, 128, 9)).astype(np.float32)
    test = rng.normal(-3.0, 0.5, size=(50, 128, 9)).astype(np.float32)
    scaler = ChannelStandardizer().fit(train)

    z = scaler.transform(train)
    assert np.allclose(z.mean(axis=(0, 1)), 0, atol=1e-3)
    assert np.allclose(z.std(axis=(0, 1)), 1, atol=1e-3)
    # Test data is transformed with *train* statistics, so it is not re-centred.
    assert scaler.transform(test).mean() < -3


def test_standardizer_requires_fit():
    with pytest.raises(RuntimeError):
        ChannelStandardizer().transform(np.zeros((1, 128, 9), dtype=np.float32))


def test_split_by_subject_is_disjoint():
    subjects = np.array([1, 1, 3, 27, 27, 30, 5])
    train_mask, val_mask = split_by_subject(subjects, (27, 28, 29, 30))
    assert set(subjects[val_mask]) == {27, 30}
    assert set(subjects[train_mask]) == {1, 3, 5}
    assert not (train_mask & val_mask).any()
