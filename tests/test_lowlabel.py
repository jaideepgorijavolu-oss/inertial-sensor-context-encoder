"""Leakage boundaries, sampling, augmentation physics, contrastive loss and the study CLI (synthetic data)."""
import json
import math
import os

import numpy as np
import pytest
import torch

import src.lowlabel as lowlabel
from src.augment import AugConfig, augment, random_rotations
from src.dataset import SIGNAL_NAMES
from src.lowlabel_data import StudyData, budget_subset, chain_ids, class_orders, quota
from src.models import SensorEncoder
from src.ssl import collapse_stats, nt_xent

BOUT_WINDOWS = 6


def make_windows(subjects, rng, bout_windows=BOUT_WINDOWS):
    """One bout per class per subject, cut into 50%-overlapping 128-sample windows like UCI HAR."""
    X, y, s, bout = [], [], [], []
    b = 0
    for subj in subjects:
        for c in range(6):
            sig = rng.normal(size=((bout_windows + 1) * 64, 9)).round(6)
            sig[:, 0:3] += sig[:, 3:6] + np.array([0.0, 0.0, 1.0])     # total = body + gravity
            for k in range(bout_windows):
                X.append(sig[k * 64:k * 64 + 128]); y.append(c); s.append(subj); bout.append(b)
            b += 1
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(s), np.asarray(bout)


def write_uci(root, split, X, y, s):
    sig_dir = os.path.join(root, split, "Inertial Signals")
    os.makedirs(sig_dir)
    for ch, name in enumerate(SIGNAL_NAMES):
        np.savetxt(os.path.join(sig_dir, f"{name}_{split}.txt"), X[:, :, ch], fmt="%.6e")
    np.savetxt(os.path.join(root, split, f"y_{split}.txt"), y + 1, fmt="%d")
    np.savetxt(os.path.join(root, split, f"subject_{split}.txt"), s, fmt="%d")


@pytest.fixture
def uci_dir(tmp_path):
    rng = np.random.default_rng(0)
    root = str(tmp_path / "UCI HAR Dataset")
    write_uci(root, "train", *make_windows([1, 3, 27, 30], rng)[:3])
    write_uci(root, "test", *make_windows([2], rng)[:3])
    return root


def test_chain_ids_recover_bouts_and_ignore_cross_subject_matches():
    X, y, s, bout = make_windows([1, 3], np.random.default_rng(1))
    X[BOUT_WINDOWS * 6] = X[0]          # subject 3's first window duplicates subject 1's: must not link
    X[BOUT_WINDOWS * 6 + 1, :64] = X[0, 64:]
    ids, pos = chain_ids(X, s)
    for b in np.unique(bout):
        members = np.flatnonzero(bout == b)
        assert len(set(ids[members])) == 1 and list(pos[members]) == list(range(len(members)))
    assert len(np.unique(ids)) == len(np.unique(bout))


def test_budget_subsets_nested_exact_contiguous_and_seeded():
    X, y, s, _ = make_windows([1, 3, 5, 6], np.random.default_rng(2), bout_windows=12)
    ids, pos = chain_ids(X, s)
    idx = np.arange(len(y))
    orders = class_orders(idx, y, s, ids, pos, seed=42, salt=1)
    prev = set()
    for b in (0.01, 0.1, 0.25, 1.0):
        sub = budget_subset(orders, b)
        assert prev <= set(sub)                                  # nested
        prev = set(sub)
        counts = np.bincount(y[sub], minlength=6)
        assert all(counts[c] == quota(48, b) for c in range(6))  # exact quota, floor of 2
        for c in np.unique(ids[sub]):                            # chosen windows form whole-block prefixes
            p = np.sort(pos[sub][ids[sub] == c])
            for blk in set(p // 4):
                in_blk = p[p // 4 == blk]
                assert list(in_blk) == list(range(blk * 4, blk * 4 + len(in_blk)))
    assert quota(48, 0.01) == 2
    again = budget_subset(class_orders(idx, y, s, ids, pos, seed=42, salt=1), 0.25)
    other = budget_subset(class_orders(idx, y, s, ids, pos, seed=43, salt=1), 0.25)
    assert np.array_equal(again, budget_subset(orders, 0.25)) and not np.array_equal(again, other)


def test_study_data_partitions_and_standardizer_boundary(uci_dir):
    d = StudyData(uci_dir)
    assert set(d.subjects[d.pool_idx]) == {1, 3} and set(d.subjects[d.val_idx]) == {27, 30}
    assert 2 not in set(d.subjects)                                        # test file never read
    np.testing.assert_allclose(d.scaler.mean.ravel(), d.X_raw[d.pool_idx].mean(axis=(0, 1)), rtol=1e-5)
    assert np.array_equal(d.pool_inputs(), d.X_raw[d.pool_idx])
    for b in (0.01, 1.0):
        sub = d.subsets(7, b)
        assert set(sub["train"]) <= set(d.pool_idx) and set(sub["val"]) <= set(d.val_idx)


def test_augmentation_keeps_shapes_and_gravity_consistency():
    X, *_ = make_windows([1], np.random.default_rng(3))
    x = torch.tensor(X)
    cstd = x.std(dim=(0, 1))
    g = torch.Generator().manual_seed(0)
    none = AugConfig(rotate=False, scale=False, jitter=False)
    assert torch.allclose(augment(x, none, cstd), x, atol=1e-5)
    rot = AugConfig(rotate=True, scale=False, jitter=False, max_degrees=30)
    xa = augment(x, rot, cstd, g)
    assert xa.shape == x.shape
    for sl in (slice(3, 6), slice(6, 9)):                                  # rotations preserve norms
        assert torch.allclose(xa[..., sl].norm(dim=-1), x[..., sl].norm(dim=-1), atol=1e-4)
    grav = lambda t: t[..., 0:3] - t[..., 3:6]
    assert torch.allclose(grav(xa).norm(dim=-1), grav(x).norm(dim=-1), atol=1e-4)
    full = augment(x, AugConfig.named("strong"), cstd, g)
    assert torch.isfinite(full).all() and not torch.allclose(full, x)
    # jitter on body_acc reaches total_acc identically: gravity estimate unchanged by jitter
    jit = augment(x, AugConfig(rotate=False, scale=False, jitter=True), cstd, g)
    assert torch.allclose(grav(jit), grav(x), atol=1e-5)
    R = random_rotations(500, 30.0, torch.Generator().manual_seed(1))
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(500, 3, 3), atol=1e-5)
    angles = torch.arccos(((R.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1))
    assert torch.allclose(torch.linalg.det(R), torch.ones(500), atol=1e-5)
    assert angles.max() <= math.radians(30) + 1e-4


def test_nt_xent_matches_reference_and_trains_encoder():
    torch.manual_seed(0)
    z1, z2, tau = torch.randn(5, 8), torch.randn(5, 8), 0.5
    z = torch.nn.functional.normalize(torch.cat([z1, z2]), dim=1)
    ref = 0.0
    for i in range(10):
        pos = (i + 5) % 10
        others = [j for j in range(10) if j != i]
        ref += -torch.log(torch.exp(z[i] @ z[pos] / tau) / sum(torch.exp(z[i] @ z[j] / tau) for j in others))
    assert torch.allclose(nt_xent(z1, z2, tau), ref / 10, atol=1e-5)
    eye = torch.eye(8)[:5]
    assert nt_xent(eye, eye, 0.1) < nt_xent(eye, eye.roll(1, 0), 0.1)
    enc = SensorEncoder()
    loss = nt_xent(enc(torch.randn(6, 128, 9)), enc(torch.randn(6, 128, 9)), 0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())


def test_collapse_stats_distinguishes_collapsed_from_spread():
    torch.manual_seed(0)
    spread = collapse_stats(torch.randn(512, 64))
    collapsed = collapse_stats(torch.ones(512, 64) + 1e-6 * torch.randn(512, 64))
    assert abs(spread["norm_std"] - 1 / 8) < 0.02 and collapsed["norm_std"] < 0.01
    assert spread["effective_rank"] > 40


def test_frozen_encoders_and_knn(uci_dir):
    ctx = lowlabel.Ctx(StudyData(uci_dir), torch.device("cpu"))
    torch.manual_seed(0)
    init = {k: v.clone() for k, v in SensorEncoder().state_dict().items()}
    rand = lowlabel.recalibrated_random_encoder(ctx, seed=0)
    changed = {k for k in init if not torch.equal(init[k], rand[k])}
    assert changed and all("running_" in k or "num_batches" in k for k in changed)   # only BN buffers
    enc = lowlabel.encoder_from(rand, ctx.device)
    assert not any(p.requires_grad for p in enc.parameters()) and not enc.training
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    sub = ctx.data.subsets(0, 1.0)
    F_tr = lowlabel.embed(enc, ctx.norm(ctx.X[sub["train"]]))
    lowlabel.fit_probe(F_tr, ctx.data.y[sub["train"]], F_tr, ctx.data.y[sub["train"]])
    assert all(torch.equal(before[k], v) for k, v in enc.state_dict().items())
    ref, yref = np.eye(6), np.arange(6)
    q = np.eye(6) + 0.01
    assert list(lowlabel.knn_predict(ref, yref, q, 1, False)) == list(range(6))


def test_study_cli_end_to_end_without_touching_test_until_evaluation(uci_dir, tmp_path, monkeypatch):
    run = str(tmp_path / "run")
    common = ["--data-dir", uci_dir, "--run-dir", run, "--ssl-epochs", "1", "--steps", "4", "--cpu"]

    def forbidden(*a, **k):
        raise AssertionError("test split loaded during training/selection")

    monkeypatch.setattr(lowlabel, "load_test", forbidden)
    lowlabel.main(["stage-a", *common])
    lowlabel.main(["stage-b", *common, "--seeds", "1", "2", "--budgets", "0.01", "1.0"])
    sel = json.load(open(os.path.join(run, "selections.json")))
    assert len(sel["entries"]) == 2 * 2 * len(lowlabel.CONDITIONS)
    e = sel["entries"]["1/0.01/ssl_knn"]
    assert e["n_train_labels"] == 12 and e["ssl_pretraining"]["sha256"]
    manifest = json.load(open(os.path.join(run, "split_manifest.json")))
    assert not set(manifest["pool_subjects"]) & set(manifest["val_subjects"])

    monkeypatch.undo()
    lowlabel.main(["evaluate-test", *common])
    with pytest.raises(RuntimeError):
        lowlabel.main(["evaluate-test", *common])                 # one-time evaluation
    with pytest.raises(RuntimeError):
        lowlabel.main(["stage-b", *common, "--seeds", "3"])      # selections frozen after test
    lowlabel.main(["report", "--run-dir", run])
    assert os.path.exists(os.path.join(run, "f1_vs_budget.png"))
    test = json.load(open(os.path.join(run, "test_metrics.json")))
    k = "1/1.0/sup"
    cm = np.array(test["entries"][k]["confusion_matrix"])
    assert cm.sum() == test["n_test"] == len(test["entries"][k]["predictions"])
