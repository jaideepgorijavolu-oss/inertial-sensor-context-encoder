"""End-to-end run of the training CLI on a tiny synthetic dataset in UCI HAR layout."""
import json
import os

import numpy as np

from src.dataset import SIGNAL_NAMES
from src.train import main


def write_split(root, split, subjects, rng):
    sig_dir = os.path.join(root, split, "Inertial Signals")
    os.makedirs(sig_dir)
    n = len(subjects)
    for sig in SIGNAL_NAMES:
        np.savetxt(os.path.join(sig_dir, f"{sig}_{split}.txt"), rng.normal(size=(n, 128)), fmt="%.4f")
    np.savetxt(os.path.join(root, split, f"y_{split}.txt"), np.arange(n) % 6 + 1, fmt="%d")
    np.savetxt(os.path.join(root, split, f"subject_{split}.txt"), subjects, fmt="%d")


def test_train_cli_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "UCI HAR Dataset"
    write_split(str(data), "train", [1] * 12 + [3] * 12 + [27] * 6 + [30] * 6, rng)
    write_split(str(data), "test", [2] * 12, rng)
    out = tmp_path / "out"

    main([
        "--data-dir", str(data), "--out-dir", str(out), "--seeds", "1", "2",
        "--conditions", "direct", "matched", "--epochs", "1", "--batch-size", "8", "--llm-dim", "32",
    ])

    results = json.loads((out / "results.json").read_text())
    assert set(results["summary"]) == {"direct", "matched"}
    assert len(results["summary"]["direct"]["macro_f1_per_seed"]) == 2
    assert (out / "seed1" / "best_direct.pt").exists()
    assert "Matched-capacity" in (out / "results.md").read_text(encoding="utf-8")

    # Rerunning with the same settings reuses saved seeds instead of retraining.
    ckpt = out / "seed1" / "best_direct.pt"
    mtime = ckpt.stat().st_mtime
    main([
        "--data-dir", str(data), "--out-dir", str(out), "--seeds", "1", "2",
        "--conditions", "direct", "matched", "--epochs", "1", "--batch-size", "8", "--llm-dim", "32",
    ])
    assert ckpt.stat().st_mtime == mtime
    assert json.loads((out / "results.json").read_text())["summary"] == results["summary"]


def test_condition_training_does_not_depend_on_run_order(tmp_path):
    """The matched model trained alone, after `direct`, or in a resumed run must be identical."""
    import torch

    rng = np.random.default_rng(1)
    data = tmp_path / "UCI HAR Dataset"
    write_split(str(data), "train", [1] * 20 + [3] * 20 + [27] * 6, rng)
    write_split(str(data), "test", [2] * 12, rng)
    common = ["--data-dir", str(data), "--seeds", "5", "--epochs", "2", "--batch-size", "8", "--llm-dim", "32"]

    main([*common, "--out-dir", str(tmp_path / "alone"), "--conditions", "matched"])
    main([*common, "--out-dir", str(tmp_path / "after"), "--conditions", "direct", "matched"])
    main([*common, "--out-dir", str(tmp_path / "resumed"), "--conditions", "direct"])
    main([*common, "--out-dir", str(tmp_path / "resumed"), "--conditions", "direct", "matched"])

    def load(run):
        return torch.load(tmp_path / run / "seed5" / "best_matched.pt")

    reference = load("alone")
    for run in ("after", "resumed"):
        other = load(run)
        assert reference.keys() == other.keys()
        for k in reference:
            assert torch.equal(reference[k], other[k]), (run, k)
