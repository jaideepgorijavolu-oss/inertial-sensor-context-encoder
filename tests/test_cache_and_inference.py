"""Resume-cache safety and inference after interrupted or unstandardized runs."""
import json

import numpy as np
import pytest
import torch

from src.dataset import SIGNAL_NAMES
from src.predict import classify, load_model
from src.train import main, parse_args, run_seed, save_metrics

CPU = torch.device("cpu")


def write_split(root, split, subjects, rng):
    sig_dir = root / split / "Inertial Signals"
    sig_dir.mkdir(parents=True)
    n = len(subjects)
    for sig in SIGNAL_NAMES:
        np.savetxt(sig_dir / f"{sig}_{split}.txt", rng.normal(size=(n, 128)), fmt="%.4f")
    np.savetxt(root / split / f"y_{split}.txt", np.arange(n) % 6 + 1, fmt="%d")
    np.savetxt(root / split / f"subject_{split}.txt", subjects, fmt="%d")


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "UCI HAR Dataset"
    write_split(root, "train", [1] * 12 + [3] * 12 + [27] * 6, rng)
    write_split(root, "test", [2] * 12, rng)
    return root


def argv(data, out, *extra):
    return ["--data-dir", str(data), "--out-dir", str(out), "--seeds", "1", "--conditions", "direct", "matched",
            "--epochs", "1", "--batch-size", "8", "--llm-dim", "32", *extra]


def assert_predicts(out, cond="matched"):
    model, mean, std = load_model(str(out), cond, 1, CPU)
    preds = classify(model, np.random.default_rng(1).normal(size=(4, 128, 9)).astype(np.float32), mean, std, CPU)
    assert len(preds) == 4


def test_resume_after_interrupt_following_last_condition_save(data, tmp_path):
    out = tmp_path / "out"
    args = parse_args(argv(data, out))
    args.device_name = "cpu"
    path = out / "seed1" / "metrics.json"

    class Interrupted(Exception):
        pass

    def save_then_die(results):
        save_metrics(str(path), results, args)
        if all(c in results for c in args.conditions):
            raise Interrupted  # killed right after the last per-condition save

    with pytest.raises(Interrupted):
        run_seed(1, args, CPU, {}, on_condition_done=save_then_die)
    main(argv(data, out))  # resume: nothing left to train
    assert_predicts(out)


@pytest.mark.parametrize("extra", [[], ["--no-standardize"]], ids=["standardized", "no-standardize"])
def test_inference_in_both_normalization_modes(data, tmp_path, extra):
    out = tmp_path / "out"
    main(argv(data, out, *extra))
    assert_predicts(out)
    assert_predicts(out, "direct")


def test_legacy_cache_without_schema_version_is_not_reused(data, tmp_path):
    out = tmp_path / "out"
    main(argv(data, out))
    path = out / "seed1" / "metrics.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    legacy = {"run_key": saved["run_key"], "results": saved["results"]}  # pre-versioning format
    legacy["results"]["direct"]["macro_f1"] = 0.123456
    path.write_text(json.dumps(legacy), encoding="utf-8")

    main(argv(data, out))
    rerun = json.loads(path.read_text(encoding="utf-8"))["results"]["direct"]
    assert rerun["macro_f1"] != 0.123456  # retrained, not reused
    assert rerun["provenance"]["schema_version"] == json.loads(path.read_text())["schema_version"]


def test_reused_results_keep_their_training_provenance(data, tmp_path):
    out = tmp_path / "out"
    main(argv(data, out))
    first = json.loads((out / "seed1" / "metrics.json").read_text(encoding="utf-8"))["results"]["direct"]["provenance"]
    main(argv(data, out))
    report = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert report["per_seed"]["1"]["direct"]["provenance"] == first
    assert "report_environment" in report and "environment" not in report


def test_missing_controls_without_checkpoint_fail_loudly(data, tmp_path):
    out = tmp_path / "out"
    main(argv(data, out))
    path = out / "seed1" / "metrics.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    # A cached LLM condition whose controls were never computed and whose checkpoint is gone.
    saved["results"]["context"] = {**saved["results"]["direct"], "settings": {}}
    path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(RuntimeError, match="control"):
        main(["--data-dir", str(data), "--out-dir", str(out), "--seeds", "1", "--conditions", "context",
              "--epochs", "1", "--batch-size", "8", "--llm-dim", "32"])
