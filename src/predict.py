"""
Inference example: rebuild a trained model from its run config, checkpoint and normalization
statistics, then classify UCI HAR test windows.

    python -m src.predict --condition direct --seed 42 --limit 10
"""
import argparse
import json
import os

import numpy as np
import torch

from src.dataset import ACTIVITY_NAMES, load_signals
from src.train import build, parse_args


def load_model(out_dir: str, condition: str, seed: int, device: torch.device):
    with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
        run = json.load(f)
    cfg = run["config"]
    args = parse_args(["--llm", cfg["llm"], "--llm-dim", str(cfg["llm_dim"]),
                       "--sensor-tokens", str(cfg["sensor_tokens"]), "--lora-r", str(cfg["lora_r"])])
    model = build(condition, args).to(device)
    state = torch.load(os.path.join(out_dir, f"seed{seed}", f"best_{condition}.pt"), map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("llm.") for k in missing), (missing, unexpected)
    if cfg.get("no_standardize"):
        return model.eval(), None, None
    stats = run["per_seed"][str(seed)].get("standardizer")
    if stats is None:
        raise RuntimeError(f"run in {out_dir} was standardized but has no saved statistics for seed {seed}")
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, -1)
    return model.eval(), mean, std


@torch.no_grad()
def classify(model, windows: np.ndarray, mean, std, device) -> list:
    """windows: [N, 128, 9] raw inertial signals in the UCI HAR channel order."""
    if mean is not None:
        windows = (windows - mean) / (std + 1e-6)
    x = torch.tensor(windows, dtype=torch.float32, device=device)
    return [ACTIVITY_NAMES[i] for i in model(x).argmax(dim=-1).tolist()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--condition", default="direct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=os.path.join("data", "UCI HAR Dataset"))
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = load_model(args.out_dir, args.condition, args.seed, device)
    windows = load_signals(args.data_dir, "test")[: args.limit]
    labels = np.loadtxt(os.path.join(args.data_dir, "test", "y_test.txt"), dtype=int)[: args.limit] - 1
    for pred, true in zip(classify(model, windows, mean, std, device), labels):
        print(f"predicted {pred:<20} actual {ACTIVITY_NAMES[true]}")


if __name__ == "__main__":
    main()
