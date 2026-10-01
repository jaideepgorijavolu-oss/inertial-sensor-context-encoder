import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

from src.dataset import ACTIVITY_NAMES, DEFAULT_DATA_DIR, DEFAULT_VAL_SUBJECTS, get_dataloaders
from src.models import (
    DEFAULT_LLM,
    ContextEmbeddingModel,
    DirectClassifier,
    MatchedCapacityClassifier,
    SensorEncoder,
    trainable_state_dict,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CONDITIONS = ("direct", "matched", "context")
CONDITION_LABELS = {
    "direct": "Direct sensor classifier",
    "matched": "Matched-capacity, no LLM (ablation)",
    "context": "Context-embedding model (frozen LLM)",
    "shuffled": "Context model, shuffled sensor tokens (control)",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


@torch.no_grad()
def predict(model: nn.Module, loader, device, shuffle: bool = False) -> tuple:
    model.eval()
    preds, labels = [], []
    for X, y in loader:
        X = X.to(device)
        if isinstance(model, ContextEmbeddingModel):
            logits = model(X, shuffle_embeddings=shuffle)
        else:
            logits = model(X)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(preds)


def macro_f1(model, loader, device, shuffle: bool = False) -> float:
    y, p = predict(model, loader, device, shuffle)
    return float(f1_score(y, p, average="macro"))


def full_metrics(model, loader, device, shuffle: bool = False) -> dict:
    y, p = predict(model, loader, device, shuffle)
    return {
        "macro_f1": float(f1_score(y, p, average="macro")),
        "accuracy": float((y == p).mean()),
        "per_class_f1": dict(zip(ACTIVITY_NAMES, map(float, f1_score(y, p, average=None, labels=range(6))))),
        "confusion_matrix": confusion_matrix(y, p, labels=range(6)).tolist(),
    }


@torch.no_grad()
def latency_ms_per_window(model: nn.Module, device, batch_size: int = 1, reps: int = 20) -> float:
    """Median single-window inference latency (the deployment-relevant number)."""
    model.eval()
    x = torch.randn(batch_size, 128, 9, device=device)
    for _ in range(3):
        model(x)
    times = []
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000 / batch_size)
    return float(np.median(times))


def train_model(model, train_loader, val_loader, device, epochs: int, lr: float, tag: str, out_dir: str):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1, best_epoch, best_state = -1.0, 0, None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_f1 = macro_f1(model, val_loader, device)
        train_loss = total_loss / len(train_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1})
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
            best_state = trainable_state_dict(model)  # real copy, see trainable_state_dict
        print(f"  [{tag}] epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} | val macro-F1 {val_f1:.4f}")

    missing, unexpected = model.load_state_dict(best_state, strict=False)
    assert not unexpected and all(k.startswith("llm.") for k in missing), (missing, unexpected)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, f"best_{tag}.pt"))
    return {"best_val_macro_f1": best_val_f1, "best_epoch": best_epoch, "history": history}


def build(condition: str, args) -> nn.Module:
    encoder = SensorEncoder()
    if condition == "direct":
        return DirectClassifier(encoder)
    if condition == "matched":
        return MatchedCapacityClassifier(encoder, llm_dim=args.llm_dim)
    return ContextEmbeddingModel.from_pretrained(
        encoder, args.llm, gradient_checkpointing=args.gradient_checkpointing
    )


def run_seed(seed: int, args, device) -> dict:
    set_seed(seed)
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, val_subjects=DEFAULT_VAL_SUBJECTS,
        standardize=not args.no_standardize, seed=seed,
    )
    out_dir = os.path.join(args.out_dir, f"seed{seed}")
    results = {}
    for cond in args.conditions:
        print(f"\n=== seed {seed} | {CONDITION_LABELS[cond]} ===")
        set_seed(seed)
        model = build(cond, args).to(device)
        trainable, frozen = count_params(model)
        print(f"  trainable params {trainable:,} | frozen params {frozen:,}")
        epochs = args.context_epochs if cond == "context" else args.epochs
        lr = args.context_lr if cond == "context" else args.lr
        train_info = train_model(model, train_loader, val_loader, device, epochs, lr, cond, out_dir)

        results[cond] = {
            **full_metrics(model, test_loader, device),
            "trainable_params": trainable,
            "frozen_params": frozen,
            "latency_ms": latency_ms_per_window(model, device),
            **train_info,
        }
        print(f"  --> test macro-F1 {results[cond]['macro_f1']:.4f}")

        if cond == "context":
            # Average over several random permutations; a single permutation is noisy.
            torch.manual_seed(seed)
            shuffled = [full_metrics(model, test_loader, device, shuffle=True) for _ in range(args.shuffle_repeats)]
            results["shuffled"] = {
                "macro_f1": float(np.mean([s["macro_f1"] for s in shuffled])),
                "accuracy": float(np.mean([s["accuracy"] for s in shuffled])),
                "trainable_params": 0,
            }
            print(f"  --> shuffled-control test macro-F1 {results['shuffled']['macro_f1']:.4f}")
        del model

    if scaler is not None:
        results["standardizer"] = scaler.state_dict()
    return results


def summarize(per_seed: dict, args) -> tuple:
    rows = []
    summary = {}
    for cond in [*args.conditions, *(["shuffled"] if "context" in args.conditions else [])]:
        f1s = [per_seed[s][cond]["macro_f1"] for s in per_seed]
        first = per_seed[next(iter(per_seed))][cond]
        summary[cond] = {
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "macro_f1_per_seed": f1s,
            "trainable_params": first["trainable_params"],
            "latency_ms": first.get("latency_ms"),
        }
        s = summary[cond]
        lat = f"{s['latency_ms']:.2f}" if s["latency_ms"] is not None else "-"
        rows.append(
            f"| {CONDITION_LABELS[cond]} | {s['macro_f1_mean']:.4f} ± {s['macro_f1_std']:.4f} "
            f"| {s['trainable_params']:,} | {lat} |"
        )
    table = "\n".join([
        f"Seeds: {list(per_seed)} | epochs: {args.epochs} (context: {args.context_epochs}) | "
        f"standardized inputs: {not args.no_standardize} | device: {args.device_name}",
        "",
        "| Condition | Test Macro-F1 (mean ± std) | Trainable params | Latency ms/window |",
        "| :--- | :--- | ---: | ---: |",
        *rows,
    ])
    return summary, table


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sensor-to-LLM context projection benchmark")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--context-epochs", type=int, default=None, help="defaults to --epochs (equal budget)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--context-lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--llm", default=DEFAULT_LLM)
    p.add_argument("--llm-dim", type=int, default=960, help="hidden size used by the matched ablation")
    p.add_argument("--gradient-checkpointing", action="store_true", help="trade compute for LLM activation memory")
    p.add_argument("--shuffle-repeats", type=int, default=5)
    p.add_argument("--no-standardize", action="store_true")
    args = p.parse_args(argv)
    if args.context_epochs is None:
        args.context_epochs = args.epochs
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Device: {args.device_name} | seeds: {args.seeds} | conditions: {args.conditions}")

    per_seed = {seed: run_seed(seed, args, device) for seed in args.seeds}
    summary, table = summarize(per_seed, args)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"config": vars(args), "summary": summary, "per_seed": per_seed}, f, indent=2)
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nSaved {args.out_dir}/results.json and {args.out_dir}/results.md")


if __name__ == "__main__":
    main()
