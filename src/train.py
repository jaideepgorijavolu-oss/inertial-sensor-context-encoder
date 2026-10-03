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

CONDITIONS = ("direct", "matched", "context", "context_mt", "context_lora")
CONTEXT_CONDITIONS = ("context", "context_mt", "context_lora")
CONDITION_LABELS = {
    "direct": "Direct sensor classifier",
    "matched": "Matched-capacity, no LLM (ablation)",
    "context": "Context model, 1 sensor token (frozen LLM)",
    "context_mt": "Context model, multi-token (frozen LLM)",
    "context_lora": "Context model, multi-token + LoRA",
    "context_shuffled_global": "Context model, 1 token: globally shuffled sensor (control)",
    "context_mt_shuffled_global": "Context model, multi-token: globally shuffled sensor (control)",
    "context_lora_shuffled_global": "Context model, multi-token + LoRA: globally shuffled sensor (control)",
    "context_zero": "Context model, 1 token: zeroed sensor (control)",
    "context_mt_zero": "Context model, multi-token: zeroed sensor (control)",
    "context_lora_zero": "Context model, multi-token + LoRA: zeroed sensor (control)",
}


def control_keys(cond: str) -> tuple:
    """Result keys of the evaluation-time controls for a context condition."""
    return f"{cond}_shuffled_global", f"{cond}_zero"


def condition_settings(cond: str, args) -> dict:
    """Condition-specific settings; saved results are reused only if these match."""
    if cond == "context_mt":
        return {"sensor_tokens": args.sensor_tokens}
    if cond == "context_lora":
        return {"sensor_tokens": args.sensor_tokens, "lora_r": args.lora_r}
    return {}


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
def predict(model: nn.Module, loader, device, perm: torch.Tensor = None, zero_sensor: bool = False) -> tuple:
    """
    Labels and predictions over the loader's whole (unshuffled) dataset. With `perm`, window
    perm[k] is paired with label k, i.e. inputs are re-paired across the entire evaluation set.
    """
    model.eval()
    X_all, y_all = loader.dataset.X, loader.dataset.y
    if perm is not None:
        X_all = X_all[perm]
    preds = []
    for k in range(0, len(y_all), loader.batch_size):
        X = X_all[k:k + loader.batch_size].to(device)
        logits = model(X, zero_sensor=True) if zero_sensor else model(X)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    return y_all.numpy(), np.concatenate(preds)


def macro_f1(model, loader, device, **kw) -> float:
    y, p = predict(model, loader, device, **kw)
    return float(f1_score(y, p, average="macro"))


def derangement(n: int, generator: torch.Generator) -> torch.Tensor:
    """Uniformly shuffled permutation with no fixed points (no window is paired with itself)."""
    order = torch.randperm(n, generator=generator)
    perm = torch.empty_like(order)
    perm[order] = order.roll(-1)  # each element maps to the next one in a random cycle
    return perm


def control_metrics(model, loader, device, seed: int, repeats: int) -> dict:
    """
    Evaluation-time controls for a context model (no retraining):
      * globally shuffled: every test label is paired with a different window drawn from the
        whole test set (a seeded derangement), averaged over `repeats` permutations. This is
        the empirical chance baseline for the trained model; shuffling only within batches is
        not, because the test set is ordered by subject and activity.
      * zeroed: the sensor tokens are replaced by zeros, so only the prompt remains.
    """
    n = len(loader.dataset)
    runs = [full_metrics(model, loader, device, perm=derangement(n, torch.Generator().manual_seed(seed * 1000 + r)))
            for r in range(repeats)]
    shuffled = {
        "macro_f1": float(np.mean([r["macro_f1"] for r in runs])),
        "macro_f1_std_over_permutations": float(np.std([r["macro_f1"] for r in runs], ddof=1)) if repeats > 1 else 0.0,
        "accuracy": float(np.mean([r["accuracy"] for r in runs])),
        "permutations": repeats,
        "trainable_params": 0,
    }
    zero = {**full_metrics(model, loader, device, zero_sensor=True), "trainable_params": 0}
    return {"shuffled_global": shuffled, "zero": zero}


def full_metrics(model, loader, device, **kw) -> dict:
    y, p = predict(model, loader, device, **kw)
    return {
        "macro_f1": float(f1_score(y, p, average="macro")),
        "accuracy": float((y == p).mean()),
        "per_class_f1": dict(zip(ACTIVITY_NAMES, map(float, f1_score(y, p, average=None, labels=range(6))))),
        "confusion_matrix": confusion_matrix(y, p, labels=range(6)).tolist(),
    }


@torch.no_grad()
def latency_ms_per_window(model: nn.Module, device, batch_size: int = 1, reps: int = 100) -> float:
    """Median single-window inference latency (the deployment-relevant number)."""
    model.eval()
    x = torch.randn(batch_size, 128, 9, device=device)
    for _ in range(10):
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
    settings = condition_settings(condition, args)
    return ContextEmbeddingModel.from_pretrained(
        encoder, args.llm,
        num_sensor_tokens=settings.get("sensor_tokens", 1), lora_r=settings.get("lora_r", 0),
    )


def run_seed(seed: int, args, device, existing: dict = None, on_condition_done=None) -> dict:
    """Train/evaluate every condition for one seed, skipping conditions already in `existing`."""
    results = dict(existing or {})
    out_dir = os.path.join(args.out_dir, f"seed{seed}")
    todo = [c for c in args.conditions if c not in results]
    # Trained context conditions whose controls are missing (e.g. results from before the
    # global-shuffle control existed) are re-evaluated from their checkpoints, not retrained.
    refill = [c for c in args.conditions if c in results and c in CONTEXT_CONDITIONS
              and any(k not in results for k in control_keys(c))
              and os.path.exists(os.path.join(out_dir, f"best_{c}.pt"))]
    if not todo and not refill:
        return results
    set_seed(seed)
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, val_subjects=DEFAULT_VAL_SUBJECTS,
        standardize=not args.no_standardize, seed=seed,
    )

    def add_controls(cond, model):
        controls = control_metrics(model, test_loader, device, seed, args.shuffle_repeats)
        shuffled_k, zero_k = control_keys(cond)
        results[shuffled_k], results[zero_k] = controls["shuffled_global"], controls["zero"]
        print(f"  --> controls: global shuffle macro-F1 {results[shuffled_k]['macro_f1']:.4f} | "
              f"zeroed sensor macro-F1 {results[zero_k]['macro_f1']:.4f}")

    for cond in refill:
        print(f"\n=== seed {seed} | {CONDITION_LABELS[cond]}: controls from checkpoint ===")
        set_seed(seed)
        model = build(cond, args).to(device)
        missing, unexpected = model.load_state_dict(
            torch.load(os.path.join(out_dir, f"best_{cond}.pt"), map_location=device), strict=False)
        assert not unexpected and all(k.startswith("llm.") for k in missing), (missing, unexpected)
        add_controls(cond, model)
        del model
        if on_condition_done is not None:
            on_condition_done(results)

    for cond in todo:
        print(f"\n=== seed {seed} | {CONDITION_LABELS[cond]} ===")
        set_seed(seed)
        # Fresh batch order per condition: the shared generator would otherwise have advanced
        # through every earlier condition, so a condition's training would depend on run order.
        train_loader.generator.manual_seed(seed)
        model = build(cond, args).to(device)
        trainable, frozen = count_params(model)
        print(f"  trainable params {trainable:,} | frozen params {frozen:,}")
        is_context = cond in CONTEXT_CONDITIONS
        epochs = args.context_epochs if is_context else args.epochs
        lr = args.context_lr if is_context else args.lr
        train_info = train_model(model, train_loader, val_loader, device, epochs, lr, cond, out_dir)

        results[cond] = {
            **full_metrics(model, test_loader, device),
            "trainable_params": trainable,
            "frozen_params": frozen,
            "latency_ms": latency_ms_per_window(model, device),
            "settings": condition_settings(cond, args),
            **train_info,
        }
        print(f"  --> test macro-F1 {results[cond]['macro_f1']:.4f}")

        if is_context:
            add_controls(cond, model)
        del model
        if on_condition_done is not None:
            on_condition_done(results)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if scaler is not None:
        results["standardizer"] = scaler.state_dict()
    return results


def summarize(per_seed: dict, args) -> tuple:
    rows = []
    summary = {}
    ordered = [*args.conditions, *(k for c in args.conditions if c in CONTEXT_CONDITIONS for k in control_keys(c))]
    ordered = [k for k in ordered if all(k in per_seed[s] for s in per_seed)]
    for cond in ordered:
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
        f"sensor tokens (multi-token): {args.sensor_tokens} | LoRA rank: {args.lora_r} | "
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
    p.add_argument("--shuffle-repeats", type=int, default=5, help="permutations for the global-shuffle control")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="retrain seeds even if saved results exist")
    p.add_argument("--sensor-tokens", type=int, default=8, help="soft tokens for the multi-token conditions")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (q/v projections) for context_lora")
    p.add_argument("--remeasure-latency", action="store_true",
                   help="re-time every condition on an otherwise idle device and update saved results")
    args = p.parse_args(argv)
    if args.context_epochs is None:
        args.context_epochs = args.epochs
    return args


def run_environment(args) -> dict:
    """Versions, code commit, dataset fingerprint and model revision, saved with every run."""
    import hashlib
    import platform
    import subprocess

    import sklearn
    import transformers

    def git(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    digest = hashlib.sha256()
    for split in ("train", "test"):
        for name in (f"y_{split}.txt", f"subject_{split}.txt"):
            path = os.path.join(args.data_dir, split, name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    digest.update(f.read())
    revision = None
    try:
        from huggingface_hub import model_info
        revision = model_info(args.llm).sha
    except Exception:
        pass
    return {
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "scikit_learn": sklearn.__version__, "numpy": np.__version__,
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "dataset_labels_sha256": digest.hexdigest(), "llm": args.llm, "llm_revision": revision,
    }


def run_key(args) -> dict:
    """Shared settings that must match for saved results to be reused."""
    keys = ("data_dir", "epochs", "context_epochs", "lr", "context_lr", "batch_size",
            "llm", "llm_dim", "shuffle_repeats", "no_standardize")
    return {k: getattr(args, k) for k in keys}


def load_reusable(path: str, args) -> dict:
    """Completed per-condition results from an earlier run with matching settings."""
    if args.no_resume or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    key = run_key(args)
    if any(saved.get("run_key", {}).get(k) != v for k, v in key.items()):
        return {}
    results = saved["results"]
    keep = {}
    for cond in CONDITIONS:
        if cond in results and results[cond].get("settings", {}) == condition_settings(cond, args):
            keep[cond] = results[cond]
            if cond in CONTEXT_CONDITIONS:
                keep.update({k: results[k] for k in control_keys(cond) if k in results})
    if "standardizer" in results:
        keep["standardizer"] = results["standardizer"]
    return keep


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Device: {args.device_name} | seeds: {args.seeds} | conditions: {args.conditions}")

    per_seed = {}
    for seed in args.seeds:
        # Results are saved after every condition, so an interrupted run resumes where it
        # stopped, and adding a new condition later only trains that condition.
        path = os.path.join(args.out_dir, f"seed{seed}", "metrics.json")
        existing = load_reusable(path, args)
        reused = [c for c in args.conditions if c in existing]
        if reused:
            print(f"Seed {seed}: reusing {reused} from {path}")

        def save(results, path=path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"run_key": run_key(args), "results": results}, f, indent=2)

        per_seed[seed] = run_seed(seed, args, device, existing, on_condition_done=save)
        save(per_seed[seed])

    if args.remeasure_latency:
        # Latency depends only on the architecture, so one clean measurement per condition
        # replaces timings taken while other work shared the machine.
        for cond in args.conditions:
            set_seed(0)
            model = build(cond, args).to(device)
            lat = latency_ms_per_window(model, device)
            print(f"latency {CONDITION_LABELS[cond]}: {lat:.2f} ms/window")
            for seed in per_seed:
                per_seed[seed][cond]["latency_ms"] = lat
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for seed in per_seed:
            path = os.path.join(args.out_dir, f"seed{seed}", "metrics.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"run_key": run_key(args), "results": per_seed[seed]}, f, indent=2)

    summary, table = summarize(per_seed, args)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "environment": run_environment(args), "summary": summary,
                   "per_seed": per_seed}, f, indent=2)
    with open(os.path.join(args.out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nSaved {args.out_dir}/results.json and {args.out_dir}/results.md")


if __name__ == "__main__":
    main()
