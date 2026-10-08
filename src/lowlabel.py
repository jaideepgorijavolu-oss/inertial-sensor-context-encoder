"""
Low-label contrastive study runner (results/sensor_contrastive/PROTOCOL.md).

    python -m src.lowlabel stage-a        # validation-only tuning on seed 42, 10% budget
    python -m src.lowlabel stage-b        # every seed x budget x condition, selection on V_b only
    python -m src.lowlabel evaluate-test  # one-time test evaluation of the frozen selections
    python -m src.lowlabel report         # tables and plot from saved metrics (no evaluation)

Only `evaluate-test` loads test windows. It refuses to run twice for the same run directory.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from src.augment import AugConfig, augment
from src.dataset import ACTIVITY_NAMES, DEFAULT_DATA_DIR
from src.lowlabel_data import StudyData, load_test, raw_data_sha256
from src.models import DirectClassifier, SensorEncoder
from src.ssl import pretrain, standardize

CONDITIONS = ("sup", "sup_aug", "rand_probe", "ssl_probe", "ssl_knn", "ssl_ft")
DEFAULT_RUN_DIR = os.path.join("results", "sensor_contrastive", "mvs")
STAGE_A_SEED, STAGE_A_BUDGET = 42, 0.10
PROBE_C = (0.01, 0.1, 1.0, 10.0)
KNN_K = (1, 3, 5, 10, 20)
SSL_GRID = [(tau, aug) for tau in (0.1, 0.5) for aug in ("weak", "strong")]
SUP_GRID = [(lr, wd) for lr in (1e-3, 3e-4) for wd in (1e-4, 1e-2)]
FT_LR = (1e-3, 1e-4)


# ----------------------------------------------------------------------------- provenance

def environment(data_dir: str) -> dict:
    def git(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    import sklearn
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu"
    diff = git("diff", "HEAD") or ""
    return {
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": np.__version__, "scikit_learn": sklearn.__version__, "device": dev,
        "platform": platform.platform(), "data_sha256": raw_data_sha256(data_dir),
        "argv": sys.argv,
    }


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro", labels=range(6), zero_division=0))


class Meter:
    """Wall time and peak memory for one run (CUDA allocator peak on GPU, process RSS on CPU)."""

    def __init__(self, device):
        self.device = device

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            self.peak_mb = torch.cuda.max_memory_allocated() / 2**20
            self.method = "torch.cuda.max_memory_allocated"
        else:
            import psutil
            self.peak_mb = psutil.Process().memory_info().rss / 2**20
            self.method = "process RSS after run"
        self.seconds = time.perf_counter() - self.t0

    def record(self) -> dict:
        return {"wall_s": self.seconds, "peak_mem_mb": self.peak_mb, "mem_method": self.method}


# ----------------------------------------------------------------------------- context

class Ctx:
    """Training-side tensors on the device: raw pool/val windows plus standardization stats."""

    def __init__(self, data: StudyData, device):
        self.data, self.device = data, device
        self.X = torch.tensor(data.X_raw, dtype=torch.float32, device=device)   # train file only
        self.y = torch.tensor(data.y, dtype=torch.long, device=device)
        self.mean = torch.tensor(data.scaler.mean.reshape(-1), device=device)
        self.std = torch.tensor(data.scaler.std.reshape(-1), device=device)
        self.cstd = torch.tensor(data.channel_std, device=device)

    def norm(self, x):
        return standardize(x, self.mean, self.std)


@torch.no_grad()
def predict_logits(model, x_norm, bs=1024):
    model.eval()
    return torch.cat([model(x_norm[i:i + bs]) for i in range(0, len(x_norm), bs)])


@torch.no_grad()
def embed(encoder, x_norm, bs=1024) -> np.ndarray:
    encoder.eval()
    return torch.cat([encoder(x_norm[i:i + bs]) for i in range(0, len(x_norm), bs)]).cpu().numpy()


# ----------------------------------------------------------------------------- methods

def train_classifier(ctx: Ctx, tr, va, seed, lr, wd, aug: AugConfig = None, init_encoder: dict = None,
                     steps: int = 1400, batch_size: int = 64) -> tuple:
    """End-to-end SensorEncoder + linear head on labeled ids `tr`; V_b evaluated 20 times (every 70
    steps at the default 1400); best checkpoint kept, earliest on ties."""
    eval_every = max(1, steps // 20)
    set_seed(seed)
    model = DirectClassifier(SensorEncoder()).to(ctx.device)
    if init_encoder is not None:
        model.encoder.load_state_dict(init_encoder)
    tr_t = torch.as_tensor(tr, device=ctx.device)
    Xva, yva = ctx.norm(ctx.X[va]), ctx.data.y[va]
    bs = min(batch_size, len(tr))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(seed)
    gdev = torch.Generator(device=ctx.device).manual_seed(seed)
    best = (-1.0, 0, None)
    history, queue = [], []
    for step in range(1, steps + 1):
        if len(queue) < bs:
            queue += tr_t[torch.randperm(len(tr), generator=g).to(ctx.device)].tolist()
        batch, queue = queue[:bs], queue[bs:]
        xb = ctx.X[batch]
        if aug is not None:
            xb = augment(xb, aug, ctx.cstd, gdev)
        model.train()
        loss = F.cross_entropy(model(ctx.norm(xb)), ctx.y[batch])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0:
            f1 = macro_f1(yva, predict_logits(model, Xva).argmax(1).cpu().numpy())
            history.append({"step": step, "loss": loss.item(), "val_macro_f1": f1})
            if f1 > best[0]:
                best = (f1, step, {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
    return best[2], {"val_macro_f1": best[0], "best_step": best[1], "history": history}


def recalibrated_random_encoder(ctx: Ctx, seed: int) -> dict:
    """Untrained encoder whose BatchNorm running stats are re-estimated on unlabeled pool inputs."""
    set_seed(seed)
    enc = SensorEncoder().to(ctx.device)
    for m in enc.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None              # cumulative average over the pass
    enc.train()
    for m in enc.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    pool = ctx.norm(ctx.X[torch.as_tensor(ctx.data.pool_idx, device=ctx.device)])
    with torch.no_grad():
        for i in range(0, len(pool), 256):
            enc(pool[i:i + 256])
    return {k: v.detach().to("cpu", copy=True) for k, v in enc.state_dict().items()}


def encoder_from(state: dict, device) -> SensorEncoder:
    enc = SensorEncoder().to(device)
    enc.load_state_dict(state)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.eval()


def fit_probe(F_tr, y_tr, F_va, y_va) -> dict:
    """Logistic-regression probe; features standardized with labeled-train statistics; C on V_b."""
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6
    best = None
    for C in PROBE_C:                       # ordered most -> least regularized; ties keep earlier
        clf = LogisticRegression(C=C, max_iter=5000).fit((F_tr - mu) / sd, y_tr)
        f1 = macro_f1(y_va, clf.predict((F_va - mu) / sd))
        if best is None or f1 > best["val_macro_f1"]:
            best = {"val_macro_f1": f1, "C": C, "coef": clf.coef_, "intercept": clf.intercept_,
                    "classes": clf.classes_, "mu": mu, "sd": sd}
    return best


def probe_predict(p, F) -> np.ndarray:
    logits = ((F - p["mu"]) / p["sd"]) @ p["coef"].T + p["intercept"]
    return p["classes"][logits.argmax(1)]


def knn_predict(F_ref, y_ref, F_q, k: int, weighted: bool) -> np.ndarray:
    """Exact cosine kNN; weighted = 1/(cosine distance + 1e-8). Vote ties -> lowest class id."""
    a = F_ref / np.linalg.norm(F_ref, axis=1, keepdims=True).clip(1e-12)
    b = F_q / np.linalg.norm(F_q, axis=1, keepdims=True).clip(1e-12)
    sim = b @ a.T
    k = min(k, len(F_ref))
    nn_idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    votes = np.zeros((len(F_q), 6))
    for j in range(k):
        cols = nn_idx[:, j]
        w = 1.0 / (1.0 - sim[np.arange(len(F_q)), cols] + 1e-8) if weighted else 1.0
        np.add.at(votes, (np.arange(len(F_q)), y_ref[cols]), w)
    return votes.argmax(1)


def fit_knn(F_tr, y_tr, F_va, y_va) -> dict:
    best = None
    for k in KNN_K:
        if k > len(F_tr):
            continue
        for weighted in (False, True):
            f1 = macro_f1(y_va, knn_predict(F_tr, y_tr, F_va, k, weighted))
            if best is None or f1 > best["val_macro_f1"]:
                best = {"val_macro_f1": f1, "k": k, "weighted": weighted}
    return best


# ----------------------------------------------------------------------------- SSL cache

def ssl_encoder(ctx: Ctx, run_dir: str, seed: int, tau: float, aug: str, epochs: int, log=print) -> tuple:
    """Pretrained encoder for (seed, tau, aug), trained once and reused (same inputs and rule)."""
    path = os.path.join(run_dir, "ssl", f"seed{seed}_tau{tau}_{aug}_ep{epochs}.pt")
    meta_path = path.replace(".pt", ".json")
    if os.path.exists(path) and os.path.exists(meta_path):
        return torch.load(path), load(meta_path)
    with Meter(ctx.device) as m:
        state, hist = pretrain(ctx.data.pool_inputs(), ctx.data.scaler.mean, ctx.data.scaler.std,
                               ctx.data.channel_std, AugConfig.named(aug), tau, seed, ctx.device,
                               epochs=epochs, log=log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    meta = {"seed": seed, "tau": tau, "aug": aug, "epochs": epochs, "inputs": "pool windows, no labels",
            "n_inputs": int(len(ctx.data.pool_idx)), "checkpoint_rule": "final epoch",
            "history": hist, "cost": m.record(), "sha256": sha256_file(path)}
    dump(meta_path, meta)
    return state, meta


# ----------------------------------------------------------------------------- stages

def stage_a(args, ctx: Ctx, log=print) -> dict:
    seed, b = STAGE_A_SEED, STAGE_A_BUDGET
    sub = ctx.data.subsets(seed, b)
    tr, va = sub["train"], sub["val"]
    out = {"seed": seed, "budget": b, "n_train_labels": int(len(tr)), "n_val_labels": int(len(va)),
           "ssl": [], "sup": [], "sup_aug": [], "ssl_ft": []}
    for tau, aug in SSL_GRID:
        state, meta = ssl_encoder(ctx, args.run_dir, seed, tau, aug, args.ssl_epochs, log)
        enc = encoder_from(state, ctx.device)
        F_tr, F_va = embed(enc, ctx.norm(ctx.X[tr])), embed(enc, ctx.norm(ctx.X[va]))
        p = fit_probe(F_tr, ctx.data.y[tr], F_va, ctx.data.y[va])
        out["ssl"].append({"tau": tau, "aug": aug, "val_macro_f1": p["val_macro_f1"], "C": p["C"],
                           "final_loss": meta["history"][-1]["loss"],
                           "final_norm_std": meta["history"][-1]["norm_std"]})
        log(f"stage A ssl tau={tau} aug={aug}: probe val F1 {p['val_macro_f1']:.4f}")
    best_ssl = max(out["ssl"], key=lambda r: r["val_macro_f1"])        # first max in grid order
    aug_cfg = AugConfig.named(best_ssl["aug"])
    for key, aug in (("sup", None), ("sup_aug", aug_cfg)):
        for lr, wd in SUP_GRID:
            _, info = train_classifier(ctx, tr, va, seed, lr, wd, aug, steps=args.steps)
            out[key].append({"lr": lr, "wd": wd, "val_macro_f1": info["val_macro_f1"]})
            log(f"stage A {key} lr={lr} wd={wd}: val F1 {info['val_macro_f1']:.4f}")
    state, _ = ssl_encoder(ctx, args.run_dir, seed, best_ssl["tau"], best_ssl["aug"], args.ssl_epochs, log)
    for lr in FT_LR:
        _, info = train_classifier(ctx, tr, va, seed, lr, 1e-4, None, init_encoder=state, steps=args.steps)
        out["ssl_ft"].append({"lr": lr, "wd": 1e-4, "val_macro_f1": info["val_macro_f1"]})
        log(f"stage A ssl_ft lr={lr}: val F1 {info['val_macro_f1']:.4f}")
    pick = lambda rows: max(rows, key=lambda r: r["val_macro_f1"])
    out["chosen"] = {"ssl": {k: best_ssl[k] for k in ("tau", "aug")},
                     "sup": {k: pick(out["sup"])[k] for k in ("lr", "wd")},
                     "sup_aug": {k: pick(out["sup_aug"])[k] for k in ("lr", "wd")},
                     "ssl_ft": {k: pick(out["ssl_ft"])[k] for k in ("lr", "wd")}}
    out["environment"] = environment(args.data_dir)
    dump(os.path.join(args.run_dir, "stage_a.json"), out)
    return out


def stage_b(args, ctx: Ctx, log=print) -> dict:
    chosen = load(os.path.join(args.run_dir, "stage_a.json"))["chosen"]
    sel_path = os.path.join(args.run_dir, "selections.json")
    sel = load(sel_path) if os.path.exists(sel_path) else {"entries": {}, "chosen": chosen}
    if os.path.exists(os.path.join(args.run_dir, "test_metrics.json")):
        raise RuntimeError("test evaluation already ran for this run directory; selections are frozen")
    mdir = os.path.join(args.run_dir, "models")
    os.makedirs(mdir, exist_ok=True)
    manifest = {"pool_subjects": sorted(set(ctx.data.subjects[ctx.data.pool_idx].tolist())),
                "val_subjects": sorted(set(ctx.data.subjects[ctx.data.val_idx].tolist())),
                "pool_ids": ctx.data.pool_idx.tolist(), "val_ids": ctx.data.val_idx.tolist(),
                "chain_ids": ctx.data.chains.tolist(), "block_len": 4, "subsets": {}}
    aug_cfg = AugConfig.named(chosen["ssl"]["aug"])
    for seed in args.seeds:
        ssl_state, ssl_meta = ssl_encoder(ctx, args.run_dir, seed, chosen["ssl"]["tau"], chosen["ssl"]["aug"],
                                          args.ssl_epochs, log)
        rand_state = recalibrated_random_encoder(ctx, seed)
        for b in args.budgets:
            sub = ctx.data.subsets(seed, b)
            tr, va = sub["train"], sub["val"]
            manifest["subsets"][f"{seed}/{b}"] = {"train": tr.tolist(), "val": va.tolist(),
                                                  "train_desc": ctx.data.describe(tr), "val_desc": ctx.data.describe(va)}
            for cond in args.conditions:
                key = f"{seed}/{b}/{cond}"
                if key in sel["entries"]:
                    continue
                entry = {"seed": seed, "budget": b, "condition": cond, "n_train_labels": int(len(tr)),
                         "n_val_labels": int(len(va)), "ssl_pretraining": None}
                path = os.path.join(mdir, f"{seed}_{b}_{cond}.pt")
                with Meter(ctx.device) as m:
                    if cond in ("sup", "sup_aug", "ssl_ft"):
                        hp = chosen[cond]
                        state, info = train_classifier(
                            ctx, tr, va, seed, hp["lr"], hp["wd"], aug_cfg if cond == "sup_aug" else None,
                            init_encoder=ssl_state if cond == "ssl_ft" else None, steps=args.steps)
                        torch.save(state, path)
                        entry.update(trainable_params=146182, val_macro_f1=info["val_macro_f1"],
                                     best_step=info["best_step"], history=info["history"], hparams=hp)
                    else:
                        enc = encoder_from(rand_state if cond == "rand_probe" else ssl_state, ctx.device)
                        F_tr, F_va = embed(enc, ctx.norm(ctx.X[tr])), embed(enc, ctx.norm(ctx.X[va]))
                        if cond == "ssl_knn":
                            p = fit_knn(F_tr, ctx.data.y[tr], F_va, ctx.data.y[va])
                            torch.save({"encoder": enc.state_dict(), "ref_ids": tr, **p}, path)
                            entry.update(trainable_params=0, val_macro_f1=p["val_macro_f1"],
                                         hparams={"k": p["k"], "weighted": p["weighted"], "metric": "cosine"})
                        else:
                            p = fit_probe(F_tr, ctx.data.y[tr], F_va, ctx.data.y[va])
                            torch.save({"encoder": enc.state_dict(), **p}, path)
                            entry.update(trainable_params=int(p["coef"].size + p["intercept"].size),
                                         val_macro_f1=p["val_macro_f1"], hparams={"C": p["C"]})
                if cond.startswith("ssl"):
                    entry["ssl_pretraining"] = {"tau": ssl_meta["tau"], "aug": ssl_meta["aug"],
                                                "epochs": ssl_meta["epochs"], "sha256": ssl_meta["sha256"], "cost": ssl_meta["cost"],
                                                "shared_across": "budgets and ssl_* conditions of this seed"}
                entry.update(cost=m.record(), checkpoint=os.path.relpath(path, args.run_dir),
                             checkpoint_sha256=sha256_file(path))
                sel["entries"][key] = entry
                log(f"seed {seed} budget {b} {cond}: val F1 {entry['val_macro_f1']:.4f} (n_val={len(va)})")
                dump(sel_path, sel)
    sel["environment"] = environment(args.data_dir)
    dump(sel_path, sel)
    dump(os.path.join(args.run_dir, "split_manifest.json"), manifest)
    return sel


def evaluate_test(args, device, log=print) -> dict:
    """One-time test evaluation of every frozen selection."""
    out_path = os.path.join(args.run_dir, "test_metrics.json")
    if os.path.exists(out_path):
        raise RuntimeError(f"{out_path} exists: the test set was already evaluated for this run")
    sel = load(os.path.join(args.run_dir, "selections.json"))
    data = StudyData(args.data_dir)                    # standardizer and kNN references: pool only
    Xte, yte, ste = load_test(args.data_dir)
    mean = torch.tensor(data.scaler.mean.reshape(-1), device=device)
    std = torch.tensor(data.scaler.std.reshape(-1), device=device)
    Xn = standardize(torch.tensor(Xte, device=device), mean, std)
    Xpool = torch.tensor(data.X_raw, device=device)
    results = {"environment": environment(args.data_dir), "n_test": int(len(yte)),
               "test_subjects": sorted(set(ste.tolist())), "entries": {}}
    for key, e in sel["entries"].items():
        path = os.path.join(args.run_dir, e["checkpoint"])
        assert sha256_file(path) == e["checkpoint_sha256"], f"checkpoint changed since selection: {path}"
        ck = torch.load(path, weights_only=False)
        if e["condition"] in ("sup", "sup_aug", "ssl_ft"):
            model = DirectClassifier(SensorEncoder()).to(device)
            model.load_state_dict(ck)
            pred = predict_logits(model, Xn).argmax(1).cpu().numpy()
        else:
            enc = encoder_from(ck["encoder"], device)
            F_te = embed(enc, Xn)
            if e["condition"] == "ssl_knn":
                ref = np.asarray(ck["ref_ids"])
                F_ref = embed(enc, standardize(Xpool[torch.as_tensor(ref, device=device)], mean, std))
                pred = knn_predict(F_ref, data.y[ref], F_te, ck["k"], ck["weighted"])
            else:
                pred = probe_predict(ck, F_te)
        per_subject = {str(s): macro_f1(yte[ste == s], pred[ste == s]) for s in sorted(set(ste.tolist()))}
        results["entries"][key] = {
            "macro_f1": macro_f1(yte, pred), "accuracy": float((pred == yte).mean()),
            "per_class_f1": dict(zip(ACTIVITY_NAMES, map(float, f1_score(yte, pred, average=None, labels=range(6), zero_division=0)))),
            "confusion_matrix": confusion_matrix(yte, pred, labels=range(6)).tolist(),
            "per_subject_macro_f1": per_subject, "predictions": pred.tolist()}
        log(f"{key}: test macro-F1 {results['entries'][key]['macro_f1']:.4f}")
    dump(out_path, results)
    return results


# ----------------------------------------------------------------------------- report

LABELS = {"sup": "Supervised CNN", "sup_aug": "Supervised CNN + aug", "rand_probe": "Random encoder + probe",
          "ssl_probe": "SimCLR + linear probe", "ssl_knn": "SimCLR + kNN", "ssl_ft": "SimCLR + fine-tune"}
PAIRS = [("ssl_ft", "sup"), ("ssl_ft", "sup_aug"), ("ssl_probe", "sup"), ("ssl_ft", "ssl_probe")]


def report(args) -> str:
    sel = load(os.path.join(args.run_dir, "selections.json"))["entries"]
    test = load(os.path.join(args.run_dir, "test_metrics.json"))["entries"]
    seeds = sorted({e["seed"] for e in sel.values()})
    budgets = sorted({e["budget"] for e in sel.values()})
    conds = [c for c in CONDITIONS if any(e["condition"] == c for e in sel.values())]
    get = lambda s, b, c: test[f"{s}/{b}/{c}"]["macro_f1"]
    summary = {}
    lines = ["| Condition | " + " | ".join(f"{b:.0%} labels" for b in budgets) + " |",
             "| :--- |" + " ---: |" * len(budgets)]
    for c in conds:
        cells = []
        for b in budgets:
            v = [get(s, b, c) for s in seeds]
            summary[f"{b}/{c}"] = {"mean": float(np.mean(v)), "std_ddof1": float(np.std(v, ddof=1)), "per_seed": v}
            cells.append(f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}")
        lines.append(f"| {LABELS[c]} | " + " | ".join(cells) + " |")
    lines += ["", "Paired differences in test macro-F1 (same seed, split and labels), mean ± std over seeds, per seed:", "",
              "| Pair | " + " | ".join(f"{b:.0%}" for b in budgets) + " |", "| :--- |" + " ---: |" * len(budgets)]
    for a, c in PAIRS:
        if a not in conds or c not in conds:
            continue
        cells = []
        for b in budgets:
            d = [get(s, b, a) - get(s, b, c) for s in seeds]
            summary[f"{b}/{a}-{c}"] = {"mean": float(np.mean(d)), "std_ddof1": float(np.std(d, ddof=1)), "per_seed": d}
            cells.append(f"{np.mean(d):+.3f} ± {np.std(d, ddof=1):.3f} ({' / '.join(f'{x:+.3f}' for x in d)})")
        lines.append(f"| {LABELS[a]} − {LABELS[c]} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    dump(os.path.join(args.run_dir, "summary.json"), summary)
    with open(os.path.join(args.run_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(f"Seeds {seeds}; test = 9 official UCI test subjects; std = sample std (ddof=1) over seeds.\n\n{table}\n")
    plot(args.run_dir, test, seeds, budgets, conds)
    print(table)
    return table


def plot(run_dir, test, seeds, budgets, conds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"sup": "#4E79A7", "sup_aug": "#A0CBE8", "rand_probe": "#BAB0AC",
              "ssl_probe": "#F28E2B", "ssl_knn": "#FFBE7D", "ssl_ft": "#E15759"}
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x = np.array(budgets) * 100
    for c in conds:
        v = np.array([[test[f"{s}/{b}/{c}"]["macro_f1"] for b in budgets] for s in seeds])
        ax.plot(x, v.mean(0), marker="o", lw=2, color=colors[c], label=LABELS[c])
        for row in v:
            ax.scatter(x, row, s=10, color=colors[c], alpha=0.5, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{b:g}%" for b in x])
    ax.set_xlabel("Labeled training windows (% of 5,867, block-sampled)")
    ax.set_ylabel("Test macro-F1 (9 unseen subjects)")
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("UCI HAR: macro-F1 vs label budget (mean line, dots = seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "f1_vs_budget.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("stage-a", "stage-b", "evaluate-test", "report"))
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.10, 1.0])
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--ssl-epochs", type=int, default=200)
    p.add_argument("--steps", type=int, default=1400)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.command == "report":
        return report(args)
    if args.command == "evaluate-test":
        return evaluate_test(args, device)
    ctx = Ctx(StudyData(args.data_dir), device)
    print(f"device {device} | pool {len(ctx.data.pool_idx)} | val {len(ctx.data.val_idx)}", flush=True)
    return stage_a(args, ctx) if args.command == "stage-a" else stage_b(args, ctx)


if __name__ == "__main__":
    main()
