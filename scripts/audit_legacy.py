"""Legacy audit (results/sensor_contrastive/AUDIT.md): run from the repo root with PYTHONPATH=.

Part 1 recomputes aggregates and confusion-matrix metrics from artifacts/results.json.
Part 2 fingerprints the raw data and reconstructs window adjacency.
Re-evaluating legacy checkpoints: python scripts/audit_legacy.py --checkpoints direct matched
"""
import sys
import json, hashlib, os, glob, numpy as np
R = "artifacts"
res = json.load(open(f"{R}/results.json"))
print("config:", {k: res["config"][k] for k in res["config"]})
print("report_env:", res["report_environment"])
conds = ["direct","matched","context","context_mt","context_lora"]
def f1_from_cm(cm):
    cm=np.array(cm); tp=np.diag(cm); p=cm.sum(0); r=cm.sum(1)
    f=np.where(p+r>0, 2*tp/np.maximum(p+r,1),0); return f.mean(), tp.sum()/cm.sum()
for s in ["42","43","44"]:
    seedfile = json.load(open(f"{R}/seed{s}/metrics.json"))
    ps = res["per_seed"][s]
    print(f"\nseed {s}: seedfile schema={seedfile['schema_version']} run_key={seedfile['run_key']}")
    print("  seedfile results == results.json per_seed:", seedfile["results"] == ps)
    for c in conds:
        r = ps[c]; f1cm, acc = f1_from_cm(r["confusion_matrix"]); pv = r["provenance"]
        ck = f"{R}/seed{s}/best_{c}.pt"
        h = hashlib.sha256(open(ck,"rb").read()).hexdigest()[:16] if os.path.exists(ck) else None
        print(f"  {c:13s} f1={r['macro_f1']:.10f} cm_f1={f1cm:.10f} d={r['macro_f1']-f1cm:+.1e} acc_d={r['accuracy']-acc:+.1e} "
              f"ep={r['best_epoch']} val={r['best_val_macro_f1']:.4f} n_test={np.sum(r['confusion_matrix'])} "
              f"commit={str(pv.get('git_commit'))[:8]} dirty={pv.get('git_dirty')} at={pv.get('trained_at')} dev={pv.get('device')} lat_dev={r.get('latency_device')} "
              f"rev={str(pv.get('llm_revision'))[:8]} torch={pv.get('torch')} ckpt={h} mtime={os.path.getmtime(ck) if h else None}")
print("\nrecomputed summary (mean, sample std ddof=1) vs stored:")
for k,v in res["summary"].items():
    f = [res["per_seed"][s][k]["macro_f1"] for s in ["42","43","44"]]
    print(f"  {k:30s} mean={np.mean(f):.16f} stored={v['macro_f1_mean']:.16f} std={np.std(f,ddof=1):.6f} stored={v['macro_f1_std']:.6f}")
st = [res["per_seed"][s]["standardizer"] for s in ["42","43","44"]]
print("\nstandardizer identical across seeds:", st[0]==st[1]==st[2])

print("\n--- data ---")
import hashlib, glob, os, numpy as np
from collections import Counter, defaultdict
from src.dataset import load_signals, load_labels_and_subjects, SIGNAL_NAMES
D = "data/UCI HAR Dataset"
h = hashlib.sha256()
for f in sorted(glob.glob(f"{D}/*/Inertial Signals/*.txt") + glob.glob(f"{D}/*/[ys]*_*.txt")):
    h.update(os.path.relpath(f, D).replace("\\", "/").encode()); h.update(open(f, "rb").read())
print("sha256(all inertial+label+subject files) =", h.hexdigest())
zp = "data/UCI HAR Dataset.zip"
print("zip sha256 =", hashlib.sha256(open(zp, "rb").read()).hexdigest() if os.path.exists(zp) else None)
for split in ("train", "test"):
    X = load_signals(D, split); y, s = load_labels_and_subjects(D, split)
    print(f"\n[{split}] windows={len(y)} subjects={sorted(set(s.tolist()))}")
    print("  per-class:", dict(sorted(Counter(y.tolist()).items())))
    # physics check: total_acc == body_acc + gravity? gravity estimate = total-body, should be smooth/low-variance
    g = X[..., 0:3] - X[..., 3:6]
    print("  |total-body| mean norm %.3f g; within-window std of (total-body) %.4f vs body std %.4f" %
          (np.linalg.norm(g, axis=-1).mean(), g.std(axis=1).mean(), X[..., 3:6].std(axis=1).mean()))
    # adjacency: window j follows window i if X[i,64:] == X[j,:64] (all 9 channels), same subject
    first = defaultdict(list)
    for j in range(len(y)): first[X[j, :64].tobytes()].append(j)
    nxt = {}; amb = 0; cross_subj = 0; seq = 0
    for i in range(len(y)):
        c = [j for j in first.get(X[i, 64:].tobytes(), []) if j != i]
        if len(c) > 1: amb += 1
        if c:
            j = c[0]
            if s[j] != s[i]: cross_subj += 1; continue
            nxt[i] = j; seq += (j == i + 1)
    has_prev = set(nxt.values())
    heads = [i for i in range(len(y)) if i not in has_prev]
    chains = []
    for hd in heads:
        c = [hd]
        while c[-1] in nxt: c.append(nxt[c[-1]])
        chains.append(c)
    L = np.array([len(c) for c in chains])
    lab_change = sum(y[i] != y[j] for i, j in nxt.items())
    print(f"  links={len(nxt)} (next row: {seq}) ambiguous={amb} cross-subject={cross_subj} label-change links={lab_change}")
    print(f"  chains={len(chains)} covering={L.sum()} len min/median/mean/max={L.min()}/{np.median(L)}/{L.mean():.1f}/{L.max()} singletons={(L==1).sum()}")
    per_subj = Counter(s[c[0]] for c in chains)
    print("  chains per subject:", dict(sorted(per_subj.items())))

if "--checkpoints" in sys.argv:
    sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--checkpoints") + 1:]
    import json, sys, numpy as np, torch
    from src.predict import load_model
    from src.dataset import load_signals, load_labels_and_subjects
    from sklearn.metrics import confusion_matrix, f1_score
    res = json.load(open("artifacts/results.json"))
    X = load_signals("data/UCI HAR Dataset", "test"); y, _ = load_labels_and_subjects("data/UCI HAR Dataset", "test")
    for cond in sys.argv[1:]:
        for s in (42, 43, 44):
            m, mean, std = load_model("artifacts", cond, s, torch.device("cpu"))
            with torch.no_grad():
                xs = torch.tensor((X - mean) / (std + 1e-6))
                p = torch.cat([m(xs[i:i+256]).argmax(-1) for i in range(0, len(xs), 256)]).numpy()
            cm = confusion_matrix(y, p, labels=range(6)).tolist()
            print(cond, s, "f1_now=%.10f stored=%.10f cm_match=%s" % (f1_score(y, p, average="macro"), res["per_seed"][str(s)][cond]["macro_f1"], cm == res["per_seed"][str(s)][cond]["confusion_matrix"]), flush=True)
