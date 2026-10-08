"""
Splits, window adjacency and label-budget sampling for the low-label contrastive study
(results/sensor_contrastive/PROTOCOL.md, sections 1-3).

Partitions (inductive): pool = official training subjects except the validation subjects,
val = subjects 27-30, test = official UCI test subjects. Test windows are only loaded by
`load_test`, which the study calls once, from `src.lowlabel evaluate-test`.
"""
import hashlib
import glob
import os
from collections import defaultdict

import numpy as np

from src.dataset import DEFAULT_VAL_SUBJECTS, ChannelStandardizer, load_labels_and_subjects, load_signals

HOP = 64          # UCI HAR windows: 128 samples at 50 Hz with 50% overlap
WINDOW = 128
RATE_HZ = 50
BLOCK_LEN = 4     # windows per labeled block (6.4 s of signal)
MIN_PER_CLASS = 2


def raw_data_sha256(data_dir: str) -> str:
    """sha256 over every inertial-signal, label and subject file (relative path + bytes)."""
    h = hashlib.sha256()
    files = glob.glob(os.path.join(data_dir, "*", "Inertial Signals", "*.txt"))
    files += glob.glob(os.path.join(data_dir, "*", "y_*.txt")) + glob.glob(os.path.join(data_dir, "*", "subject_*.txt"))
    for f in sorted(files, key=lambda p: os.path.relpath(p, data_dir).replace("\\", "/")):
        h.update(os.path.relpath(f, data_dir).replace("\\", "/").encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def chain_ids(X: np.ndarray, subjects: np.ndarray) -> tuple:
    """
    Recover recording adjacency: window j follows window i when i's second half equals j's first
    half exactly (all channels) and both belong to the same subject. Returns (chain id, position
    within chain) per window; a chain is a maximal run of consecutive overlapping windows.
    Ambiguous successors or predecessors raise instead of guessing.
    """
    first = defaultdict(list)
    for j in range(len(X)):
        first[X[j, :HOP].tobytes()].append(j)
    nxt, prev = {}, {}
    for i in range(len(X)):
        cands = [j for j in first.get(X[i, WINDOW - HOP:].tobytes(), []) if j != i and subjects[j] == subjects[i]]
        if len(cands) > 1:
            raise ValueError(f"ambiguous successor for window {i}: {cands}")
        if cands:
            j = cands[0]
            if j in prev:
                raise ValueError(f"window {j} has two predecessors")
            nxt[i], prev[j] = j, i
    ids = np.full(len(X), -1, dtype=np.int64)
    pos = np.zeros(len(X), dtype=np.int64)
    cid = 0
    for i in range(len(X)):
        if i in prev:
            continue
        k, p = i, 0
        while True:
            ids[k], pos[k] = cid, p
            if k not in nxt:
                break
            k, p = nxt[k], p + 1
        cid += 1
    if (ids < 0).any():
        raise ValueError("cycle in window adjacency")
    return ids, pos


def blocks_for(indices: np.ndarray, chains: np.ndarray, pos: np.ndarray, block_len: int = BLOCK_LEN) -> list:
    """Tile each chain (restricted to `indices`) into consecutive non-overlapping blocks of window ids."""
    by_chain = defaultdict(list)
    for i in indices:
        by_chain[int(chains[i])].append(int(i))
    blocks = []
    for c in sorted(by_chain):
        members = sorted(by_chain[c], key=lambda i: pos[i])
        blocks += [members[k:k + block_len] for k in range(0, len(members), block_len)]
    return blocks


def class_orders(indices, y, subjects, chains, pos, seed: int, salt: int, block_len: int = BLOCK_LEN) -> dict:
    """
    For each class, one seeded ordering of that class's windows: blocks are shuffled within each
    subject, subjects are shuffled, and blocks are interleaved round-robin across subjects. Any
    prefix of the order is therefore a set of contiguous blocks spread over many subjects, and a
    budget is just a prefix length (so budgets are nested by construction).
    """
    rng = np.random.default_rng([seed, salt])
    orders = {}
    for c in sorted(set(int(v) for v in y[indices])):
        cls_idx = np.asarray([i for i in indices if y[i] == c])
        per_subj = defaultdict(list)
        for b in blocks_for(cls_idx, chains, pos, block_len):
            per_subj[int(subjects[b[0]])].append(b)
        subj_order = list(per_subj)
        rng.shuffle(subj_order)
        queues = []
        for s in subj_order:
            bl = per_subj[s]
            perm = rng.permutation(len(bl))
            queues.append([bl[k] for k in perm])
        order = []
        while any(queues):
            for q in queues:
                if q:
                    order += q.pop(0)
        orders[c] = order
    return orders


def quota(n_class: int, budget: float, min_per_class: int = MIN_PER_CLASS) -> int:
    return min(n_class, max(int(round(budget * n_class)), min_per_class))


def budget_subset(orders: dict, budget: float) -> np.ndarray:
    """Labeled window ids for a budget fraction: the first quota(n_c) windows of each class order."""
    chosen = [w for c, order in orders.items() for w in order[:quota(len(order), budget)]]
    return np.asarray(sorted(chosen), dtype=np.int64)


def unique_seconds(ids, chains, pos) -> float:
    """Distinct signal time covered by a set of windows (overlapping samples counted once)."""
    covered = set()
    for i in ids:
        start = int(pos[i]) * HOP
        covered.update((int(chains[i]), t) for t in range(start, start + WINDOW))
    return len(covered) / RATE_HZ


class StudyData:
    """Training-side data only: pool and validation windows (raw), labels, subjects, chains."""

    def __init__(self, data_dir: str, val_subjects=DEFAULT_VAL_SUBJECTS):
        X = load_signals(data_dir, "train")
        y, s = load_labels_and_subjects(data_dir, "train")
        self.data_dir = data_dir
        self.chains, self.pos = chain_ids(X, s)
        is_val = np.isin(s, val_subjects)
        self.pool_idx = np.flatnonzero(~is_val)
        self.val_idx = np.flatnonzero(is_val)
        assert not set(s[self.pool_idx]) & set(s[self.val_idx]), "pool/val subject overlap"
        self.X_raw, self.y, self.subjects = X, y, s
        self.scaler = ChannelStandardizer().fit(X[self.pool_idx])      # pool inputs only
        self.channel_std = X[self.pool_idx].std(axis=(0, 1)).astype(np.float32)

    def pool_inputs(self) -> np.ndarray:
        """Unlabeled pretraining inputs: raw pool windows, no labels."""
        return self.X_raw[self.pool_idx]

    def orders(self, seed: int) -> dict:
        return {
            "pool": class_orders(self.pool_idx, self.y, self.subjects, self.chains, self.pos, seed, salt=1),
            "val": class_orders(self.val_idx, self.y, self.subjects, self.chains, self.pos, seed, salt=2),
        }

    def subsets(self, seed: int, budget: float) -> dict:
        o = self.orders(seed)
        return {"train": budget_subset(o["pool"], budget), "val": budget_subset(o["val"], budget)}

    def describe(self, ids) -> dict:
        counts = defaultdict(lambda: [0] * 6)
        for i in ids:
            counts[int(self.subjects[i])][int(self.y[i])] += 1
        return {"windows": int(len(ids)), "unique_seconds": unique_seconds(ids, self.chains, self.pos),
                "per_class": np.bincount(self.y[ids], minlength=6).tolist(),
                "per_subject_class": {str(k): v for k, v in sorted(counts.items())}}


def load_test(data_dir: str):
    """Raw test windows, labels and subjects. Called only by the one-time test evaluation."""
    X = load_signals(data_dir, "test")
    y, s = load_labels_and_subjects(data_dir, "test")
    return X, y, s
