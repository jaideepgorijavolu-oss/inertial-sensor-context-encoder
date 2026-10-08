# Audit: inertial-sensor-context-encoder before the contrastive study

Date: 2026-10-07. Auditor: Claude Code (Opus 5.5), for Jaideep Gorijavolu.
Checkout: `C:\Users\Jai Gori\Downloads\inertial-sensor-context-encoder`, `main` = `origin/main` =
`bebe5b204226108402b869b2e802957a6b7ad280`, clean tree (verified with `git ls-remote`, no fetch).
A second, older clone at `C:\Users\Jai Gori\inertial-sensor-context-encoder` (`0c3b961`, one untracked
`.code-workspace` file) was not touched.

Evidence levels used below:

- **Inspected**: code or files read.
- **Ran**: existing checks executed.
- **Recomputed**: numbers derived again from stored artifacts or checkpoints.
- **Reproduced training**: nothing yet. No model was retrained during this audit.

## 1. Build, run, structure

| Item | Finding |
| --- | --- |
| Environment | `venv/`: Python 3.14.6, torch 2.14.1+cu130, numpy 2.5.3, scikit-learn 1.9.1, transformers 5.18.0, peft 0.21.2. `requirements.txt` uses unpinned `>=` floors. CI uses Python 3.12 and CPU torch. |
| GPU | **`torch.cuda.is_available()` is False** in this session (also with the sandbox disabled), and `nvidia-smi` reports insufficient permissions. Legacy runs were on an RTX 5060 Laptop GPU. |
| Data | `data/UCI HAR Dataset/` present (gitignored). Zip sha256 `2045e435…bfeb0`. sha256 over all inertial, label and subject files: `d59f95b8…600e9` (script: section 7). |
| Code | `src/dataset.py` (loading, subject split, `ChannelStandardizer`), `src/models.py` (`SensorEncoder`, `DirectClassifier`, `MatchedCapacityClassifier`, `ContextEmbeddingModel`, LoRA), `src/train.py` (multi-seed, multi-condition CLI with resume), `src/predict.py`, `src/download_data.py`. |
| Tests (ran) | `HF_HUB_OFFLINE=1 python -m pytest tests -q` → **33 passed** in 30 s (CPU). |
| Commands | `python -m src.download_data`; `python -m pytest tests -v`; `python -m src.train [--seeds … --conditions … --no-resume --remeasure-latency]`; `python -m src.predict --condition direct --seed 42`. |

**Reusable encoder.** `SensorEncoder` has 3 Conv1d blocks (k 7/5/3, stride 2, 64/128/256 channels),
each with BatchNorm, ReLU and Dropout 0.1 (no dropout after the last block), followed by global average pooling.
It outputs a 256-d embedding and has 144,640 parameters; `DirectClassifier` adds a 1,542-parameter linear head
(146,182 total, matching the stored count).

## 2. Current split (inspected and recomputed)

| Partition | Subjects | Windows | Per class (walk, up, down, sit, stand, lay) |
| --- | --- | ---: | --- |
| Train pool | 1 3 5 6 7 8 11 14 15 16 17 19 21 22 23 25 26 | 5867 | 997 857 786 1022 1091 1114 |
| Validation | 27 28 29 30 | 1485 | 229 216 200 264 283 293 |
| Test (official UCI) | 2 4 9 10 12 13 18 20 24 | 2947 | 496 471 420 491 532 537 |

Subject disjointness is asserted in `split_by_subject` and `get_dataloaders`, and tested in `tests/test_splits.py`.
The standardizer is fit on the training pool only and is identical across seeds (the data is deterministic).

**Window adjacency (recomputed).** Within each split, window *j* follows window *i* when
`X[i, 64:] == X[j, :64]` on all 9 channels (exact equality). Train: 6752 links. Every link is to the next row,
with 0 ambiguous matches, 0 cross-subject links and 0 links across a label change. That gives 600 chains
(about 28 per subject; length median 12, max 29, one singleton). Test: 257 chains. **Each chain is one
single-activity bout**, so contiguous blocks can be sampled exactly. Subject-disjoint partitions already
guarantee that no overlapping windows cross train, validation and test.

**Channel physics (recomputed).** `total_acc - body_acc` has a mean norm of 1.02 g and a within-window std
of 0.008 g, compared with 0.096 g for body_acc. So total = body + a slowly varying gravity estimate, as the UCI
documentation describes. Augmentations must preserve this relationship (see PROTOCOL.md).

## 3. Verified result-source table

All values come from `artifacts/results.json`, `summary` and `per_seed`. Means and sample std (ddof = 1)
were recomputed from the per-seed records and **match the stored summary to full float precision** for all
11 keys. Each per-seed macro-F1 and accuracy was also recomputed from its stored confusion matrix: all
15 match exactly. `artifacts/seed4{2,3,4}/metrics.json` equal the `per_seed` blocks byte-for-byte after
JSON parsing.

| Key | Architecture | Mean macro-F1 | Sample std | Per seed 42 / 43 / 44 | Best epoch (val) |
| --- | --- | ---: | ---: | --- | --- |
| `direct` | SensorEncoder + linear | 0.9336525666983485 | 0.005573 | 0.93682 / 0.92722 / 0.93692 | 10 / 4 / 7 |
| `matched` | encoder + MLP 256→960→960 + linear, no LLM | 0.9249961857110409 | 0.013206 | 0.91129 / 0.93764 / 0.92606 | 2 / 10 / 5 |
| `context` | encoder + projector → 1 soft token, frozen SmolLM2-360M | 0.9333020071954223 | 0.003338 | 0.93544 / 0.93501 / 0.92946 | 13 / 13 / 5 |
| `context_mt` | as above, 8 temporal tokens | 0.9205057404955915 | 0.010785 | 0.92468 / 0.90826 / 0.92858 | 5 / 2 / 5 |
| `context_lora` | 8 tokens + LoRA r = 8 on q/v | 0.9337494672006138 | 0.001899 | 0.93273 / 0.93594 / 0.93258 | 13 / 10 / 5 |

These are the same values as the brief's historical table. Shared protocol for all rows: validation subjects
27–30, checkpoint = best validation macro-F1 over 15 epochs, standardization on, batch 64, lr 1e-3
(context 5e-4), AdamW, cosine schedule. Provenance on every row: commit `e9837dee`, `git_dirty: true`,
trained 2026-10-04 03:40–10:37 UTC on the RTX 5060 Laptop GPU, LLM revision `a10cc151…`.

**Where "0.924" comes from.** No stored context result is 0.924. The closest is `matched`
(0.92499…, which truncates to 0.924 and rounds to 0.925). The stored `context` value is 0.9333.

## 4. Discrepancies

| # | Severity | Finding | Evidence |
| --- | --- | --- | --- |
| D1 | **High (provenance)** | **The checkpoints on disk are not the checkpoints behind the stored metrics.** `results.json` was produced with `out_dir: artifacts/reproduction`, and that directory no longer exists. The `artifacts/seed*/best_*.pt` files are dated 2026-10-01 23:45 to 10-02 02:08 UTC, which is before the 10-04 training timestamps. Re-evaluating them: `direct` reproduces the stored confusion matrices exactly for all 3 seeds, but `matched` does not (0.9368 / 0.9293 / 0.9176 vs stored 0.9113 / 0.9376 / 0.9261). LLM checkpoints: see D1b. This is consistent with the README note that only the conditions trained first in their runs were unaffected by the batch-order fix. | ckpt_check output, section 7 |
| D1b | High | LLM-condition recomputation from checkpoints was **not completed**: the CPU job died on a shell resource error. The owner chose not to pursue the legacy checkpoints further, so the LLM results stand as "stored records verified internally only". | task log, 2026-10-07 |
| D2 | Medium | The training commit is dirty. Committed code at `e9837de` equals HEAD for `src/`, `tests/`, requirements and CI, but the uncommitted diff at training time was not recorded, so the exact training source can't be proven. | `git diff --stat e9837de bebe5b2 -- src tests …` is empty |
| D3 | Medium | The dataset fingerprint (`dataset_labels_sha256`) hashes only `y_*.txt` and `subject_*.txt`, not the inertial signals. | `run_environment()` |
| D4 | Low (latent) | `--remeasure-latency` overwrites `latency_device`, and the "trained on" header in `results.md` is built from `latency_device`. Remeasuring on a different machine would therefore mislabel the training device in the table. `provenance.device` remains correct. This is not triggered in the current artifacts (both say RTX 5060). | `train.py:315, 475` |
| D5 | Low | The resume cache key checks schema version and hyperparameters, but not code commit or data hash. A cached result from older code can be silently reused; its provenance is kept, so this is detectable. | `load_reusable()` |
| D6 | Low | GPU determinism: `cudnn.deterministic` is set, but `torch.use_deterministic_algorithms` is not. Retraining may not be bitwise identical. | `set_seed()` |
| D7 | Scientific wording | The README and TECHNICAL_NOTE say "statistically indistinguishable" from 3 seeds with no test. The brief disallows equivalence claims from 3 seeds. The legacy result itself is not changed; only the wording is flagged. | README Findings 1 |
| D8 | Disclosure | The test set was evaluated for every condition and every run in earlier work, and model and design choices were made after seeing test numbers. The new study must disclose this; a locked protocol limits, but does not undo, that history. | git history, README |

**No discrepancy found** in: subject disjointness, standardizer fit boundary (training pool only),
standardizer restoration in `predict.py` (same eps, per-seed stats), checkpoint copies (`trainable_state_dict`
copies tensors), condition-order independence (per-condition `set_seed` and DataLoader reseed; the direct
checkpoints reproduce exactly), or aggregate arithmetic.

**Does any discrepancy change the legacy scientific conclusions?** No metric is contradicted by its own
stored confusion matrices. D1 means the stored LLM and matched numbers can't currently be re-derived from
weights. They rest on stored per-seed records only. That limits verification, not the stated conclusions.

## 5. What this audit did and did not do

- Inspected: all of `src/`, `tests/` (by name and by running them), CI, README, TECHNICAL_NOTE, artifacts,
  and git history and branches. All non-main branches are squash-merged ancestors with no unmerged code.
- Ran: the existing test suite (33 passed).
- Recomputed: all aggregates; per-seed F1 and accuracy from confusion matrices; test predictions from the
  `direct` and `matched` checkpoints (and the LLM checkpoints, D1b); split counts; data fingerprints; adjacency.
- Not done: any retraining, and any GPU measurement (no CUDA available).

## 6. Proposed legacy reproduction (in the approved study, before new experiments)

- Retrain `direct` for seeds 42–44 under the original protocol (`--conditions direct --no-resume --out-dir
  results/sensor_contrastive/legacy_repro`). This is cheap. Compare against the stored per-seed values and
  confusion matrices, and report any exact match or GPU nondeterminism.
- LLM conditions: no retraining unless you ask for it (about 6–8 GPU hours). Report them as "stored records
  verified internally; matching checkpoints not retained" (D1).

## 7. Audit scripts

Scratch scripts (not committed): `audit.py` (aggregates, confusion-matrix recomputation, provenance),
`ckpt_check.py` (re-evaluates checkpoints on the test split), `data_audit.py` (fingerprints, class counts,
adjacency, gravity check). They will be committed as `scripts/audit_legacy.py` with the study so these
numbers can be regenerated.
