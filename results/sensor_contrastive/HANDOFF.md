# Handoff: contrastive low-label study (phase one)

Branch `experiment/sensor-contrastive` on `origin` (pushed; the remote commit is checked with `git ls-remote` after
every push). Not merged; merging and publishing are the owner's call. Base: `main` @ `bebe5b2`.

## What changed

| Path | Change |
| --- | --- |
| `src/lowlabel_data.py` | New: adjacency reconstruction, block tiling, nested stratified budget sampler, raw-data sha256 |
| `src/augment.py` | New: scaling, jitter and small 3-D rotation that keep total = body + gravity |
| `src/ssl.py` | New: SimCLR NT-Xent, projection head, label-free pretraining, collapse statistics |
| `src/lowlabel.py` | New CLI: `stage-a`, `stage-b`, `evaluate-test` (one-time, locked), `report` |
| `tests/test_lowlabel.py` | 8 new tests: leakage boundaries, sampler, augmentation physics, loss vs reference, gradient flow, frozen encoders, kNN, end-to-end CLI with a test-load guard |
| `scripts/audit_legacy.py` | Regenerates the AUDIT.md numbers |
| `results/sensor_contrastive/` | AUDIT, PROTOCOL (+ Amendment 1), REPORT, this file, run directories |
| `README.md`, `TECHNICAL_NOTE.md` | New README section; "statistically indistinguishable" softened (AUDIT D7) |
| `requirements.txt` | Added matplotlib and psutil |

The legacy `src/train.py`, `src/models.py`, `src/dataset.py` and `artifacts/` are unchanged.

## Checks

- `python -m pytest tests -q`: 41 passed (33 legacy + 8 new), locally and in GitHub Actions CI on every push. The
  CI run includes a CPU smoke run of the full study CLI on synthetic data.
- Legacy direct CNN retrained: per-seed confusion matrices are bit-identical to `artifacts/results.json`.
- Skipped: LLM checkpoint re-evaluation (AUDIT D1b, owner's decision) and LLM retraining.

## Reproduce

```bash
python -m src.download_data
python -m pytest tests -q
python -m src.lowlabel stage-a
python -m src.lowlabel stage-b
python -m src.lowlabel evaluate-test          # once; refuses to rerun
python -m src.lowlabel report
# Amendment 1 (reuses stage-A choices and SSL encoders)
R=results/sensor_contrastive; C="--stage-a $R/mvs/stage_a.json --ssl-cache $R/mvs/ssl"
python -m src.lowlabel stage-b --run-dir $R/budgets_5_25 $C --budgets 0.05 0.25
python -m src.lowlabel stage-b --run-dir $R/iid $C --block-len 1 --budgets 0.01 0.1
python -m src.lowlabel stage-b --run-dir $R/ablation_norot $C --ssl-aug strong_norot --conditions sup_aug ssl_probe ssl_ft
# then evaluate-test and report with the same --run-dir for each
python -m src.train --conditions direct --no-resume --out-dir $R/legacy_repro    # legacy reproduction
```

On an RTX 5060 Laptop GPU every stage takes minutes; a full rerun is under 15 minutes.

## Artifacts

- Per run directory: `stage_a.json`, `selections.json` (choices, validation scores, histories, cost, checkpoint
  sha256), `split_manifest.json` (subject and window IDs, chains, every labeled subset with per-subject × class
  counts), `test_metrics.json` (per-class F1, confusion matrices, per-subject F1, all predictions),
  `summary.json`, `results.md`, and logs. Each JSON records commit, dirty flag, diff hash, versions, device and
  data sha256.
- **Checkpoints are local only** (gitignored `*.pt`, about 40 MB, under `results/sensor_contrastive/*/models` and
  `mvs/ssl`). Their sha256 values are in `selections.json` and `ssl/*.json`, so a later upload can be verified.
  Attach them to a GitHub release if reviewers need them; that is not done, as publishing was not authorized.
- UCI HAR (CC BY 4.0) is not committed; `data_sha256` identifies the exact files used.

## Known limitations

One dataset, 3 seeds, test set already seen in earlier work, near-arbitrary validation at 1 %, and stage A tuned at
10 % labels. Details are in REPORT.md under "Limitations" and "What the results do not support".

## Proposed next step (needs approval, not started)

A second dataset only if it tests a defined claim. Candidate: **HHAR** (Stisen et al., 2015; phones and watches,
9 users, several device models). Claim: "the rotation-dependent SSL gain holds when sensor placement and device
vary." HHAR varies placement more than UCI HAR, so the ±30° rotation choice is directly at stake. License, size,
sampling rates and channel compatibility (no separate body/gravity split) must be checked before download.
This would be a separate benchmark, not cross-dataset transfer.

## Candidate wording

README: done (section "Follow-up study").

Resume:
- Designed a leakage-controlled study of SimCLR pretraining for wearable activity recognition on unseen subjects
  (PyTorch). Recovered window adjacency to sample labels as contiguous blocks, and locked the protocol and
  validation selections in git before a one-time test evaluation.
- Found that pretraining helps only at 1 % labels (+0.023 to +0.047 macro-F1 over a from-scratch CNN in every
  seed) and that a frozen encoder with a 1.5K-parameter linear probe reaches 0.950 macro-F1 with all labels.
  An augmentation ablation traced both gains to rotation invariance.
