# Handoff: contrastive low-label study (phase one, revised after external review)

Branch `experiment/sensor-contrastive`. Its first version was fast-forward merged into `main` at `8bb2d17`
by the owner. This revision (review fixes) is on the branch; merging it is again the owner's step. Base of the
study: `main` @ `bebe5b2`.

## What changed

| Path | Change |
| --- | --- |
| `src/lowlabel_data.py` | Adjacency reconstruction, block tiling, nested stratified budget sampler, raw-data sha256 |
| `src/augment.py` | Scaling, jitter and small 3-D rotation that keep total = body + gravity |
| `src/ssl.py` | SimCLR NT-Xent, projection head, label-free pretraining, collapse statistics |
| `src/lowlabel.py` | CLI `stage-a`, `stage-b`, `evaluate-test` (one-time, locked), `report`; run-config and checkpoint-hash checks; fixed weighted kNN |
| `tests/test_lowlabel.py` | 10 tests: leakage boundaries, sampler, augmentation physics, loss vs reference, gradient flow, frozen encoders, end-to-end CLI with a test-load guard, float32 kNN exact match, refused reuse after settings changes or checkpoint tampering |
| `scripts/audit_legacy.py`, `scripts/recheck_knn_fix.py`, `scripts/replay_checkpoints.py` | Regenerate the audit, the kNN-fix check and the full checkpoint replay |
| `results/sensor_contrastive/` | AUDIT, PROTOCOL (+ Amendment 1), REPORT (+ Corrections), this file, run directories, recheck/replay JSON |
| `README.md`, `TECHNICAL_NOTE.md` | Study section, narrowed claims, fresh-run commands; legacy "statistically indistinguishable" softened |
| `requirements.txt` | matplotlib, psutil |

Legacy `src/train.py`, `src/models.py`, `src/dataset.py`, `src/predict.py` and `artifacts/` are unchanged.

## Checks

- `python -m pytest tests -q`: 43 passed (33 legacy + 10 new). CI runs the same suite, including a CPU smoke run
  of the whole study CLI on synthetic data.
- Checkpoint replay: all 153 selected models, re-evaluated with the current code on the study GPU, reproduce every
  published test prediction (`replay_cuda.json`).
- kNN fix: all 21 kNN selections and predictions unchanged on the GPU (`knn_fix_recheck.json`). One prediction
  differs on CPU because of cross-device float differences.
- Legacy direct CNN retrained: saved metrics and confusion matrices matched exactly. Weight equality was not checked.
- Skipped: LLM checkpoint re-evaluation (the weights behind the stored LLM results no longer exist; AUDIT D1).

## Reproduce

```bash
python -m src.download_data
python -m pytest tests -q
# regenerate tables and plots from the published predictions
python -m src.lowlabel report --run-dir results/sensor_contrastive/mvs
# download the published checkpoints, check their hashes, replay all 153 selected models
curl -L -O https://github.com/jaideepgorijavolu-oss/inertial-sensor-context-encoder/releases/download/sensor-contrastive-v1/sensor_contrastive_checkpoints_v1.tar.gz
tar -xzf sensor_contrastive_checkpoints_v1.tar.gz
sha256sum -c results/sensor_contrastive/checkpoints_sha256.txt
PYTHONPATH=. python scripts/replay_checkpoints.py      # writes replay_<device>.json; GPU replay matched exactly
# fresh end-to-end run: always a NEW run directory
D=results/sensor_contrastive/reproduction_01
python -m src.lowlabel stage-a --run-dir $D
python -m src.lowlabel stage-b --run-dir $D
python -m src.lowlabel evaluate-test --run-dir $D
python -m src.lowlabel report --run-dir $D
```

Expansion runs reuse stage-A choices with `--stage-a <dir>/stage_a.json --ssl-cache <dir>/ssl`, plus
`--budgets 0.05 0.25`, `--block-len 1`, or `--ssl-aug strong_norot --conditions sup_aug ssl_probe ssl_ft`, each in its own
new run directory. A full rerun takes under 15 minutes on an RTX 5060 Laptop GPU.

## Artifacts

- Per run directory: `stage_a.json`, `selections.json`, `split_manifest.json`, `test_metrics.json` (all predictions),
  `summary.json`, `results.md`, plot and logs, with provenance and checkpoint sha256 values.
- **Checkpoints:** all 165 `.pt` files (162 study + 3 legacy-reproduction; 87 MB compressed) are in the GitHub release
  [`sensor-contrastive-v1`](https://github.com/jaideepgorijavolu-oss/inertial-sensor-context-encoder/releases/tag/sensor-contrastive-v1). Their sha256 values are in
  `results/sensor_contrastive/checkpoints_sha256.txt`; they also match those recorded in `selections.json` and `ssl/*.json`.
- UCI HAR (CC BY 4.0) is not committed; `data_sha256` identifies the exact files.

## Known limitations

One dataset; 3 seeds; test set already seen in earlier work; near-arbitrary validation at 1 %; stage A tuned at 10 %
labels; the block-vs-single-window comparison is confounded by subject coverage.

## Proposed next step (needs approval)

HHAR as a second benchmark, testing "the rotation-dependent SSL gain holds under varied devices and placement".
Check its license, rates and channels first. This would be a separate benchmark, not transfer.

## Candidate resume wording

- Designed a leakage-controlled study of SimCLR pretraining for wearable activity recognition on unseen subjects
  (PyTorch). Recovered window adjacency to block-sample labels, locked the protocol and validation selections in git
  before a one-time test evaluation, and verified all 153 models replay to their published predictions.
- Fine-tuning from SimCLR gave the largest gains at 1 % labels (+0.023 to +0.047 macro-F1 over a from-scratch CNN in
  every seed), and a frozen SimCLR encoder with a linear probe reached 0.950 macro-F1 with all labels. An ablation
  showed both results depend on rotation augmentation.
