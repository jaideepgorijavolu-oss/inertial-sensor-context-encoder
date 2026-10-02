# Sensor Context Encoder

Can a frozen language model reason over raw inertial telemetry if the signal is injected
directly into its token-embedding space, with no text serialization? This repo projects
128×9 IMU windows (UCI HAR) into `HuggingFaceTB/SmolLM2-360M-Instruct` as a single soft
token and compares it against a dedicated CNN classifier, a parameter-matched no-LLM
ablation, and a shuffled-token negative control. See [TECHNICAL_NOTE.md](TECHNICAL_NOTE.md).

## Experimental conditions

| Condition | What it isolates |
| :--- | :--- |
| Direct sensor classifier (1D-CNN + linear head) | Strong task-specific baseline |
| Matched-capacity ablation (CNN, projector, head, **no LLM**) | Whether the LLM adds anything beyond the extra trainable parameters |
| Context model, 1 sensor token (frozen SmolLM2) | Sensor-as-token injection |
| Context model, 8 temporal sensor tokens (frozen) | Whether exposing time structure to the LLM helps |
| Context model, 8 tokens + LoRA (r=8, q/v) | Whether adapting the LLM helps |
| Shuffled-token controls (one per context model) | Whether predictions actually depend on the sensor tokens |

## Results (3 seeds, RTX 5060 Laptop GPU)

| Condition | Test Macro-F1 | Latency |
| :--- | :--- | ---: |
| Direct CNN | **0.934 ± 0.006** | 0.44 ms |
| Matched, no LLM | 0.928 ± 0.010 | 0.45 ms |
| Frozen LLM, 1 token | 0.924 ± 0.006 | 25.0 ms |
| Frozen LLM, 8 tokens | 0.921 ± 0.011 | 26.0 ms |
| LLM + LoRA, 8 tokens | 0.916 ± 0.028 (0.931 / 0.932 / 0.883) | 31.8 ms |
| Shuffled controls | about 0.36 | — |

A frozen LLM classifies raw IMU embeddings at 0.92 macro-F1 and genuinely depends on them (shuffling drops it to 0.36), but adds no accuracy over a parameter-matched network without it; LoRA gives the best LLM runs on 2 of 3 seeds but is unstable under small-validation-set checkpoint selection. Full analysis in [TECHNICAL_NOTE.md](TECHNICAL_NOTE.md).

Protocol: subject-disjoint splits (val = subjects 27–30, test = official UCI test subjects),
per-channel standardization fit on the training subjects only, equal epoch budgets by default,
best-validation checkpoint selection, and mean ± std over multiple seeds.

## Setup

```bash
git clone https://github.com/jaideepgorijavolu-oss/inertial-sensor-context-encoder.git
cd inertial-sensor-context-encoder

python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python -m src.download_data        # fetches UCI HAR into data/UCI HAR Dataset
```

## Run

```bash
# Unit tests (no dataset or model download needed; uses a tiny random Llama)
python -m pytest tests -v

# Full benchmark: all 5 conditions, 3 seeds, equal 15-epoch budgets (~2 h on an RTX 5060 Laptop GPU)
# Results are saved after every condition; rerunning resumes and skips completed work.
python -m src.train

# Re-time every architecture on an idle GPU
python -m src.train --remeasure-latency

# Faster iteration examples
python -m src.train --seeds 42 --conditions direct matched
python -m src.train --context-epochs 5 --gradient-checkpointing   # lower memory for the LLM run
```

Outputs land in `artifacts/`:

- `results.md`: summary table (macro-F1 mean ± std, trainable params, latency per window)
- `results.json`: per-seed metrics, per-class F1, confusion matrices, training curves, normalization stats
- `seed*/best_<condition>.pt`: best-validation weights (trainable parameters only; the frozen LLM is not duplicated)

## Repository layout

```text
src/
  dataset.py        UCI HAR loading, subject-disjoint split, train-only standardization
  models.py         SensorEncoder, DirectClassifier, MatchedCapacityClassifier, ContextEmbeddingModel
  train.py          Multi-seed training/evaluation CLI, results table
  download_data.py  Dataset download + extraction
tests/              Shape, split, standardization, gradient-flow, and end-to-end CLI tests
TECHNICAL_NOTE.md   Method, protocol, and results discussion
```
