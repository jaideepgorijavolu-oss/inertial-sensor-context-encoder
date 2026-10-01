# Sensor Context Encoder

Can a frozen language model reason over raw inertial telemetry if the signal is injected
directly into its token-embedding space, with no text serialization? This repo projects
128×9 IMU windows (UCI HAR) into `HuggingFaceTB/SmolLM2-360M-Instruct` as a single soft
token and compares it against a dedicated CNN classifier, a parameter-matched no-LLM
ablation, and a shuffled-token negative control. See [TECHNICAL_NOTE.md](TECHNICAL_NOTE.md).

## Experimental conditions

| # | Condition | What it isolates |
| :- | :--- | :--- |
| 1 | Direct sensor classifier (1D-CNN + linear head) | Strong task-specific baseline |
| 2 | Context-embedding model (CNN → MLP projector → frozen SmolLM2 → head) | Sensor-as-token injection |
| 3 | Matched-capacity ablation (CNN → same projector → head, **no LLM**) | Whether the LLM adds anything beyond the extra trainable parameters |
| 4 | Condition 2 with sensor tokens permuted across the batch | Whether predictions actually depend on the sensor token |

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

# Full benchmark: all conditions, 3 seeds, equal 15-epoch budgets
python -m src.train

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
