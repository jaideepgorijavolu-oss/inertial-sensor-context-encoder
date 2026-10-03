# Sensor Context Encoder: Injecting Raw IMU Signals into a Frozen LLM

[![CI](https://github.com/jaideepgorijavolu-oss/inertial-sensor-context-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/jaideepgorijavolu-oss/inertial-sensor-context-encoder/actions/workflows/ci.yml)

Can a language model read raw sensor data without turning it into text? This project encodes 128×9 inertial
windows (accelerometer + gyroscope, UCI HAR) with a 1D-CNN, projects them into the token-embedding space of
**SmolLM2-360M-Instruct** as soft tokens, and classifies human activity from the LLM's hidden state.

It is set up as a controlled study rather than a single model: 5 conditions × 3 seeds on subjects never seen in
training, a parameter-matched ablation that isolates the LLM's contribution, and shuffled-token negative controls that
test whether the LLM is actually using the sensor input.

**Highlights**

- **0.92 macro-F1 from a frozen 360M LLM** reading raw IMU embeddings directly, on unseen subjects
- **Proven sensor dependence:** shuffling sensor tokens across the batch drops every LLM variant to ≈0.36 (chance ≈ 0.17)
- **Rigorous comparison:** parameter-matched no-LLM ablation, frozen vs **LoRA-adapted** backbone, 1 vs 8 temporal sensor tokens, equal epoch budgets, 3 seeds
- **Bug found and fixed:** a `torch.no_grad()` around the frozen LLM had silently blocked gradients to the encoder; fixing it raised the LLM model from 0.513 to 0.924 macro-F1
- **Reproducible:** retraining reproduces identical test scores; results are written per condition and runs resume after interruption
- **25 tests** run in seconds on CPU with a tiny random Llama: gradient flow, frozen-weight and LoRA-only training, checkpoint snapshots, leakage-free splits, end-to-end CLI and resume

---

## Results

UCI HAR, subject-disjoint evaluation (test = the 9 official test subjects). 3 seeds × 15 epochs per condition,
NVIDIA RTX 5060 Laptop GPU. Latency = median single-window inference over 100 runs on an idle GPU.

| Condition | Test Macro-F1 (mean ± std) | Per seed (42 / 43 / 44) | Trainable params | Latency |
| :--- | :--- | :--- | ---: | ---: |
| Direct CNN classifier | **0.934 ± 0.006** | 0.937 / 0.927 / 0.937 | 146K | 0.44 ms |
| Matched capacity, no LLM | 0.928 ± 0.010 | 0.937 / 0.929 / 0.918 | 1.32M | 0.45 ms |
| Frozen LLM, 1 sensor token | 0.924 ± 0.006 | 0.921 / 0.930 / 0.920 | 1.32M | 25.0 ms |
| Frozen LLM, 8 sensor tokens | 0.921 ± 0.011 | 0.925 / 0.908 / 0.929 | 1.32M | 26.0 ms |
| LLM + LoRA (r=8), 8 tokens | 0.916 ± 0.028 | **0.931 / 0.932** / 0.883 | 2.14M | 31.8 ms |
| Shuffled-token controls | 0.359 – 0.362 | | | |

The LLM conditions additionally carry 361.8M frozen backbone parameters. Per-class F1, confusion matrices and
training curves for every run are in [`artifacts/results.json`](artifacts/results.json).

### Findings

1. **A frozen LLM can consume raw sensor embeddings.** With no text serialization, one soft token reaches 0.924 macro-F1, within about 1 point of a dedicated CNN.
2. **The predictions really come from the sensor.** Permuting sensor tokens across the batch cuts macro-F1 by 61% (0.924 → 0.362) with near-zero variance, so the LLM is not leaning on the prompt.
3. **No improvement from the frozen LLM was observed at matched capacity in this experiment.** The identical trainable stack without the LLM scores 0.928 vs 0.924, a gap within seed-to-seed spread.
4. **More sensor tokens do not help.** Eight temporal tokens match one pooled token (0.921 vs 0.924).
5. **LoRA: two strong seeds, but a lower mean and higher variance.** The adapted backbone produced the two best LLM runs (0.931, 0.932, beating the direct CNN on seed 43). On seed 44, the best-validation checkpoint (epoch 2) generalized poorly on Sitting vs Standing (test 0.883): with only 4 validation subjects, checkpoint selection is noisy, and the higher-capacity model is most exposed to it.
6. **Cost:** the LLM pathway is 57–72× slower per window and adds 1.4 GB of weights. For closed-set activity recognition a specialized encoder is the right tool; the LLM pathway earns its cost only when the task needs language (explanations, open-ended questions, reasoning over sensor context).

Full method and discussion: [TECHNICAL_NOTE.md](TECHNICAL_NOTE.md).

---

## Method

```text
IMU window [128 × 9] ──► 1D-CNN encoder (3 conv blocks, 256-d) ──► MLP projector (256 → 960)
                                                                         │
            "Classify the activity ... Sensor context:" [sensor token(s)] "Activity:"
                                                                         │
                                  SmolLM2-360M (frozen, optional LoRA on q/v) ──► last hidden state ──► linear head (6 classes)
```

| Condition | What it isolates |
| :--- | :--- |
| Direct CNN classifier | Strong task-specific baseline |
| Matched capacity, no LLM | Same encoder + projector + head without the LLM: is the LLM itself adding anything? |
| Frozen LLM, 1 token | Sensor-as-token injection, globally pooled |
| Frozen LLM, 8 tokens | Conv features pooled into 8 temporal segments: does exposing time structure help? |
| LLM + LoRA, 8 tokens | Rank-8 adapters on attention q/v (+819K params): does adapting the backbone help? |
| Shuffled controls | Each LLM model with sensor tokens permuted across the batch (mean of 5 permutations) |

**Protocol**
- **Leakage-free splits:** validation = training subjects 27–30, test = the official UCI test subjects. Disjointness is asserted at load time and tested.
- **Normalization:** per-channel standardization is fit on training subjects only.
- **Training:** AdamW with a cosine schedule and gradient clipping. Every condition gets the same 15-epoch budget, and the best-validation checkpoint is kept (as a true copy of the weights).
- **Frozen backbone:** gradients flow *through* the frozen LLM into the projector and encoder. Only the LoRA adapters, when enabled, update inside the backbone.

---

## Setup and reproduction

```bash
git clone https://github.com/jaideepgorijavolu-oss/inertial-sensor-context-encoder.git
cd inertial-sensor-context-encoder
python -m venv venv
source venv/bin/activate              # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt       # for NVIDIA GPUs install the CUDA build of torch first
python -m src.download_data           # UCI HAR → data/UCI HAR Dataset

python -m pytest tests -v             # 25 tests, CPU only, no downloads
python -m src.train                   # all 5 conditions × 3 seeds (≈2 h on an RTX 5060 Laptop GPU)
python -m src.train --remeasure-latency   # re-time each architecture on an idle GPU
python -m src.predict --condition direct --seed 42   # inference from a saved checkpoint
```

- **Resumable:** results are saved after every condition. Rerunning skips finished work, and adding a condition trains only that condition.
- **Faster runs:** use `--seeds 42 --conditions direct matched` for quick iteration.

Outputs in `artifacts/`:
- `results.md`: the summary table.
- `results.json`: per-seed metrics, per-class F1, confusion matrices, training curves, normalization stats.
- `seed*/metrics.json`: per-seed results used for resuming.
- `seed*/best_<condition>.pt`: trainable weights only, not committed.

## Repository layout

```text
src/
  dataset.py        UCI HAR loading, subject-disjoint split, train-only standardization
  models.py         SensorEncoder, DirectClassifier, MatchedCapacityClassifier, ContextEmbeddingModel, LoRA
  train.py          Multi-seed, multi-condition training/evaluation CLI with per-condition resume
  download_data.py  Dataset download and extraction
tests/              Gradient flow, LoRA, frozen backbone, splits, standardization, end-to-end CLI and resume
artifacts/          Final results (tables, per-seed metrics)
TECHNICAL_NOTE.md   Method, protocol, full results and discussion
```

## License

MIT, see [LICENSE](LICENSE).
