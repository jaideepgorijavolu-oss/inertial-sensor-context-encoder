# Sensor Context Encoder: Injecting Raw IMU Signals into a Frozen LLM

[![CI](https://github.com/jaideepgorijavolu-oss/inertial-sensor-context-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/jaideepgorijavolu-oss/inertial-sensor-context-encoder/actions/workflows/ci.yml)

Can a language model read raw sensor data without turning it into text? This project encodes 128×9 inertial
windows (accelerometer + gyroscope, UCI HAR) with a 1D-CNN, projects them into the token-embedding space of
**SmolLM2-360M-Instruct** as soft tokens, and classifies human activity from the LLM's hidden state.

It is set up as a controlled study rather than a single model: 5 conditions × 3 seeds on subjects never seen in
training, a parameter-matched ablation that isolates the LLM's contribution, and shuffled-token negative controls that
test whether the LLM is actually using the sensor input.

**Highlights**

- **0.933 macro-F1 from a frozen 360M LLM** reading raw IMU embeddings directly, on unseen subjects, matching a dedicated CNN (0.934); LoRA reaches 0.934 ± 0.002
- **Proven sensor dependence:** a global shuffle that pairs every test label with an unrelated window drops every LLM model to chance (0.16), and zeroed sensor tokens to 0.05
- **Rigorous comparison:** parameter-matched no-LLM ablation, frozen vs **LoRA-adapted** backbone, 1 vs 8 temporal sensor tokens, equal epoch budgets, 3 seeds
- **Bug found and fixed:** a `torch.no_grad()` around the frozen LLM had silently blocked gradients to the encoder; fixing it raised the LLM model from 0.513 to over 0.92 macro-F1
- **Reproducible runs:** per-condition seeding independent of run order, versioned result cache with per-condition provenance (commit, environment, model revision), safe resume after interruption
- **33 tests** run in seconds on CPU with a tiny random Llama: gradient flow, frozen-weight and LoRA-only training, checkpoint snapshots, leakage-free splits, end-to-end CLI and resume

---

## Results

UCI HAR, subject-disjoint evaluation (test = the 9 official test subjects). 3 seeds × 15 epochs per condition, NVIDIA RTX 5060 Laptop GPU, full rerun with the current code (`python -m src.train --no-resume`). Latency = median single-window inference measured at the end of each training run. Controls are mean over 3 seeds; the global shuffle averages 5 seeded permutations per seed.

| Condition | Test Macro-F1 (mean ± std) | Per seed (42 / 43 / 44) | Trainable params | Latency |
| :--- | :--- | :--- | ---: | ---: |
| Direct CNN classifier | **0.934 ± 0.006** | 0.937 / 0.927 / 0.937 | 146K | 0.49 ms |
| Matched capacity, no LLM | 0.925 ± 0.013 | 0.911 / 0.938 / 0.926 | 1.32M | 0.89 ms |
| Frozen LLM, 1 sensor token | 0.933 ± 0.003 | 0.935 / 0.935 / 0.929 | 1.32M | 29.0 ms |
| Frozen LLM, 8 sensor tokens | 0.921 ± 0.011 | 0.925 / 0.908 / 0.929 | 1.32M | 29.8 ms |
| LLM + LoRA (r=8), 8 tokens | **0.934 ± 0.002** | 0.933 / 0.936 / 0.933 | 2.14M | 30.1 ms |
| Global-shuffle control (all LLM models) | 0.163 – 0.164 | | | |
| Zero-sensor control (all LLM models) | 0.048 – 0.050 | | | |

The LLM conditions additionally carry 361.8M frozen backbone parameters. Per-class F1, confusion matrices, training curves and per-condition provenance are in [`artifacts/results.json`](artifacts/results.json).

### Findings

1. **A frozen LLM reading raw sensor embeddings matches a dedicated CNN.** One soft token into frozen SmolLM2-360M reaches 0.933 ± 0.003 macro-F1, statistically indistinguishable from the direct CNN (0.934 ± 0.006).
2. **Predictions depend on the sensor input.** Pairing every test label with an unrelated window (global shuffle) drops every LLM model to 0.163–0.164 macro-F1, chance level for 6 classes; replacing the sensor tokens with zeros gives 0.048–0.050 (a single constant prediction). The prompt alone carries no class information.
3. **LoRA gives the most stable LLM result.** Rank-8 adapters reach 0.934 ± 0.002, tying the CNN with the lowest seed-to-seed variance of any condition.
4. **No clear gain from the LLM at matched capacity.** The frozen-LLM and LoRA models score about 0.8 points above the parameter-matched network without the LLM (0.925 ± 0.013), but the gap is within its seed-to-seed spread across 3 seeds.
5. **More sensor tokens do not help.** Eight temporal tokens (0.921 ± 0.011) score below one pooled token.
6. **Cost:** the LLM pathway is about 60× slower per window (29–30 ms vs 0.49 ms) and adds 1.4 GB of weights. For closed-set activity recognition a specialized encoder is the efficient choice; the LLM pathway earns its cost when the task needs language (explanations, open-ended questions, reasoning over sensor context).

> An earlier version of these results (before the per-condition batch-order fix and the global-shuffle control) reported a batch-local shuffle control of about 0.36 and a LoRA seed at 0.883. With the fixes, the conditions that trained first in their runs (direct CNN, 8-token frozen) reproduced exactly; the others changed as listed above, and the LoRA outlier did not recur.

Full method and discussion: [TECHNICAL_NOTE.md](TECHNICAL_NOTE.md).

---

## Follow-up study: contrastive pretraining under limited labels

Does SimCLR-style pretraining of the same 1D-CNN on unlabeled training-subject windows help when labels are scarce?
Inductive protocol locked before any run: test subjects never enter pretraining, normalization, selection or kNN references;
labels are sampled as contiguous blocks recovered from the 50% window overlap; validation labels scale with the budget.
Validation selections were committed before the one-time test evaluation.

| Condition (test macro-F1, 3 seeds) | 1% labels (59) | 10% (587) | 100% (5,867) |
| :--- | ---: | ---: | ---: |
| Supervised CNN | 0.800 ± 0.014 | 0.913 ± 0.008 | 0.926 ± 0.011 |
| Supervised CNN + same augmentations | 0.830 ± 0.019 | 0.900 ± 0.023 | 0.906 ± 0.005 |
| SimCLR, frozen + linear probe (1.5K params) | 0.778 ± 0.011 | 0.899 ± 0.006 | **0.950 ± 0.004** |
| SimCLR + full fine-tune | **0.847 ± 0.021** | 0.915 ± 0.012 | 0.932 ± 0.008 |

- At 1% labels, pretraining + fine-tuning beats the from-scratch CNN in all 3 seeds (+0.047), but most of that gain is
  matched by training the CNN with the same augmentations (+0.017, mixed across seeds).
- With all labels, a frozen SimCLR encoder + linear probe is the best model in either study (0.950, +0.024 over the CNN
  in every seed), almost entirely from the hard Sitting/Standing pair.
- The legacy direct CNN was retrained and reproduced bit-exactly.

Report, protocol, audit and raw artifacts: [`results/sensor_contrastive/`](results/sensor_contrastive/REPORT.md).

```bash
python -m src.lowlabel stage-a && python -m src.lowlabel stage-b    # validation-only tuning and selection (~5 min GPU)
python -m src.lowlabel evaluate-test && python -m src.lowlabel report
```

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
| Global-shuffle control | Each LLM model evaluated with every test label paired with a different window from the whole test set (seeded derangement, mean ± std over 5 permutations): the empirical chance baseline |
| Zero-sensor control | Each LLM model with its sensor tokens replaced by zeros, so only the prompt remains |

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

python -m pytest tests -v             # 33 tests, CPU only, no downloads
python -m src.train                   # all 5 conditions × 3 seeds (≈2–3 h on an RTX 5060 Laptop GPU at full power)
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
