# Technical Note: Multimodal Inertial Sensor-to-Language Context Projection

## 1. Architectural Overview

* **Input Representation**: 9-channel inertial telemetry (triaxial total acceleration, body acceleration, angular velocity) sampled at 50 Hz across 128 timesteps (`[B, 128, 9]`), z-scored per channel with statistics from the training subjects only.
* **Feature Extraction**: 3-stage 1D-CNN (kernels 7/5/3, stride 2, channels 64/128/256) with batch normalization, ReLU, dropout, and global average pooling to a 256-d vector.
* **Cross-Modal Projector**: 2-layer MLP (Linear → GELU → Linear) projecting 256-d sensor features into SmolLM2's 960-d token-embedding space.
* **Language Backbone**: Frozen `HuggingFaceTB/SmolLM2-360M-Instruct` base transformer (no LM head). The projected sensor vector is spliced as one soft token between prompt-prefix and prompt-suffix token embeddings (`inputs_embeds`).
* **Classification Head**: Linear layer on the final-layer hidden state at the last position (`d_model = 960`) → 6 activity logits.
* **Gradient path**: backbone weights have `requires_grad=False`, but the forward pass is *not* wrapped in `torch.no_grad()`, so the loss backpropagates through the frozen transformer into the projector and encoder. (An earlier version wrapped the backbone in `no_grad`, which detached the sensor token from the loss and left the encoder and projector at their random initialization; `tests/test_context_model.py` guards against this regression.)

---

## 2. Evaluation Protocol & Data Isolation

* **Dataset**: UCI Human Activity Recognition (30 participants).
* **Leakage Prevention**: Subject-disjoint splits. Validation = training subjects 27–30; test = the official UCI test subjects. Disjointness is asserted at load time and in the test suite. Normalization statistics come from training subjects only.
* **Model selection**: best validation macro-F1 checkpoint (a true copy of the weights, not a reference to the live parameters).
* **Budget parity**: all conditions train for the same number of epochs by default (AdamW, cosine LR schedule, gradient clipping at 1.0).
* **Repetition**: 3 seeds by default; results reported as mean ± sample std.
* **Primary Metric**: Macro-F1 across the 6 classes (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying). Per-class F1 and confusion matrices are saved to `artifacts/results.json`.

---

## 3. Conditions

| Condition | Purpose |
| :--- | :--- |
| **Direct sensor classifier** | Task-specific CNN baseline |
| **Matched-capacity ablation** | Identical trainable stack (CNN + projector + head) without the LLM, separating the LLM's contribution from the extra parameters |
| **Context model, 1 token (frozen)** | CNN + projector into one soft token in frozen SmolLM2-360M |
| **Context model, 8 tokens (frozen)** | Conv features pooled into 8 temporal-segment tokens, so the LLM sees the signal's time structure |
| **Context model, 8 tokens + LoRA** | As above, plus rank-8 LoRA adapters on attention q/v projections (+819K trainable parameters) |
| **Global-shuffle control** | Each context model with every test label paired with a different window from the whole test set (seeded derangement, mean ± std over 5 permutations); the empirical chance baseline |
| **Zero-sensor control** | Each context model with sensor tokens replaced by zeros (prompt only) |

---

## 4. Results

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

The LLM conditions also carry 361.8M frozen backbone parameters.

Mean per-class F1 over seeds:

| Class | Direct | Matched (no LLM) | 1 token | 8 tokens | 8 tokens + LoRA |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Walking | 0.997 | 0.991 | 0.986 | 0.973 | 0.981 |
| Walking upstairs | 0.951 | 0.949 | 0.962 | 0.955 | 0.964 |
| Walking downstairs | 0.976 | 0.962 | 0.957 | 0.953 | 0.952 |
| Sitting | 0.835 | 0.813 | 0.842 | 0.817 | 0.847 |
| Standing | 0.859 | 0.844 | 0.859 | 0.837 | 0.861 |
| Laying | 0.984 | 0.991 | 0.993 | 0.987 | 0.998 |

> An earlier version of these results (before the per-condition batch-order fix and the global-shuffle control) reported a batch-local shuffle control of about 0.36 and a LoRA seed at 0.883. With the fixes, the conditions that trained first in their runs (direct CNN, 8-token frozen) reproduced exactly; the others changed as listed above, and the LoRA outlier did not recur.

> The first version of this repo reported Context 0.5130 (single seed): a `torch.no_grad()` around the backbone meant the encoder and projector never trained.

---

## 5. Findings

1. **A frozen LLM reading raw sensor embeddings matches a dedicated CNN.** One soft token into frozen SmolLM2-360M reaches 0.933 ± 0.003 macro-F1, within seed-to-seed spread of the direct CNN (0.934 ± 0.006; 3 seeds, no significance test).
2. **Predictions depend on the sensor input.** Pairing every test label with an unrelated window (global shuffle) drops every LLM model to 0.163–0.164 macro-F1, chance level for 6 classes; replacing the sensor tokens with zeros gives 0.048–0.050 (a single constant prediction). The prompt alone carries no class information.
3. **LoRA gives the most stable LLM result.** Rank-8 adapters reach 0.934 ± 0.002, tying the CNN with the lowest seed-to-seed variance of any condition.
4. **No clear gain from the LLM at matched capacity.** The frozen-LLM and LoRA models score about 0.8 points above the parameter-matched network without the LLM (0.925 ± 0.013), but the gap is within its seed-to-seed spread across 3 seeds.
5. **More sensor tokens do not help.** Eight temporal tokens (0.921 ± 0.011) score below one pooled token.
6. **Cost:** the LLM pathway is about 60× slower per window (29–30 ms vs 0.49 ms) and adds 1.4 GB of weights. For closed-set activity recognition a specialized encoder is the efficient choice; the LLM pathway earns its cost when the task needs language (explanations, open-ended questions, reasoning over sensor context).
* **Errors are the expected ones.** All models lose most of their F1 on Sitting vs. Standing (0.81–0.86), the known hard pair in UCI HAR; dynamic activities and Laying are at least 0.95.

---

## 6. Reproduction

```bash
python -m src.download_data
python -m pytest tests -v
python -m src.train
```
