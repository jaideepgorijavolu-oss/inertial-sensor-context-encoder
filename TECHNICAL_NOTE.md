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
| **Context-embedding model** | CNN + projector into frozen SmolLM2-360M |
| **Matched-capacity ablation** | Identical trainable stack (CNN + projector + head) without the LLM, separating the LLM's contribution from the extra parameters |
| **Shuffled negative control** | Context model with sensor tokens permuted across the batch (averaged over 5 permutations); tests sensor dependence |

---

## 4. Results

Run `python -m src.train`; the table below is generated verbatim in `artifacts/results.md`.

> **Note:** results from the earlier version of this repo (Direct 0.9337, Context 0.5130,
> Shuffled 0.2618 macro-F1, single seed) were produced while the encoder/projector of the
> context model received no gradients and with an unequal epoch budget (15 vs 5). They
> are superseded and should be regenerated with the current code.

| Condition | Test Macro-F1 (mean ± std) | Trainable params | Latency ms/window |
| :--- | :--- | ---: | ---: |
| Direct sensor classifier | _rerun_ | _rerun_ | _rerun_ |
| Matched-capacity, no LLM (ablation) | _rerun_ | _rerun_ | _rerun_ |
| Context-embedding model (frozen LLM) | _rerun_ | _rerun_ | _rerun_ |
| Context model, shuffled sensor tokens (control) | _rerun_ | 0 | — |

---

## 5. What to look for

* **Context vs. matched ablation**: the cleanest test of whether the frozen LLM adds value; both have the same trainable parameters, so any difference comes from the backbone.
* **Context vs. shuffled control**: a large drop confirms predictions are driven by the sensor token rather than the static prompt.
* **Direct vs. context**: the accuracy-per-compute trade-off. Latency per window is reported for each condition; the 360M-parameter backbone dominates inference cost.
* **Per-class F1**: Sitting vs. Standing is the known hard pair on UCI HAR; check the confusion matrices before attributing gains to the LLM.

---

## 6. Reproduction

```bash
python -m src.download_data
python -m pytest tests -v
python -m src.train
```
