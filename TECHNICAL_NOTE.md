# Technical Note: Multimodal Inertial Sensor-to-Language Context Projection

## 1. Architectural Overview

* **Input Representation**: 9-channel inertial telemetry (triaxial total acceleration, body acceleration, angular velocity) sampled at 50 Hz across 128 timesteps (`[B, 9, 128]`).
* **Feature Extraction**: 3-stage 1D-CNN backbone (kernel size 5, stride 1) with batch normalization, ReLU activations, and adaptive average pooling down to an embedding vector.
* **Cross-Modal Projector**: 2-layer MLP projecting 256-dimensional sensor features into SmolLM2's native 960-dimensional token space.
* **Language Backbone**: Frozen `HuggingFaceTB/SmolLM2-360M` (361,821,120 parameters). Sensor vectors are concatenated between text prompt tokens (`inputs_embeds`).
* **Classification Head**: Linear projection layer mapping the final LLM hidden state (`d_model = 960`) to 6 output activity logits.
* **Parameter Efficiency**: 1,319,686 trainable parameters (~0.36% of backbone size).

---

## 2. Evaluation Protocol & Data Isolation

* **Dataset**: UCI Human Activity Recognition (30 participants).
* **Leakage Prevention**: Evaluated strictly on unseen subjects. Subject-level validation partitions ensure 0% identity or time-series overlap between train, validation, and test splits.
* **Primary Metric**: Macro-F1 across all 6 classes (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) to account for slight class frequency variations.

---

## 3. Quantitative Results

| Experimental Condition | Test Macro-F1 | Trainable Parameters | Description |
| :--- | :--- | :--- | :--- |
| **Condition 1: Direct Sensor Classifier** | **0.9337** | 145,094 | Dedicated 1D-CNN baseline directly optimized for classification |
| **Condition 2: Context-Embedding Model** | **0.5130** | 1,319,686 | 1D-CNN + MLP projector into frozen SmolLM2-360M |
| **Condition 3: Negative Control (Shuffled)** | **0.2618** | 0 (Inference) | Condition 2 evaluated with permuted sensor tokens across batch |

---

## 4. Key Findings & Discussion

* **Physical Feature Grounding**: Permuting the sensor token across the batch (Condition 3) drops test Macro-F1 from 0.5130 down to 0.2618 (a 48.97% relative drop). This confirms the classification head extracts state signals directly from injected sensor embeddings rather than over-indexing on the static text prompt.
* **Compute-to-Accuracy Trade-off**: The direct 1D-CNN baseline achieves 0.9337 Macro-F1 with 89% fewer trainable parameters, sub-millisecond per-window latency, and negligible RAM footprint. Contextual LLM injection introduces significant memory and inference overhead, demonstrating that for pure discrete classification, specialized lightweight encoders remain vastly superior.
* **Autograd Optimization**: Wrapping the frozen 360M transformer forward pass in `torch.no_grad()` prevented computational graph retention across 30 transformer layers, eliminating CPU memory thrashing.

---

## 5. Reproduction

```bash
# Run unit tests
python -m pytest tests -v

# Train and reproduce full evaluation summary
python -m src.train