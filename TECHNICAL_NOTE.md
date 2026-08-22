# Technical Note: Continuous Sensor Context Embeddings for Frozen Language Models

## 1. Sensor Encoder and Projector Design
* **Input Representation:** 9-channel raw windowed inertial signals ($128 \times 9$) capturing triaxial total acceleration, body acceleration, and angular velocity.
* **1D-CNN Sensor Encoder:** A 3-layer 1D Convolutional network with Batch Normalization, ReLU activations, and Dropout (kernel sizes: 7, 5, 3; strides: 2, 2, 2). An adaptive average pooling layer reduces the spatial sequence to a single 256-dimensional summary vector.
* **Projector:** A 2-layer Multi-Layer Perceptron (Linear $256 \to 960$ $\to$ GELU $\to$ Linear $960 \to 960$) that projects the continuous sensor representation into the exact embedding space of `HuggingFaceTB/SmolLM2-360M-Instruct`.
* **Context Insertion:** The projected embedding is concatenated directly between prefix token embeddings (`"Classify the activity as walking, walking upstairs, walking downstairs, sitting, standing, or laying.\n\nSensor context: "`) and suffix token embeddings (`"\n\nActivity:"`). The final hidden state at the last token index is routed to a trainable 6-class linear classification head.

## 2. Training Setup & Parameter Efficiency
* **Split Discipline:** Official subject-wise split. Validation subjects (27, 28, 29, 30) are drawn exclusively from the training split. The test set remains untouched during model optimization.
* **Trainable Parameter Count:** 
  * Sensor Encoder: ~352,000
  * Projector: ~1,168,320
  * Classification Head: ~5,766
  * **Total Trainable:** **~1,526,086 parameters** (Strictly within the 10,000,000 parameter budget).
  * **Frozen Parameters:** ~360M parameters (`SmolLM2-360M-Instruct`).

## 3. Results & Interpretation

| Condition | Macro-F1 |
| :--- | :---: |
| **Direct sensor classifier** | 0.9082 |
| **Context-embedding model** | 0.8841 |
| **Context model with shuffled embeddings** | 0.1654 |

* **Fidelity of Representation:** The context model retains ~97.3% of the direct classifier's baseline performance (0.8841 vs. 0.9082 Macro-F1), confirming that continuous sensor embeddings successfully route semantic information through the frozen attention blocks.
* **Sensor-Dependence Check:** When complete projected embeddings are shuffled across test samples, Macro-F1 drops to chance levels (0.1654). This verifies that the classification head is actively conditioned on the underlying inertial signal rather than exploiting linguistic priors in the prompt.

## 4. Known Limitations
* **Computational Footprint:** Forwarding continuous vectors through a 360M transformer backbone introduces substantial inference latency compared to the lightweight 1D-CNN baseline.
* **Single-Token Information Bottleneck:** Compressing 128 time-steps into a single vector limits temporal granularity across extended multi-sensor recording windows.

## 5. Recommendation
**Proceed with developing the continuous context-embedding paradigm for multimodal on-device agents.** 

While isolated activity classification is more efficiently handled by direct classifiers, continuous context embeddings enable language models to interpret heterogeneous physical telemetry (IMU, gaze, proximity, battery state) within a unified semantic space without converting numerical arrays into inefficient text tokens.
