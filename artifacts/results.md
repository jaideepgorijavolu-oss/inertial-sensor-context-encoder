Seeds: [42, 43, 44] | epochs: 15 (context: 15) | standardized inputs: True | device: NVIDIA GeForce RTX 5060 Laptop GPU

| Condition | Test Macro-F1 (mean ± std) | Trainable params | Latency ms/window |
| :--- | :--- | ---: | ---: |
| Direct sensor classifier | 0.9337 ± 0.0056 | 146,182 | 0.44 |
| Matched-capacity, no LLM (ablation) | 0.9279 ± 0.0096 | 1,319,686 | 0.46 |
| Context-embedding model (frozen LLM) | 0.9238 ± 0.0057 | 1,319,686 | 25.66 |
| Context model, shuffled sensor tokens (control) | 0.3618 ± 0.0013 | 0 | - |
