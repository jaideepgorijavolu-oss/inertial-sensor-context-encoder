Seeds: [42, 43, 44] | epochs: 15 (context: 15) | sensor tokens (multi-token): 8 | LoRA rank: 8 | standardized inputs: True | device: NVIDIA GeForce RTX 5060 Laptop GPU

| Condition | Test Macro-F1 (mean ± std) | Trainable params | Latency ms/window |
| :--- | :--- | ---: | ---: |
| Direct sensor classifier | 0.9337 ± 0.0056 | 146,182 | 0.44 |
| Matched-capacity, no LLM (ablation) | 0.9279 ± 0.0096 | 1,319,686 | 0.45 |
| Context model, 1 sensor token (frozen LLM) | 0.9238 ± 0.0057 | 1,319,686 | 24.97 |
| Context model, multi-token (frozen LLM) | 0.9205 ± 0.0108 | 1,319,686 | 25.97 |
| Context model, multi-token + LoRA | 0.9155 ± 0.0278 | 2,138,886 | 31.83 |
| Context model, 1 token, shuffled (control) | 0.3618 ± 0.0013 | 0 | - |
| Context model, multi-token, shuffled (control) | 0.3610 ± 0.0039 | 0 | - |
| Context model, multi-token + LoRA, shuffled (control) | 0.3590 ± 0.0097 | 0 | - |
