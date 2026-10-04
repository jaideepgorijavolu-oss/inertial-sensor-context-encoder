Seeds: [42, 43, 44] | epochs: 15 (context: 15) | sensor tokens (multi-token): 8 | LoRA rank: 8 | standardized inputs: True | trained on: ['NVIDIA GeForce RTX 5060 Laptop GPU'] | report generated on: NVIDIA GeForce RTX 5060 Laptop GPU

| Condition | Test Macro-F1 (mean ± std) | Trainable params | Latency ms/window |
| :--- | :--- | ---: | ---: |
| Direct sensor classifier | 0.9337 ± 0.0056 | 146,182 | 0.49 |
| Matched-capacity, no LLM (ablation) | 0.9250 ± 0.0132 | 1,319,686 | 0.89 |
| Context model, 1 sensor token (frozen LLM) | 0.9333 ± 0.0033 | 1,319,686 | 28.97 |
| Context model, multi-token (frozen LLM) | 0.9205 ± 0.0108 | 1,319,686 | 29.81 |
| Context model, multi-token + LoRA | 0.9337 ± 0.0019 | 2,138,886 | 30.14 |
| Context model, 1 token: globally shuffled sensor (control) | 0.1636 ± 0.0015 | 0 | - |
| Context model, 1 token: zeroed sensor (control) | 0.0500 ± 0.0021 | 0 | - |
| Context model, multi-token: globally shuffled sensor (control) | 0.1633 ± 0.0020 | 0 | - |
| Context model, multi-token: zeroed sensor (control) | 0.0483 ± 0.0028 | 0 | - |
| Context model, multi-token + LoRA: globally shuffled sensor (control) | 0.1637 ± 0.0010 | 0 | - |
| Context model, multi-token + LoRA: zeroed sensor (control) | 0.0501 ± 0.0022 | 0 | - |
