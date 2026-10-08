Seeds [42, 43, 44]; test = 9 official UCI test subjects; std = sample std (ddof=1) over seeds.

| Condition | 5% labels | 25% labels |
| :--- | ---: | ---: |
| Supervised CNN | 0.899 ± 0.019 | 0.935 ± 0.007 |
| Supervised CNN + aug | 0.884 ± 0.023 | 0.906 ± 0.006 |
| Random encoder + probe | 0.848 ± 0.001 | 0.896 ± 0.012 |
| SimCLR + linear probe | 0.892 ± 0.009 | 0.930 ± 0.008 |
| SimCLR + kNN | 0.800 ± 0.038 | 0.894 ± 0.005 |
| SimCLR + fine-tune | 0.902 ± 0.008 | 0.926 ± 0.016 |

Paired differences in test macro-F1 (same seed, split and labels), mean ± std over seeds, per seed:

| Pair | 5% | 25% |
| :--- | ---: | ---: |
| SimCLR + fine-tune − Supervised CNN | +0.004 ± 0.021 (+0.028 / -0.007 / -0.009) | -0.009 ± 0.019 (+0.013 / -0.017 / -0.023) |
| SimCLR + fine-tune − Supervised CNN + aug | +0.018 ± 0.031 (+0.029 / +0.042 / -0.016) | +0.020 ± 0.020 (+0.036 / +0.025 / -0.003) |
| SimCLR + linear probe − Supervised CNN | -0.007 ± 0.013 (+0.004 / -0.020 / -0.004) | -0.005 ± 0.015 (+0.005 / -0.022 / +0.002) |
| SimCLR + fine-tune − SimCLR + linear probe | +0.011 ± 0.015 (+0.023 / +0.014 / -0.005) | -0.004 ± 0.018 (+0.008 / +0.005 / -0.026) |
