Seeds [42, 43, 44]; test = 9 official UCI test subjects; std = sample std (ddof=1) over seeds.

| Condition | 1% labels | 10% labels |
| :--- | ---: | ---: |
| Supervised CNN | 0.861 ± 0.014 | 0.922 ± 0.012 |
| Supervised CNN + aug | 0.866 ± 0.012 | 0.912 ± 0.016 |
| Random encoder + probe | 0.777 ± 0.027 | 0.894 ± 0.002 |
| SimCLR + linear probe | 0.822 ± 0.028 | 0.917 ± 0.010 |
| SimCLR + kNN | 0.739 ± 0.020 | 0.868 ± 0.014 |
| SimCLR + fine-tune | 0.884 ± 0.005 | 0.929 ± 0.009 |

Paired differences in test macro-F1 (same seed, split and labels), mean ± std over seeds, per seed:

| Pair | 1% | 10% |
| :--- | ---: | ---: |
| SimCLR + fine-tune − Supervised CNN | +0.023 ± 0.010 (+0.030 / +0.028 / +0.012) | +0.008 ± 0.006 (+0.013 / +0.002 / +0.008) |
| SimCLR + fine-tune − Supervised CNN + aug | +0.018 ± 0.016 (+0.000 / +0.030 / +0.023) | +0.018 ± 0.021 (-0.007 / +0.032 / +0.029) |
| SimCLR + linear probe − Supervised CNN | -0.039 ± 0.015 (-0.044 / -0.050 / -0.022) | -0.005 ± 0.021 (+0.016 / -0.005 / -0.026) |
| SimCLR + fine-tune − SimCLR + linear probe | +0.062 ± 0.024 (+0.073 / +0.079 / +0.034) | +0.013 ± 0.019 (-0.002 / +0.007 / +0.034) |
