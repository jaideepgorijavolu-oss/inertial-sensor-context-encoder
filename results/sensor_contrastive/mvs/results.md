Seeds [42, 43, 44]; test = 9 official UCI test subjects; std = sample std (ddof=1) over seeds.

| Condition | 1% labels | 10% labels | 100% labels |
| :--- | ---: | ---: | ---: |
| Supervised CNN | 0.800 ± 0.014 | 0.913 ± 0.008 | 0.926 ± 0.011 |
| Supervised CNN + aug | 0.830 ± 0.019 | 0.900 ± 0.023 | 0.906 ± 0.005 |
| Random encoder + probe | 0.678 ± 0.018 | 0.867 ± 0.004 | 0.910 ± 0.006 |
| SimCLR + linear probe | 0.778 ± 0.011 | 0.899 ± 0.006 | 0.950 ± 0.004 |
| SimCLR + kNN | 0.656 ± 0.028 | 0.858 ± 0.018 | 0.915 ± 0.003 |
| SimCLR + fine-tune | 0.847 ± 0.021 | 0.915 ± 0.012 | 0.932 ± 0.008 |

Paired differences in test macro-F1 (same seed, split and labels), mean ± std over seeds, per seed:

| Pair | 1% | 10% | 100% |
| :--- | ---: | ---: | ---: |
| SimCLR + fine-tune − Supervised CNN | +0.047 ± 0.026 (+0.076 / +0.026 / +0.038) | +0.002 ± 0.006 (-0.005 / +0.003 / +0.007) | +0.007 ± 0.019 (-0.014 / +0.010 / +0.023) |
| SimCLR + fine-tune − Supervised CNN + aug | +0.017 ± 0.034 (+0.055 / -0.009 / +0.004) | +0.015 ± 0.023 (-0.009 / +0.036 / +0.016) | +0.026 ± 0.012 (+0.013 / +0.029 / +0.037) |
| SimCLR + linear probe − Supervised CNN | -0.022 ± 0.018 (-0.004 / -0.040 / -0.023) | -0.014 ± 0.013 (-0.008 / -0.005 / -0.029) | +0.024 ± 0.010 (+0.013 / +0.031 / +0.028) |
| SimCLR + fine-tune − SimCLR + linear probe | +0.069 ± 0.010 (+0.080 / +0.066 / +0.061) | +0.016 ± 0.018 (+0.003 / +0.008 / +0.036) | -0.017 ± 0.012 (-0.027 / -0.021 / -0.004) |
