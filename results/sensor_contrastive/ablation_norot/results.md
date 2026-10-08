Seeds [42, 43, 44]; test = 9 official UCI test subjects; std = sample std (ddof=1) over seeds.

| Condition | 1% labels | 10% labels | 100% labels |
| :--- | ---: | ---: | ---: |
| Supervised CNN + aug | 0.805 ± 0.013 | 0.913 ± 0.010 | 0.923 ± 0.006 |
| SimCLR + linear probe | 0.597 ± 0.064 | 0.867 ± 0.015 | 0.913 ± 0.009 |
| SimCLR + fine-tune | 0.760 ± 0.064 | 0.913 ± 0.004 | 0.920 ± 0.017 |

Paired differences in test macro-F1 (same seed, split and labels), mean ± std over seeds, per seed:

| Pair | 1% | 10% | 100% |
| :--- | ---: | ---: | ---: |
| SimCLR + fine-tune − Supervised CNN + aug | -0.044 ± 0.053 (+0.010 / -0.048 / -0.096) | +0.000 ± 0.006 (+0.003 / +0.005 / -0.007) | -0.003 ± 0.024 (+0.012 / -0.030 / +0.009) |
| SimCLR + fine-tune − SimCLR + linear probe | +0.163 ± 0.058 (+0.171 / +0.217 / +0.102) | +0.046 ± 0.011 (+0.051 / +0.053 / +0.033) | +0.007 ± 0.016 (+0.023 / -0.009 / +0.006) |
