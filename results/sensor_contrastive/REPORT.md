# Phase-one report: contrastive pretraining under limited labels (UCI HAR)

Minimum viable study, 2026-10-07. Protocol: [PROTOCOL.md](PROTOCOL.md) (committed `f963d68`, before any run).
Audit: [AUDIT.md](AUDIT.md). The validation selections were frozen and committed (`846fb66`) before the
one-time test evaluation. Code: `src/lowlabel*.py`, `src/augment.py`, `src/ssl.py`; tests:
`tests/test_lowlabel.py`.

## Result

Test = 9 unseen official UCI test subjects (2947 windows). Values are mean ± sample std (ddof = 1) over seeds
42/43/44. Each seed fixes the label subset, validation subset, initialization and SSL run, and all methods
share them. Labels are block-sampled: 59 / 587 / 5867 training windows, plus 15 / 148 / 1485 validation windows.

| Condition | 1 % | 10 % | 100 % | Trainable params |
| :--- | ---: | ---: | ---: | ---: |
| Supervised CNN | 0.800 ± 0.014 | 0.913 ± 0.008 | 0.926 ± 0.011 | 146,182 |
| Supervised CNN + same augmentations | 0.830 ± 0.019 | 0.900 ± 0.023 | 0.906 ± 0.005 | 146,182 |
| Random encoder + linear probe | 0.678 ± 0.018 | 0.867 ± 0.004 | 0.910 ± 0.006 | 1,542 |
| SimCLR + linear probe (frozen) | 0.778 ± 0.011 | 0.899 ± 0.006 | **0.950 ± 0.004** | 1,542 |
| SimCLR + exact kNN (frozen) | 0.656 ± 0.028 | 0.858 ± 0.018 | 0.915 ± 0.003 | 0 |
| SimCLR + full fine-tune | **0.847 ± 0.021** | 0.915 ± 0.012 | 0.932 ± 0.008 | 146,182 |

Paired differences (same seed, split and labels; per seed 42 / 43 / 44):

| Pair | 1 % | 10 % | 100 % |
| :--- | --- | --- | --- |
| fine-tune − supervised | +0.047 (+0.076 / +0.026 / +0.038) | +0.002 (−0.005 / +0.003 / +0.007) | +0.007 (−0.014 / +0.010 / +0.023) |
| fine-tune − supervised + aug | +0.017 (+0.055 / −0.009 / +0.004) | +0.015 (−0.009 / +0.036 / +0.016) | +0.026 (+0.013 / +0.029 / +0.037) |
| linear probe − supervised | −0.022 (−0.004 / −0.040 / −0.023) | −0.014 (−0.008 / −0.005 / −0.029) | +0.024 (+0.013 / +0.031 / +0.028) |
| fine-tune − linear probe | +0.069 (+0.080 / +0.066 / +0.061) | +0.016 (+0.003 / +0.008 / +0.036) | −0.017 (−0.027 / −0.021 / −0.004) |

Raw data: `mvs/test_metrics.json` (per-class F1, confusion matrices, per-subject F1, every prediction),
`mvs/selections.json`, `mvs/summary.json`, `mvs/f1_vs_budget.png`, and the logs.

![macro-F1 vs label budget](mvs/f1_vs_budget.png)

## What the results support

1. **At 1 % labels (59 windows), SimCLR pretraining followed by fine-tuning beats the same CNN trained from
   scratch in all 3 seeds** (+0.047 mean). Most of this advantage disappears against the supervised CNN
   trained with the same augmentations (+0.017, with one seed negative). The augmentations explain a large
   share of the low-label gain; pretraining adds a smaller, inconsistent amount on top.
2. **Freezing hurts at low labels and helps at full labels.** Fine-tuning beats the frozen probe by 0.069
   at 1 % (all seeds), while the frozen probe beats fine-tuning by 0.017 at 100 % (all seeds). This is the
   distinction the fine-tune condition was added to separate.
3. **At 100 % labels the frozen SimCLR encoder with a 1,542-parameter linear probe is the best model in this
   study:** 0.950 ± 0.004, above the supervised CNN in all 3 seeds (+0.024). It is also above the legacy direct
   CNN (0.934, which reproduced bit-exactly). The gain comes almost entirely from the hard Sitting/Standing pair
   (per-class F1 0.914 / 0.913 vs 0.811 / 0.839). The probe's worst test subject is also higher
   (0.76–0.82 vs 0.65–0.73).
4. **Representation learning matters.** A random encoder with BN statistics recalibrated on the same unlabeled
   pool trails the SimCLR probe at every budget (by 0.03–0.10).
5. **Exact kNN on SimCLR embeddings is the weakest SimCLR readout** at every budget. The embedding geometry is
   less class-separable than a learned linear boundary reveals.

## What the results do not support

- No significance or equivalence claims: there are 3 seeds, and seed variance does not cover variation across
  people. Differences under about 0.02 with mixed-sign seeds (10 % budget, fine-tune vs supervised) are
  inconclusive.
- "Pretraining beats augmentation-matched supervision" is not shown at 1 % or 10 %.
- Why the frozen probe wins at 100 % is not established. A plausible explanation is that pretraining under
  rotation and scaling invariance yields posture features that full supervision on 17 subjects overfits away.
  Another is that the probe's validation-tuned regularization suits the subject shift. The planned ablation
  (removing rotation) would test the first.
- The SSL methods saw all 5867 unlabeled pool windows; supervised baselines did not. That is the point of the
  comparison, but it is extra information.

## Unexpected findings and negative results (experiment log)

- **Validation granularity:** at 10 %, `V_b` has 148 windows. All supervised and fine-tune stage-A configs tied
  at 0.9250 because the models peak at the first checkpoint (step 70) and macro-F1 on 148 block-sampled windows
  takes few values. This is not a bug (histories inspected); the predeclared tie-break chose the first grid entry.
  At 1 %, `V_b` has 15 windows and validation F1 often hits 1.000, so selection there is close to arbitrary.
  This is an inherent cost of honest low-label validation and is reported, not tuned around.
- **Augmentation hurts the supervised CNN at 100 %** (0.906 vs 0.926). Its Sitting F1 drops to 0.734: strong
  rotations (up to 30°) partly blur the gravity-direction cue between Sitting and Standing. The frozen SimCLR
  probe does not show this.
- The new `sup` at 100 % (0.926) is below the legacy direct CNN (0.934). It uses the same architecture but a
  different schedule: 1400 steps, validation every 70 steps, and the stage-A hyperparameters.
- SSL stage A: τ = 0.1 beat τ = 0.5, and strong augmentation beat weak (probe validation F1 0.960 vs 0.912 at
  τ = 0.1). No run collapsed: normalized per-dimension std was 0.040–0.044, against a 0.0063 alarm threshold;
  effective rank was 71–81 of 256.

## Cost (RTX 5060 Laptop GPU, `torch.cuda.max_memory_allocated`)

SimCLR pretraining: 31–50 s per run (200 epochs, batch 256), 226 MB peak, amortized across all budgets and
SSL conditions of a seed. Supervised and fine-tune runs: 4–6 s (1400 steps), 154 MB. Probes and kNN: under 8 s.
Embedding dimension: 256.

## Comparison with published work

The method is SimCLR's NT-Xent objective (Chen et al., 2020) applied to the existing CNN. It follows Tang et al.
(2020) in applying SimCLR to HAR, and Haresamudram, Essa & Plötz (2022) in evaluating with frozen linear probes
across label budgets. Their reported numbers use different datasets and splits (often random rather than
subject-held-out, and different label-sampling schemes), so they are **not yet quoted**. That comparison is
pending: it needs the papers' exact tables, which were not fetched in this session, and no figure is
reproduced from memory.

## Limitations

There is one dataset (smartphone at the waist, 30 subjects) and 3 seeds. The test set was inspected in earlier
work (AUDIT D8). Stage-A tuning used 10 % labels, more than the 1 % budget, symmetrically for both families.
The 5 % and 25 % budgets, i.i.d. sampling sensitivity and the rotation ablation are not run yet (protocol §9).

## Questions you should be able to answer

1. What does one SimCLR batch contain, and which 2N − 2 views are negatives? Why are 50 %-overlapping
   neighbours a false-negative risk?
2. Why can't a test-subject window reach the standardizer, the SSL inputs or the kNN reference set? Which test
   enforces each boundary?
3. Why does scaling validation labels with the budget make 1 % selection noisy, and why is that still the
   honest choice?
4. Why is "SimCLR + fine-tune beats supervised at 1 %" weaker evidence once `sup_aug` is included?
5. Why exclude full SO(3) rotation, and what does the `sup_aug` Sitting F1 drop suggest about even 30°?
6. Why does the frozen probe beat fine-tuning at 100 % but lose at 1 %?
