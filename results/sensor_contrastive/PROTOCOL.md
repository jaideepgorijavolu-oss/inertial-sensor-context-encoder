# Protocol: contrastive pretraining under limited labels (UCI HAR)

Status: **v1, approved by Jaideep Gorijavolu on 2026-10-07** (all six proposed choices accepted; time box
open; publishing = push to the feature branch only, owner merges). Once approved, this file is committed before any test
evaluation. Later changes are added as dated amendments at the bottom and never edited in place.

## Question

Does contrastive pretraining on unlabeled training-subject windows improve activity recognition on unseen
subjects when few labels are available, compared with a matched supervised CNN?

A null or negative result is an acceptable outcome. The legacy LLM study is preserved and not used here.

## 1. Data and splits (inductive)

- **Test:** the 9 official UCI test subjects (2 4 9 10 12 13 18 20 24), 2947 windows. They are loaded only by
  `evaluate_test.py`, which runs once after every selection decision is frozen. No test input enters
  pretraining, normalization, training, selection, or any kNN reference set.
- **Validation subjects:** 27 28 29 30 (unchanged from the legacy study), 1485 windows. Their inputs never
  enter pretraining or the fitted standardizer.
- **Training pool:** the 17 remaining official training subjects, 5867 windows. This is the only data for
  SSL pretraining (inputs only) and for the standardizer fit.
- Window IDs are 0-based row indices into the UCI `train/` or `test/` files. Every manifest stores subject
  IDs, window IDs, chain IDs and a sha256 of the raw files.
- Standardizer: per-channel z-score fit on the raw training pool, shared by every method and seed (identical to
  the legacy fit).

## 2. Label budgets and sampling

- Minimum viable study (MVS) budgets: **1 %, 10 %, 100 %** of the training-pool windows, giving 59 / 587 / 5867
  labeled windows. 5 % and 25 % are added only after the MVS is reviewed.
- **Primary sampling: contiguous blocks.** Adjacency is reconstructed by exact matching of overlapping
  halves (AUDIT §2): every chain is one single-activity bout. Each bout is tiled into consecutive,
  non-overlapping blocks of **L = 4 windows** (6.4 s of signal; a bout's last block may be shorter).
- **Stratification:** per-class quota `q_c = max(round(b · n_c), 2)`. For each seed and class, blocks are
  put into one random order, interleaved round-robin across subjects so that small budgets span many
  people. Blocks are taken in that order until `q_c` is reached; the final block is truncated, keeping it
  contiguous, so the count is exact.
- **Nested:** within a seed, the 1 % subset is a prefix of the 10 % subset, which is a prefix of 100 %. The
  same subset is used by every method.
- **Reporting:** budgets are reported as labeled window counts, plus the unique labeled seconds
  (overlapping samples counted once), and counts by subject × class. The remaining pool windows are
  unlabeled. SSL methods use their inputs; supervised methods ignore them. This difference in information
  use is stated with every result.
- Sensitivity analysis (expansion phase, not MVS): i.i.d. stratified window sampling, which matches most
  published low-label HAR setups.

## 3. Validation labels (addendum item 1): **scaled with the budget**

- **Choice:** the labeled validation set `V_b` uses the same fraction *b* of validation-subject windows,
  with the same block sampler and the `max(·, 2)` per-class floor. It is nested across budgets.
  Sizes: about **15 / 149 / 1485** windows at 1 / 10 / 100 %.
- **Why:** the validation-to-train label ratio then stays at about 0.25 at every budget, the same as the
  legacy 100 % setting. A fixed set such as 60 windows would give the 1 % setting as many validation labels
  as training labels. The full 1485 would give it 25× more.
- **Cost:** selection at 1 % is noisy (2–3 windows per class). This is mitigated by small, predeclared
  grids (§6) and applies equally to every method. Every result reports total labels used (train + validation).

## 4. Conditions (MVS)

| Key | Method | Trainable params | Information used |
| --- | --- | ---: | --- |
| `sup` | `SensorEncoder` + linear head from scratch, budget labels (legacy `direct` architecture) | 146,182 | budget labels |
| `sup_aug` | same, trained with the SSL augmentations (§5). **My addition:** the hardest fair baseline; separates "augmentation helps" from "pretraining helps" | 146,182 | budget labels |
| `rand_probe` | untrained `SensorEncoder` frozen, BN statistics recalibrated on unlabeled pool inputs, linear probe | 1,542 | budget labels + pool inputs |
| `ssl_probe` | SimCLR-pretrained encoder frozen, linear probe | 1,542 | budget labels + pool inputs |
| `ssl_knn` | same frozen embeddings, exact kNN; reference set = budget-labeled pool windows only | 0 | budget labels + pool inputs |
| `ssl_ft` | SimCLR initialization, new head, **full end-to-end fine-tuning** with budget labels (addendum item 2) | 146,182 | budget labels + pool inputs |

- **Features:** the 256-d encoder output after global pooling, taken *before* the projection head
  (following SimCLR). All frozen encoders run in eval mode (dropout off, BN running statistics).
- `sup` vs `ssl_ft` is the main comparison ("does pretraining help?"). `ssl_probe` vs `ssl_ft` asks
  "does freezing hurt?" `rand_probe` shows how much a random encoder already gives. The frozen-probe vs
  end-to-end comparisons are not a clean factor isolation.
- Linear probe: multinomial logistic regression (scikit-learn), features standardized with statistics from
  the labeled training windows only.
- kNN: exact search with cosine similarity over L2-normalized embeddings.

## 5. Contrastive method

- **Objective:** SimCLR NT-Xent (Chen et al., 2020). Two augmented views per window; the other 2N − 2 views in
  the batch are negatives. Projection head 256 → 256 → 128 (ReLU). This is **not TS2Vec** and won't be
  called that.
- **Pretraining:** training-pool inputs only. The dataset object holds no labels, and a test asserts this.
  Batch 256, AdamW lr 1e-3, weight decay 1e-4, cosine schedule, **200 epochs (about 4600 steps).
  Checkpoint rule, declared now: the final epoch**, which needs no labels. One pretraining run per seed,
  reused across budgets and across `ssl_probe`, `ssl_knn` and `ssl_ft` in that seed. It is recorded as
  shared, not counted as independent runs.
- **Augmentations** are applied to raw signals before standardization. Channels: total_acc (0–2), body_acc
  (3–5), body_gyro (6–8). Gravity is estimated per sample as `g = total − body`.
  1. **Scaling:** body_acc and gyro are multiplied by one factor s ~ N(1, σ_s). This assumes invariance
     to movement intensity.
  2. **Jitter:** noise n ~ N(0, σ_j · channel std) is added to body_acc, and **the same n is added to
     total_acc**. Gyro gets independent noise (a separate sensor). This assumes invariance to sensor noise.
  3. **Small 3-D rotation:** one random rotation R (random axis, angle ≤ θ) is applied jointly to the body,
     g and gyro triplets, then total = R·body + R·g. This assumes invariance to small phone-placement
     changes. **Full SO(3) rotation is deliberately excluded:** the direction of gravity is what separates
     laying from sitting and standing, and sitting from standing via hip tilt.
  - Strength grid: weak (σ_s 0.1, σ_j 0.05, θ 10°) and strong (σ_s 0.2, σ_j 0.1, θ 30°).
- **Collapse checks**, logged every epoch on a fixed unlabeled pool subset: per-dimension std of
  L2-normalized embeddings (about 1/√256 when healthy, near 0 when collapsed), effective rank, and
  alignment/uniformity. A run is flagged if the std falls below 0.1/√256.
- **False negatives,** to be discussed in the report: about 1/6 of in-batch negatives share the anchor's
  activity. Adjacent windows overlap by 50 % but are still treated as negatives; at batch 256 that is about
  11 overlapping pairs per batch. Standard SimCLR is kept for the MVS; masking chain neighbors is a possible
  ablation.

## 6. Tuning and selection: validation only, predeclared, logged

- **Stage A** (seed 42, 10 % budget, `V_10%`). It is run symmetrically for both families, and every config and
  score is logged.
  - SSL: τ ∈ {0.1, 0.5} × augmentation {weak, strong}, 4 pretraining runs. Chosen by `ssl_probe`
    validation macro-F1.
  - Supervised: lr ∈ {1e-3, 3e-4} × weight decay ∈ {1e-4, 1e-2}, 4 runs (the `sup_aug` augmentation strength
    follows the SSL choice).
  - Fine-tuning lr ∈ {1e-3, 1e-4}.
  - The winners are fixed for all seeds and budgets. **Disclosure:** stage A uses 10 % labels (587 + 149),
    which is more than the 1 % budget for every method family.
- **Stage B** (each seed and budget, on `V_b`):
  - `sup`, `sup_aug`, `ssl_ft`: 1400 steps (about the legacy 15 epochs at 100 %), batch min(64, n). Validation
    is evaluated every 70 steps; the best checkpoint is kept, with ties broken toward the earliest.
  - Probes: C ∈ {0.01, 0.1, 1, 10}, ties broken toward the smaller C.
  - kNN: k ∈ {1, 3, 5, 10, 20} (k ≤ n) × {uniform, distance-weighted}, ties broken toward the smaller k and
    uniform weights.
- No test metric is computed until all stage-B selections for every condition, seed and budget are frozen
  in `selections.json`. `evaluate_test.py` then runs once and logs its invocation.

## 7. Metrics and reporting

- **Primary:** test macro-F1. Also per-class F1, confusion matrices, accuracy, and per-test-subject
  macro-F1 (9 subjects; windows are not independent people).
- **Seeds** 42, 43, 44. A seed sets the label subset, `V_b`, initialization and the SSL run. Every seed is
  reported, with mean and sample std (ddof = 1) and **paired differences** per seed (`ssl_ft − sup`,
  `ssl_ft − sup_aug`, `ssl_probe − sup`, `ssl_ft − ssl_probe`), plus per-subject paired differences.
  There are no significance or equivalence claims from 3 seeds.
- Plot: macro-F1 against label budget, log x, showing all seeds and the mean.
- Cost: training wall time, SSL pretraining cost (shown separately, amortized across budgets), trainable
  parameters, embedding dimension (256), and peak memory (`torch.cuda.max_memory_allocated` on GPU; peak RSS
  via psutil on CPU). Measurement method and device are recorded per run.
- **Literature comparison:** SimCLR (Chen et al., 2020); Tang et al. (2020), contrastive learning for HAR;
  Haresamudram, Essa & Plötz (2022), assessing self-supervised HAR. Their reported numbers will be quoted with
  their dataset and protocol differences (e.g. random vs subject splits, label-sampling scheme). They are
  not presented as directly comparable.
- **History disclosure:** UCI HAR test results were inspected in the legacy study (AUDIT D8).

## 8. Legacy reproduction (before the new runs)

Retrain the legacy `direct` model for 3 seeds under its original protocol into
`results/sensor_contrastive/legacy_repro/`, compare it with the stored records, and report the result.
The LLM conditions are not retrained (AUDIT D1/D1b).

## 9. After the MVS (not authorized yet)

5 % and 25 % budgets, i.i.d.-sampling sensitivity, one validation-chosen ablation (e.g. removing rotation),
and a possible second dataset. Each needs a separate review.

## 10. Artifacts

`results/sensor_contrastive/<run_id>/`: `config.json`, `command.txt`, `env.json` (commit, dirty flag and diff
hash, package versions, device, data sha256), `split_manifest.json`, `budget_subsets.json`,
`selections.json`, per-run `metrics.json` with per-example test predictions, stdout and stderr logs, and
checkpoint sha256 values. `.pt` files stay gitignored and go to a GitHub release later. Legacy `artifacts/`
is never written to.

## Amendments

(none)

### Amendment 1 (2026-10-07, after the MVS report; owner: "keep going")

Runs the expansion from §9 with **no change to methods or stage-A choices** (`mvs/stage_a.json` is reused), and
reuses the SSL encoders in `mvs/ssl/` (same inputs and checkpoint rule). Each run has its own directory and its own
one-time test evaluation:

- `budgets_5_25/`: budgets 5 % and 25 %, all conditions. These budgets were predeclared in §2.
- `iid/`: i.i.d. stratified window sampling (`--block-len 1`) at 1 % and 10 %, all conditions. This is the
  sensitivity analysis from §2. At 100 % the sampling scheme makes no difference.
- `ablation_norot/`: the strong augmentation without rotation, for `sup_aug`, `ssl_probe` and `ssl_ft` at 1 / 10 /
  100 %. This is the ablation named in §9 before any test result was seen. The validation scores are reported next
  to the test scores.

No further tuning. Results are added to REPORT.md as a separate section, and MVS numbers are not edited.
