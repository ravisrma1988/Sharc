# Preregistration: Same-Network Deployment Grid (audit-to-label decision)

**Registered:** 10 Oct 2026, before any code for this experiment was written or run. Saved on D drive and DGX; SHA-256 recorded in the experiment log at launch.
**Requested by:** consolidated final audit (`JBD_Consolidated_Final_Suggestions_With_Targeted_Experiment.pdf`, Section 3-4).

## 1. Question
In repeated deployments within one screening network, does the gated policy avoid target annotation when only prevalence changes, and require annotation for appearance and mixed shifts? The experiment tests the decision to buy labels, not classifier accuracy.

## 2. Data and leakage discipline (fixed from v9; no new training)
* Frozen predictions and features of the v9 models: ConvNeXt-T (seeds 0-4), ResNet-50 (0-2), DINOv2-S frozen (0-2).
* Units: held-out EyePACS calibration patients only (5,269), never used in training or validation. Per run: half -> source pool split into disjoint C_gate and C_crc; the other half -> target clinic split into adaptation pool (gate half A, label half B) and test pool, exactly as in `e_v9.py` E1. Pool disjointness asserted in every run.
* 100 repetitions per seed: dependent re-partitions of fixed data, not independent cohorts.
* This is a **controlled deployment simulation within one network**, not a new real clinic.

## 3. Deployment grid (41 cells)
* Prevalence multiplier on referable units: 0.4x, 1x, 3x (class-weighted resampling of the same patients, so class-conditional scores are unchanged).
* Appearance: none, and blur / contrast loss / darkening / colour cast at levels 1, 2, 3 (identical parameters to v9; no tuning).
* Grid = 3 x 13 = 39 cells (Settings A: 1x none; B: 0.4x and 3x none; C: 1x with degradation; D: 0.4x or 3x with degradation).
* Plus 2 concept-shift cells (stricter referral standard, 1x, images unchanged) as the declared non-identifiability boundary: **the method is not expected to succeed there.**

## 4. Policies compared (all thresholds and gate parameters fixed from v9: BNULL 200, each test at 0.025, K_min = 114 / 45 / 22)
1. GATED (proposed): flag -> label-until-K target CRC (and exact-binomial variant); no flag -> source CRC.
2. SOURCE: always source CRC (zero labels).
3. ALWAYS_TARGET: always label-until-K, target CRC and exact binomial.
4. FIXED_BUDGET: always label the first 100 units of the label half (random order), target CRC and exact binomial on the referable units found (refer-all if infeasible).
5. GATED_SCORE_ONLY and GATED_FEATURE_ONLY (each test alone at 0.05).
6. ORACLE_DECISION (analysis only, not deployable): use the target branch iff source CRC's test miss rate in that run exceeds alpha.

## 5. Outcomes (hierarchy)
* **Primary:** test FNR at alpha = 0.05 (sensitivity analyses at 0.02 and 0.10), per cell and policy.
* **Co-primary (annotation):** rate of zero-label deployments; units inspected per deployment (mean, median, 90th percentile).
* **Secondary:** fraction of deployments in harmful cells not flagged, where a cell is **harmful** if SOURCE's mean FNR over all its runs exceeds alpha (defined from SOURCE before looking at GATED); total intervention workload (auto-referral + deferral, three-way gamma = 0.10); false automatic-referral rate among non-referable units; refer-all fallback frequency.

## 6. Statistics and reporting rules
* Seed-level means and t-intervals (n = 5 or 3 seeds); paired per-seed differences GATED minus each comparator. Repetitions are not treated as independent replicates.
* Equal annotation accounting: every policy's inspected units are counted, including zero.
* **All 41 cells are reported**, favourable or not; mixed-shift and near-boundary misses are reported in the main text.
* Success is defined as credible zero-label deployments in benign cells (1x none, 0.4x none, 3x none) without hiding losses in harmful cells. No percentage-of-wins target is set.
* Figures: (i) safety vs labels per cell, (ii) zero-label rate by shift type, (iii) paired label cost and miss vs SOURCE and ALWAYS_TARGET, (iv) heatmap of flag rate and GATED FNR over prevalence x appearance severity.

## 7. Decision rule
If the run reveals leakage, duplicated units across pools or code errors, it is stopped and reported as such; results are not cherry-picked or re-tuned. After this experiment, experimentation is frozen.
