# Research Log

---

## WEEK 1 — Project Bootstrapping & Design Lock-In

1. Set up the development environment and tools for the project.
   - Python 3.13, virtual environment, dependencies: `numpy ≥ 1.22`, `pandas ≥ 1.3`, `matplotlib ≥ 3.4`, `scikit-learn ≥ 1.5`, `scipy ≥ 1.7`.
2. Create a skeleton structure for the project, including folders for data, scripts, and documentation.
   - Established `src/data`, `src/models`, `src/policies`, `src/runner`, `src/evaluation`, `docs/`, `results/`.
3. Create a skeleton for the paper.
4. Lock my drift modules and policy parameters.
   - Drift parameters finalized: 10 features, 10,000 samples, drift at t = 5000, recurrence period = 1000.
   - `DriftGenerator` class implemented with abrupt, gradual, and recurring modes using weight-vector switching.
   - `StreamingModel` wrapping `SGDClassifier(loss="log_loss")` with `partial_fit()` + `retrain()` methods.
   - `RetrainPolicy` base class with budget decrement and latency-guard (`is_in_latency_period`) logic.
   - Full factorial design scoped: 3 drift × 3 policy × 3 budget × 3 latency × 3 seeds = 243 runs.

---

## WEEK 2 — Periodic Policy Experiments (Abrupt & Gradual Drift)

1. Run preliminary experiments to test the drift simulation and retraining policies.
   - Verified that `DriftGenerator` produces deterministic output for seeds 42, 123, 456.
   - Confirmed `ExperimentRunner` streaming loop: predict → evaluate → buffer → maybe retrain → partial_fit.
2. Run abrupt drift with periodic retraining with various budgets and latency levels.
   - 27 runs (3 budgets × 3 latencies × 3 seeds).
   - Observation: periodic policy achieves 100 % budget utilization at low/medium latency but only ~50 % at high latency with K = 20 because the 520-step latency window exceeds the 500-step interval, halving usable retrains from 20 to ~10.
3. Run gradual drift with periodic retraining with various budgets and latency levels.
   - 27 runs.
   - Observation: accuracy drop under gradual drift is comparable to abrupt drift (≈ −0.03 to −0.05) because the 1,000-step transition still fully shifts the concept within two periodic intervals.
4. Analyze and document the results of the preliminary experiments, noting any issues or insights.
   - Key insight: pre-drift accuracy varies significantly across seeds (e.g., seed 42 ≈ 0.77, seed 123 ≈ 0.71, seed 456 ≈ 0.82), confirming the importance of multi-seed evaluation.
   - Periodic retrains fire evenly across the stream — roughly half before drift and half after, regardless of drift type.

---

## WEEK 3 — Periodic Policy Experiments (Recurring Drift) & Paper Drafting

1. Run recurring drift with periodic retraining with various budgets and latency levels.
   - Final 27 runs for periodic policy. Summary CSV now contains all 81 rows.
   - Observation: recurring drift creates multiple accuracy dips (one per concept switch every 1,000 steps), and the periodic policy can only coincidentally align a retrain with a switch.
2. Analyze and document the results of the experiments, noting any issues or insights.
   - Under recurring drift with high latency, the model spends a large fraction of post-drift time inside latency windows, meaning stale weights dominate predictions.
   - Budget utilization remains 100 % at low/medium latency for all budget levels, confirming the latency–budget interaction only bites at high latency.
3. Drafted up an intro and related work sections for the paper.
4. Start drafting the methods section of the paper, detailing the experimental setup and parameters.

---

## WEEK 4 — Error-Threshold Policy Calibration & Full Runs

1. Did a threshold calibration for error threshold policy and documented it.
   - Swept thresholds in [0.20, 0.25, 0.27, 0.30, 0.35] with window = 200 over seed 42 + abrupt drift.
   - Threshold = 0.20 → fires 8 times pre-drift (pure noise triggers, budget wasted).
   - Threshold = 0.27 → fires 5 times pre-drift for seeds 42/123, 0 times for seed 456 — selected as best trade-off.
   - Threshold = 0.35 → never fires pre-drift, but delays detection by ~300 steps post-drift.
2. Ran experiments with error threshold retraining policy across all drift types, budgets, and latency levels.
   - 81 runs total (3 drift × 3 budget × 3 latency × 3 seeds).
   - Critical finding (abrupt drift): for seeds 42 and 123 with K = 5, **all 5 retrains fired before t = 5000**, leaving 0 retrains for the actual drift. This means the model ran entirely on stale-or-incremental weights post-drift.
   - For seed 456, all retrains fired post-drift (0 before, 5 after) — the error threshold was not exceeded by pre-drift noise for this seed, showcasing high seed sensitivity.
3. Analyzed the results of the experiments, comparing the performance of the error threshold policy under various conditions.
   - Budget utilization: 100 % for low/medium latency at all budget levels. At high latency with K = 5, seed 456 achieved only 80 % (4 out of 5 retrains) because the 520-step latency window blocked the final trigger.
   - Post-drift accuracy: comparable across policies (~0.73–0.79 depending on seed), suggesting that the error-threshold policy did not dramatically improve post-drift performance when budget was wasted pre-drift.

---

## WEEK 5 — ADWIN Calibration & Paper Sections

1. Did a threshold calibration for drift triggered policy and documented it.
   - Swept δ ∈ {0.05, 0.01, 0.005, 0.002, 0.001} with window = 500 and min_samples = 300.
   - δ = 0.05 → fires ~3 times pre-drift (false alarms) on some seeds.
   - δ = 0.002 → 0 false alarms across all seeds/drift types during the pre-drift phase. Selected as final value.
   - δ = 0.001 → also 0 false alarms, but sometimes missed abrupt drift entirely for seeds 42/123 (too conservative).
   - min_samples = 300 was chosen to skip the warm-up noise in the first 300 steps where `partial_fit` is still converging.
2. Worked on drafting the sections of paper related to the error threshold policy, including the experimental setup and discussion.

---

## WEEK 6 — Full ADWIN Runs & Cross-Policy Analysis

1. Continued to run experiments for error threshold and drift trigger policies across all drift types, budgets, and latency levels.
   - All 81 drift-triggered runs completed. Summary CSV: `summary_results_drift_triggered_retrain_3seed.csv` (81 rows).
   - All three policy summary CSVs verified: 82 lines each (1 header + 81 data rows).
2. Analyzed the results of the experiments, comparing the performance of different policies under various conditions.

---

## WEEK 7 — Extended 10-Seed Runs (Phase 2)

---

1. Extended the seed set from 3 to 10 seeds: `[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]`.
2. Ran all 810 experiments (10 seeds × 3 drifts × 3 budgets × 3 latencies × 3 policies) in **3 batches of 270 runs**, one batch per policy:
   - **Batch 1 — Periodic policy (270 runs):** Branch `develop-10Seed-periodic-retrain-tests`
   - **Batch 2 — Error-Threshold policy (270 runs):** Branch `develop-10Seed-error-threshold-retrain-tests`
   - **Batch 3 — Drift-Triggered ADWIN policy (270 runs):** Branch `develop-10Seed-drift-triggered-retrain-tests`
3. Implemented `NeverRetrainPolicy` in `src/policies/never_retrain_policy.py` — inherits from `RetrainPolicy` with budget=0, latency=0, `should_retrain()` always returns `False`.
4. Added `_run_no_retrain_sweep()` to `main.py` — a dedicated sweep function with a simplified grid (3 drift types × N seeds, no budget/latency loops).
5. Added `plot_summary_for_no_retrain()` to `plot_summary.py` — a 2×2 baseline-specific dashboard (mean accuracy, pre/post drift, accuracy drop, box plot).
6. Ran baseline with **3 seeds** (9 runs) and **10 seeds** (30 runs)

---

## WEEK 8 — Extreme Latency Experiments (Phase 3)

1. Added two extreme latency levels that bracket the original range to stress-test policy behavior at the boundary conditions:
2. Ran full sweeps for all 4 policies with both seed sets
3. Investigated the IEEE-CIS Fraud Detection dataset (590,540 rows, 428 features, 3.5% fraud) as a real-world validation dataset. Built `FraudDataLoader`, `RealDriftGenerator`, calibration and diagnosis tooling. 
4. Calibrated policy hyperparameters on fraud data and confirmed feature-level drift via KS tests (190/428 features shifted at p<0.001 at the midpoint split). 
5. Ran streaming sanity checks at three temporal offsets (0, 20,000, 40,000). Only offset=20,000 showed mild F1 degradation post-drift (−0.003); the other two showed flat or improving F1. Accuracy was uninformative at all offsets due to 97% majority class.
6. **Discarded the dataset** — could not find 3 consistent seed offsets where drift degrades model performance, which is the minimum requirement for the factorial experiment design. Feature-level shift does not translate to task-level degradation on this data.

---

## WEEK 9 — Real-World Datasets (Phase 4)

1. Investigated the **LUFlow Network Intrusion Detection dataset** (Lancaster University, 28 day-CSVs, ~21 M rows, 11 flow-level features, binary: benign vs malicious) as a real-world validation dataset.
2. Built `luflow_fitness_check.py` — a three-gate suitability check and designed three **pool-pair configurations** ("seeds") for the factorial experiment
3. **Calibrated policy hyperparameters** on LUFlow abrupt condition nad built `luflow_main.py` — a dedicated experiment runner that mirrors the synthetic `main.py` architecture but handles LUFlow data loading, pool-based stream construction, and `StandardScaler` feature normalization. 
4. Ran the full **Phase 4 sweep**: 3 pool configs × 3 drift types × 3 budgets × 3 latencies × 3 active policies = 243 active runs + 9 no-retrain baseline runs = **252 total runs**.
5. Generated summary dashboards for all 4 policies on LUFlow data. Merged into `main`. 
6. Investigated the **Kelmarsh Wind-Farm SCADA dataset** (6 Senvion MM92 turbines, 2019–2022, 10-min resolution, ~52 K rows/turbine/year) as a second real-world validation dataset. Built `kelmarsh_loader.py` (IEC 61400-26 state labels, permissive NaN fill, multi-turbine aggregation), ran diagnostics, and built `kelmarsh_fitness_check.py` + `kelmarsh_main.py` for a 252-run factorial sweep across 3 temporal seed configs. 
7. Hit a **structural limitation**: years with enough positive signal (2021–2022) are too temporally close for multi-seed drift contrasts, while years with sufficient spread (2019 → 2022) lack positives in the early years. Exhausted six label strategies (fault windows at 6 sizes, broad faults, IEC states, single/multi-year pools, aggregated turbines) — all fail Gate 1 or compress the temporal window unacceptably.
8. **Discarded the dataset** — Kelmarsh is unsuitable for the factorial design.
9. Investigated the **Pump Sensor Data** dataset (50,000 samples, multivariate sensor readings, binary classification) as a third real-world validation candidate. Built a data loader and ran the **no-retrain baseline** sweep: 3 seeds × 3 drift types = 9 runs with drift point at t = 25,000.
10. Hit a **fundamental suitability failure**: the SGDClassifier achieves near-perfect accuracy (~99.97–99.99 %) on this task both before and after drift injection. The maximum observed accuracy drop across all 9 runs was −0.00044 (0.044 %), and one recurring-drift run actually showed a *positive* accuracy change (+0.00008). Pre-drift accuracy sat at 0.999–1.000 across all seeds, leaving essentially zero headroom for degradation.
11. **Discarded the dataset** — because the classification task is trivially separable for the streaming model, concept drift does not translate into meaningful performance loss. Retraining policies have nothing to recover, so running the remaining 243 active-policy experiments would produce flat, uninformative results.

---

## WEEK 10 — LendingClub Real-World Dataset (Phase 5)

1. Investigated the **LendingClub Loan Default dataset** (Kaggle, accepted loans 2007–2018, ~2.26 M rows raw, ~1.35 M after filtering to Fully Paid / Charged Off, 16 origination-time features → 34 after one-hot encoding) as a second real-world validation dataset.
2. Built `lendingclub_loader.py` — origination-time-only feature selection (no post-origination leakage), `issue_d` parsing for year-cohort extraction, `emp_length` numeric conversion, one-hot encoding of `home_ownership` and `purpose`, and `get_year_cohort()` with shuffled sampling via `rng.choice`.
3. Built `lendingclub_fitness_check.py` — a three-gate suitability check and designed three **year-pair configurations** ("seeds") with per-seed `random_state` for shuffled sub-sampling.
4. **Calibrated policy hyperparameters** on LendingClub abrupt conditions and built `lendingclub_main.py` — a dedicated experiment runner that mirrors the synthetic and LUFlow architectures but handles LendingClub data loading, year-pair pool selection, and `StandardScaler` feature normalization.
5. Ran the full **Phase 5 sweep**: 3 seeds × 3 drift types × 3 budgets × 3 latencies × 3 active policies = 243 active runs + 9 no-retrain baseline runs = **252 total runs**. 
6. Generated summary dashboards for all 4 policies on LendingClub data. Merged into `main`.

---

