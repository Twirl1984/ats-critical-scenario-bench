# Pre-registration: sensitivity of a learned sub-model vs. tolerable latency in the closed loop

Registered 29 Oct 2026, before any confirmatory run. Code: `src/sbench/hyp_slip.py` (this commit). Result file: `results_verify/hyp_slip.json` (later commit).

## Question

In the see–think–act chain only one sub-model is learned: the slip map ŝ(a, v) inside the ego-speed estimator (the plant keeps the physical slip). Does the **sensitivity** of the learned map, rather than its **accuracy**, determine how much **latency** the closed loop tolerates at the **worst admissible amplitude deviation**?

This links the five parts of the chain:

| Part | Role in the test |
| --- | --- |
| Controller | ACC + staged AEB on the RSS criterion, unchanged |
| Machine learning | 24 MLPs for ŝ(a, v), varied in regularisation, width and label noise |
| Optimisation | worst amplitude corner θ* of the tolerance box by single probes, then bisection for the largest tolerable added delay ΔT* |
| Scenarios | cut-in edge cases of the ODD variant `cut_in_odd` (valid, nominally passed, nominal g ≤ 1.5 s) |
| Deviations | amplitude (distance bias, wheel-speed scale, rate bias, IMU bias) and time (added latency) |

## Design (fixed)

- **Training data:** 400 points, a ∈ [−9.5, 2] m/s², v ∈ [0, 20] m/s, uniform, seed 0; label = physical slip + Gaussian noise σ.
- **Estimators:** `MLPRegressor(tanh, lbfgs, max_iter=3000)`, grid α ∈ {1e-6, 1e-4, 1e-2, 1} × hidden ∈ {(16,), (64, 64)} × σ ∈ {0, 0.002, 0.01}, random_state = index → K = 24. Each map is tabulated on a 0.05 m/s² × 0.125 m/s grid and evaluated bilinearly (same tabulation for the physical baseline).
- **Per estimator:** RMSE against the physical slip on a 61 × 41 test grid; sensitivity J = 95th percentile of |∂ŝ/∂a| on that grid (central differences).
- **Scenarios:** first N = 60 valid samples of `cut_in_odd` (prior, seed 21) with 0 < g ≤ 1.5 s under the physical estimator.
- **Per (estimator, scenario):** θ* from one +tolerance probe per amplitude parameter (tolerances `robustness.Tolerances`), then ΔT* = largest added delay k·DT, k ∈ {0, …, 50}, with g > 0 at θ*, found by bisection (ΔT* = 0 if g(θ*, 0) ≤ 0).
- **Outcome per estimator:** M = median ΔT* over the 60 scenarios.

## Hypotheses and tests (one-sided, α = 0.05, Holm correction over H1–H3)

- **H1:** Spearman ρ(J, M) < 0 across the 24 estimators. Permutation test, 20 000 permutations, seed 0.
- **H2 (sensitivity beyond accuracy):** partial Spearman ρ(J, M | RMSE) < 0: ranks of J and M are each regressed on the rank of RMSE and the residuals correlated; permutation of the J residuals, 20 000 permutations, seed 1.
- **H3 (paired):** ΔT* under the physical estimator is larger than under the estimator with the largest J. Wilcoxon signed-rank over the 60 scenarios (ties and zero differences: `zero_method="wilcox"`).

Effect sizes reported with them: ρ and partial ρ with a bootstrap 95 % CI over estimators (10 000 resamples, seed 2); median paired difference for H3.

## Rules

- One confirmatory run with exactly this configuration. No re-tuning of estimators, scenarios or grid after seeing results.
- Estimators that fit badly stay in the sample (they are part of the variation).
- Anything not listed above (flip rates in the tolerance box, other scenario types, correlation with RMSE alone) is **exploratory** and reported as such.
- A non-significant result is reported with the same prominence as a significant one.

## Limits known in advance

All plant, sensor and brake parameters are assumptions (see the main document). The test speaks about this model, not about a real shuttle. The latency resolution is DT = 0.02 s.
