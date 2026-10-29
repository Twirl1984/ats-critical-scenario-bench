# Pre-registration H1b: controlled sensitivity at fixed accuracy

Registered 29 Oct 2026 after H1 (not confirmed, see `../H1_slip_sensitivity/RESULT.md`) and **before any H1b run**.
Code: `src/sbench/hyp_slip.py`, function `run_h1b` (this commit). Result: `result.json` (later commit).

## Why a follow-up

H1 varied regularisation, width and label noise; all networks learned nearly the physical slope, so sensitivity hardly varied and was confounded with bias. H1b varies sensitivity directly while holding accuracy fixed.

## Design (fixed)

- **Estimators:** ŝ(a, v) = s(a, v) + ε·sin(ω·a + φ) in the speed estimator only (the plant keeps s). ε = 0.002·√2 so that the RMSE is ≈ 0.002 for all ω (same order as the H1 networks). ω ∈ {0.25, 0.5, 1, 2, 4, 8, 12, 16} rad per m/s² × φ ∈ {0, 2π/3, 4π/3} → K = 24. Tabulated and evaluated exactly as in H1 (period at ω = 16 ≈ 8 grid points).
- **Metrics:** RMSE and J (95th percentile of |∂ŝ/∂a|) as in H1.
- **Scenarios, optimisation, ΔT*:** identical to H1 (same 60 cut-in edge scenarios, probe corner θ*, bisection of added delay).
- **Primary outcome per estimator:** mean ΔT* over the 60 scenarios (finer than the median, which tied in H1; declared here before the run).

## Hypotheses (one-sided, α = 0.05, Holm over H1b-1 and H1b-2)

- **H1b-1:** Spearman ρ(J, mean ΔT*) < 0 across the 24 estimators; permutation test, 20 000 permutations, seed 0.
- **H1b-2:** ΔT* under the physical estimator is larger than under ω = 16, φ = 0; Wilcoxon signed-rank over the 60 scenarios.

Reported alongside: ρ with bootstrap 95 % CI (10 000, seed 2), mean ΔT* per ω, Spearman of RMSE with mean ΔT* (exploratory).

## Rules

As in H1: one confirmatory run, no re-tuning, non-significant results reported with equal prominence.
