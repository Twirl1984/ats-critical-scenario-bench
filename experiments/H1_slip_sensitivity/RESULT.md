# H1 result: not confirmed (design without leverage)

Confirmatory run 29 Oct 2026, 24 estimators × 60 cut-in edge scenarios, versions in `result.json`.

| Test | Statistic | p (one-sided) | Holm | H0 rejected |
| --- | --- | --- | --- | --- |
| H1 ρ(J, M) < 0 | ρ = +0.71, 95 % CI [0.35, 0.87] | 0.99995 | 0.025 | no |
| H2 partial ρ(J, M \| RMSE) < 0 | +0.43, 95 % CI [0.01, 0.74] | 0.980 | 0.017 | no |
| H3 ΔT* physical > ΔT* of max-J estimator (mlp08) | median paired difference 0.00 s, 29 of 60 non-zero | 0.99999 | 0.05 | no |

**Reading.** None of the hypotheses is supported; the observed association even points the other way. The design is the reason: all 24 networks learned almost the physical slope (J between 0.022 and 0.029 against 0.027 for the physical map). The only real variation comes from the strongly regularised networks (α = 1): flatter (smaller J) *and* less accurate (RMSE ≈ 8·10⁻³), with slightly smaller tolerable latency (M = 0.33 s against 0.35 s). Sensitivity and bias are confounded there, so the run cannot separate them.

**Consequence.** The question stays open. A follow-up with controlled, accuracy-matched sensitivity is registered separately as H1b (`../H1b_controlled_sensitivity/`). H1 is not re-run or re-tuned.
