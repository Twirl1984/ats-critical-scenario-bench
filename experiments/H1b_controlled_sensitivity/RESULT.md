# H1b result: not confirmed — sensitivity of the learned sub-model is not the bottleneck here

Confirmatory run 29 Oct 2026, 24 perturbed slip maps (RMSE ≈ 0.002 each) × the 60 cut-in edge scenarios of H1.

| Test | Statistic | p (one-sided) | Holm | H0 rejected |
| --- | --- | --- | --- | --- |
| H1b-1 ρ(J, mean ΔT*) < 0 | ρ = +0.03, 95 % CI [−0.41, 0.48] | 0.56 | 0.025 | no |
| H1b-2 ΔT* physical > ΔT* at ω = 16, φ = 0 | mean paired difference −0.005 s | 0.9994 | 0.05 | no |

Mean tolerable added latency by ω (physical map: 0.393 s):

| ω | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J (approx.) | 0.028 | 0.028 | 0.028 | 0.029 | 0.033 | 0.042 | 0.050 | 0.055 |
| mean ΔT* [s] | 0.393 | 0.393 | 0.393 | 0.394 | 0.394 | 0.400 | 0.402 | 0.407 |

**Reading.** Doubling the slope sensitivity of the learned slip map at fixed accuracy does not reduce the latency the closed loop tolerates at the worst amplitude corner. With the assumed tolerances (IMU bias ±0.2 m/s²) the extra sensitivity changes the slip estimate by a few thousandths, i.e. well under one percent in speed, while the latency budget is set by sensing and actuation timing. This is consistent with the earlier finding that latency dominates 85–98 % of the robust failures. [the mechanism is an interpretation, not tested]

**What follows.** The question becomes *where* the learned part turns binding: an exploratory sweep over the IMU-bias tolerance and over the sensitivity range would locate the threshold; any confirmatory claim about it needs its own registration.
