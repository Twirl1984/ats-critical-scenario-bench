# Benchmark ats_ped:v=6.9,brake=6,a_e=4

Budget 300 Evaluationen, 8 Seeds, constrained=True, Laufzeit 110s.

| algo | found_valid_% | valid_fails | all_fails | implausible_% | unavoidable_% | first_valid_med | clusters | nn_dist | g_min | mean_nll | pf_valid | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 100.00 | 11.12 | 11.12 | 0.00 | 0.00 | 30.00 | 7.62 | 0.27 | -6.13 | 5.47 | – | 2.41 |
| grid | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | – | 0.00 | 0.00 | – | – | – | 2.56 |
| lhs | 100.00 | 10.25 | 10.25 | 0.00 | 0.00 | 21.00 | 6.25 | 0.26 | -6.15 | 4.90 | – | 2.36 |
| bayesopt | 100.00 | 183.38 | 183.38 | 0.00 | 0.00 | 24.00 | 28.00 | 0.10 | -7.04 | 29.07 | – | 12.77 |
| nsga2 | 100.00 | 69.62 | 69.62 | 0.00 | 0.00 | 30.00 | 2.25 | 0.01 | -6.06 | 8.97 | – | 1.92 |
| adv_pg | 100.00 | 87.38 | 87.38 | 0.00 | 0.00 | 35.50 | 12.75 | 0.14 | -6.66 | 1.29 | – | 1.55 |
| ce_is | 100.00 | 85.50 | 85.50 | 0.00 | 0.00 | 39.50 | 2.12 | 0.05 | -5.97 | -0.80 | 0.0086 | 1.71 |
| surrogate_ce | 100.00 | 200.38 | 200.38 | 0.00 | 0.00 | 39.50 | 2.00 | 0.05 | -6.44 | -1.10 | – | 1.68 |

**Lesehilfe:** *valid* = Fehler (g ≤ 0), der physikalisch plausibel (Zielverzögerung ≤ µ·g) und vom Oracle vermeidbar ist. `implausible_%`/`unavoidable_%` = Anteil der gefundenen Fehler, die das nicht sind. `clusters` = Anzahl Fehlermuster (Complete-Linkage, Schwelle 0.3). `pf_valid` = IS-Schätzung der Fehlerwahrscheinlichkeit unter dem Prior (nur ce_is).