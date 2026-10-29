# Benchmark lead_brake

Budget 500 Evaluationen, 5 Seeds, constrained=True, Laufzeit 112s.

| algo | found_valid_% | valid_fails | all_fails | implausible_% | unavoidable_% | first_valid_med | clusters | nn_dist | g_min | mean_nll | pf_valid | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 60.00 | 1.20 | 1.20 | 0.00 | 0.00 | 54.00 | 0.80 | 0.08 | -3.59 | 3.34 | – | 1.81 |
| grid | 60.00 | 1.00 | 1.00 | 0.00 | 0.00 | 222.00 | 0.80 | 0.05 | -3.42 | 2.81 | – | 1.82 |
| lhs | 60.00 | 0.80 | 0.80 | 0.00 | 0.00 | 322.00 | 0.60 | 0.05 | -2.90 | 1.35 | – | 1.78 |
| bayesopt | 80.00 | 183.20 | 183.20 | 0.00 | 0.00 | 110.00 | 1.20 | 0.01 | -9.82 | 21.84 | – | 29.22 |
| nsga2 | 60.00 | 26.60 | 26.60 | 0.00 | 0.00 | 180.00 | 0.60 | 0.005 | -5.68 | 2.47 | – | 2.17 |
| adv_pg | 40.00 | 0.40 | 0.40 | 0.00 | 0.00 | 393.00 | 0.40 | 0.00 | -1.57 | 1.03 | – | 2.10 |
| ce_is | 60.00 | 32.40 | 32.40 | 0.00 | 0.00 | 87.00 | 0.60 | 0.02 | -4.98 | 0.60 | 1.1e-05 | 1.86 |
| surrogate_ce | 40.00 | 108.40 | 108.40 | 0.00 | 0.00 | 70.50 | 0.40 | 0.0077 | -6.35 | 0.82 | – | 3.25 |

**Lesehilfe:** *valid* = Fehler (g ≤ 0), der physikalisch plausibel (Zielverzögerung ≤ µ·g) und vom Oracle vermeidbar ist. `implausible_%`/`unavoidable_%` = Anteil der gefundenen Fehler, die das nicht sind. `clusters` = Anzahl Fehlermuster (Complete-Linkage, Schwelle 0.3). `pf_valid` = IS-Schätzung der Fehlerwahrscheinlichkeit unter dem Prior (nur ce_is).