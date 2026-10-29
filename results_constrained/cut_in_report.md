# Benchmark cut_in

Budget 500 Evaluationen, 5 Seeds, constrained=True, Laufzeit 98s.

| algo | found_valid_% | valid_fails | all_fails | implausible_% | unavoidable_% | first_valid_med | clusters | nn_dist | g_min | mean_nll | pf_valid | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 100.00 | 6.00 | 6.00 | 0.00 | 0.00 | 109.00 | 4.40 | 0.28 | -6.87 | 1.68 | – | 1.36 |
| grid | 100.00 | 4.20 | 4.20 | 0.00 | 0.00 | 43.00 | 3.00 | 0.28 | -5.49 | 0.58 | – | 1.36 |
| lhs | 100.00 | 6.80 | 6.80 | 0.00 | 0.00 | 28.00 | 3.40 | 0.20 | -7.02 | 0.88 | – | 1.33 |
| bayesopt | 100.00 | 227.40 | 227.40 | 0.00 | 0.00 | 61.00 | 2.60 | 0.02 | -10.57 | 14.93 | – | 28.65 |
| nsga2 | 80.00 | 32.40 | 32.40 | 0.00 | 0.00 | 257.50 | 1.20 | 0.0068 | -6.01 | -0.46 | – | 1.77 |
| adv_pg | 80.00 | 2.40 | 2.40 | 0.00 | 0.00 | 46.00 | 1.00 | 0.13 | -4.05 | -1.15 | – | 0.93 |
| ce_is | 100.00 | 15.20 | 15.20 | 0.00 | 0.00 | 41.00 | 3.20 | 0.14 | -6.61 | -2.10 | 0.03 | 1.19 |
| surrogate_ce | 100.00 | 17.60 | 17.60 | 0.00 | 0.00 | 41.00 | 2.60 | 0.11 | -8.07 | -1.80 | – | 1.96 |

**Lesehilfe:** *valid* = Fehler (g ≤ 0), der physikalisch plausibel (Zielverzögerung ≤ µ·g) und vom Oracle vermeidbar ist. `implausible_%`/`unavoidable_%` = Anteil der gefundenen Fehler, die das nicht sind. `clusters` = Anzahl Fehlermuster (Complete-Linkage, Schwelle 0.3). `pf_valid` = IS-Schätzung der Fehlerwahrscheinlichkeit unter dem Prior (nur ce_is).