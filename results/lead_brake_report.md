# Benchmark lead_brake

Budget 500 Evaluationen, 5 Seeds, constrained=False, Laufzeit 106s.

| algo | found_valid_% | valid_fails | all_fails | implausible_% | unavoidable_% | first_valid_med | clusters | nn_dist | g_min | mean_nll | pf_valid | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 60.00 | 1.20 | 121.20 | 68.63 | 69.05 | 54.00 | 0.80 | 0.08 | -3.59 | 3.34 | – | 1.65 |
| grid | 60.00 | 1.00 | 122.80 | 71.07 | 64.82 | 222.00 | 0.80 | 0.05 | -3.42 | 2.81 | – | 1.67 |
| lhs | 60.00 | 0.80 | 114.40 | 68.93 | 69.11 | 322.00 | 0.60 | 0.05 | -2.90 | 1.35 | – | 1.67 |
| bayesopt | 40.00 | 0.60 | 470.40 | 72.75 | 88.61 | 35.50 | 0.60 | 0.09 | -5.81 | 5.24 | – | 28.62 |
| nsga2 | 20.00 | 0.20 | 322.40 | 19.72 | 96.53 | 255.00 | 0.20 | 0.00 | -3.39 | 8.05 | – | 1.14 |
| adv_pg | 40.00 | 0.40 | 392.40 | 98.90 | 74.32 | 162.00 | 0.40 | 0.00 | -2.76 | 1.82 | – | 1.09 |
| ce_is | 40.00 | 5.60 | 117.80 | 54.11 | 42.50 | 171.00 | 0.40 | 0.01 | -3.35 | 0.25 | 3.1e-06 | 1.58 |
| surrogate_ce | 40.00 | 0.80 | 299.80 | 60.30 | 40.10 | 134.50 | 0.40 | 0.03 | -4.10 | 0.42 | – | 2.30 |

**Lesehilfe:** *valid* = Fehler (g ≤ 0), der physikalisch plausibel (Zielverzögerung ≤ µ·g) und vom Oracle vermeidbar ist. `implausible_%`/`unavoidable_%` = Anteil der gefundenen Fehler, die das nicht sind. `clusters` = Anzahl Fehlermuster (Complete-Linkage, Schwelle 0.3). `pf_valid` = IS-Schätzung der Fehlerwahrscheinlichkeit unter dem Prior (nur ce_is).