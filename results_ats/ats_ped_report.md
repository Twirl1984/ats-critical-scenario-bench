# Benchmark ats_ped

Budget 300 Evaluationen, 5 Seeds, constrained=True, Laufzeit 64s.

| algo | found_valid_% | valid_fails | all_fails | implausible_% | unavoidable_% | first_valid_med | clusters | nn_dist | g_min | mean_nll | pf_valid | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 100.00 | 22.40 | 22.40 | 0.00 | 0.00 | 15.00 | 12.40 | 0.24 | -10.24 | 3.02 | – | 2.12 |
| grid | 100.00 | 36.20 | 36.20 | 0.00 | 0.00 | 3.00 | 9.60 | 0.18 | -6.65 | 0.27 | – | 2.28 |
| lhs | 100.00 | 18.20 | 18.20 | 0.00 | 0.00 | 19.00 | 10.00 | 0.26 | -9.65 | 3.16 | – | 2.15 |
| bayesopt | 80.00 | 159.80 | 159.80 | 0.00 | 0.00 | 25.50 | 13.00 | 0.05 | -10.73 | 28.90 | – | 11.95 |
| nsga2 | 100.00 | 105.00 | 105.00 | 0.00 | 0.00 | 15.00 | 3.20 | 0.01 | -9.64 | 3.81 | – | 1.75 |
| adv_pg | 100.00 | 80.00 | 80.00 | 0.00 | 0.00 | 32.00 | 12.60 | 0.15 | -9.88 | -0.77 | – | 1.41 |
| ce_is | 100.00 | 90.40 | 90.40 | 0.00 | 0.00 | 30.00 | 4.00 | 0.06 | -9.93 | -2.33 | 0.04 | 1.64 |
| surrogate_ce | 100.00 | 211.80 | 211.80 | 0.00 | 0.00 | 30.00 | 3.60 | 0.05 | -9.99 | -2.34 | – | 1.61 |

**Lesehilfe:** *valid* = Fehler (g ≤ 0), der physikalisch plausibel (Zielverzögerung ≤ µ·g) und vom Oracle vermeidbar ist. `implausible_%`/`unavoidable_%` = Anteil der gefundenen Fehler, die das nicht sind. `clusters` = Anzahl Fehlermuster (Complete-Linkage, Schwelle 0.3). `pf_valid` = IS-Schätzung der Fehlerwahrscheinlichkeit unter dem Prior (nur ce_is).