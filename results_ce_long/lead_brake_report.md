# Benchmark lead_brake

Budget 4000 Evaluationen, 8 Seeds, constrained=True, Laufzeit 54s.

| algo | found_valid_% | valid_fails | all_fails | implausible_% | unavoidable_% | first_valid_med | clusters | nn_dist | g_min | mean_nll | pf_valid | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ce_is | 100.00 | 10.62 | 10.62 | 0.00 | 0.00 | 321.00 | 1.38 | 0.04 | -4.60 | 0.59 | 0.00042 | 13.31 |

**Lesehilfe:** *valid* = Fehler (g ≤ 0), der physikalisch plausibel (Zielverzögerung ≤ µ·g) und vom Oracle vermeidbar ist. `implausible_%`/`unavoidable_%` = Anteil der gefundenen Fehler, die das nicht sind. `clusters` = Anzahl Fehlermuster (Complete-Linkage, Schwelle 0.3). `pf_valid` = IS-Schätzung der Fehlerwahrscheinlichkeit unter dem Prior (nur ce_is).