# Critical-scenario search for an L4 shuttle (ATS): see–think–act benchmark

**Status 29 Oct 2026 · research prototype · all sensor, ODD and brake parameters are assumptions, not data of any real vehicle.**

This repository contains `sbench`, a small Python benchmark that

1. derives a search space from an ODD description (scenario parameters × environment × admissible sensor/timing deviations),
2. predicts the worst-case corner of the tolerance box from a few model evaluations (sensitivity of the full chain) instead of brute force,
3. compares search algorithms for critical scenarios (random, LHS, Bayesian optimisation, NSGA-II, policy-gradient, cross-entropy importance sampling, surrogate) with plausibility and avoidability checks, and
4. classifies every finding as `robust_fail`, `fragile` or `robust_pass` under measurement deviation and latency.

The system model is a see–think–act chain: phenomenological sensors (radar, camera, lidar) → world model with age limit → RSS/AEB decision → vehicle with actuator delay (1-D longitudinal).

| Document | Content |
|---|---|
| [docs/Kritische_Szenarien_L4-Shuttle.md](docs/Kritische_Szenarien_L4-Shuttle.md) | Concept, equations, results, gap analysis, open mathematical questions (German) |
| [docs/ERGEBNISSE.md](docs/ERGEBNISSE.md) | Sensor-suite coverage study and back-calculation of requirements (German) |
| [results_suite/Sensormodelle_Gesamtkette_Abdeckung.xlsx](results_suite/Sensormodelle_Gesamtkette_Abdeckung.xlsx) | Coverage, traceability to UN R157, assumptions, gaps to a real system |

## Repository layout: code tests and experiments are separate

| Folder | Purpose | Run |
|---|---|---|
| `tests/` | **Code verification:** unit, golden regression, property-based fuzzing (Hypothesis, derandomised), differential and metamorphic tests | `pytest` |
| `verification/` | Code verification incl. **mutation testing** (mutmut) and a deterministic report `REPORT.md` | `verification/run.sh` |
| `experiments/` | **Scientific claims about the model:** pre-registered hypothesis tests with their result files; each folder holds `PREREGISTRATION.md` (committed before the run) and `RESULT.md` | `python -m sbench.hyp_slip [--h1b]` |
| `results_*` | Study outputs referenced by the documents | see the document |

Experiments so far: **H1** and **H1b** (sensitivity of a learned slip sub-model vs. tolerable latency) — both *not confirmed*; H1b shows that at the assumed tolerances the learned part is not the bottleneck. See `experiments/*/RESULT.md` and `ROADMAP.md`.

**Main results so far (under the stated assumptions):** a 7-evaluation probe predicts the worst tolerance corner and reproduced every brute-force flip in the test set (see the document for numbers and limits); Bayesian optimisation finds the most distinct failure clusters at 300 evaluations; latency dominates the robust failures.
**Known limitation that matters most:** the avoidability oracle brakes at µ·g, not at the platform limit; the document quantifies the effect.

```
pip install -e ".[plot,dev]"
pytest
python -m sbench.probe_study          # worst-corner probe test (approx. 10–20 min)
```

Author: Dr.-Ing. Christoph Funda. Licence: MIT (see `LICENSE`).

---

# sbench – Benchmark für Algorithmen zur Suche kritischer Szenarien

Leichtgewichtiges Python-Framework, um Szenario-Suchalgorithmen fair zu vergleichen **und** die
Robustheit der gefundenen Fehler gegen Messabweichung und Zeitversatz zu prüfen.

```
pip install -e ".[plot,dev]"
sbench run --problem lead_brake --algos all --budget 500 --seeds 5 --constrained
sbench reference --problem lead_brake --n 20000          # Referenz-P_f unter dem Prior
sbench robustness results/lead_brake_run.json --top 20   # Robustheit gefundener Fehler
pytest
```

## Architektur

| Modul | Inhalt |
|---|---|
| `sim.py` | Längsdynamik + ACC/AEB (RSS-Kriterium), Sensorpfad mit `slip = f(a, v)`, `SensorConfig` (Bias, Skalierung, Rauschen, Latenz) |
| `problems.py` | `lead_brake` (6D), `cut_in` (7D), `toy_ball` (Test). Prior = Produkt aus Betas → `nll` (Unwahrscheinlichkeit) und Importance Sampling |
| `algos/` | `random`, `grid`, `lhs`, `bayesopt` (GP+EI), `nsga2` (g vs. nll), `adv_pg` (REINFORCE), `ce_is` (Cross-Entropy-IS, unverzerrter P_f-Schätzer), `surrogate_ce` (RF-Screening) |
| `metrics.py` | Kriterialität, Diversität (Cluster, NN-Distanz), Plausibilität/Vermeidbarkeit, Effizienz, P_f |
| `robustness.py` | Jacobi/Linearisierung, Monte-Carlo, Zeit-/Amplitudenschlauch, Szenario-Verdikt unter Sensorabweichung |
| `runner.py`, `cli.py` | Algorithmen × Seeds parallel, JSON/CSV/Markdown/PNG |

**Zielfunktion** `g(u)`: Kollision → `-(Aufprallgeschwindigkeit+0.1)`, sonst minimale TTC (Cap 10 s). Fehler ⇔ `g ≤ 0`.

**Gültiger Fehler** = Zielverzögerung ≤ µ·g (plausibel) **und** vom Oracle (perfekte Sensorik, Vollbremsung ab
Ereignis) vermeidbar. Ohne diese Prüfung „gewinnen" Algorithmen mit physikalisch unmöglichen Szenarien.
`--constrained`: ungültige Fehler gelten im Suchsignal als Beinahe-Treffer (`g=0.05`).

## Mapping der Notizen → Code

| Notiz | Code |
|---|---|
| `f(w, v, s, a, slip=f(a,v))` | `sim.decision_signal(w, vr, s, a)`; v_ego aus Raddrehzahl mit Schlupfkompensation (`ego_speed_from_wheel`) |
| `Δ = ∂f/∂w Δw + …` | `robustness.jacobian`, `linear_bounds` (Worst-Case und RSS) |
| `f_mess` / `f_theor` mit `±Δ` | `tube_required_dF`, `tube_required_dT`, `tube_tradeoff` |
| `ΔT, Δf → Zeit mit Amplitude als Hilfe` | `tube_tradeoff`: nötiges Δf je ΔT (Zeitversatz „kauft" Amplitudentoleranz) |
| Messabweichung im Regelkreis | `scenario_robustness`: 32 Toleranzecken + Monte-Carlo + dg/dθ je Sensor |
| `Szenario-Likelihood + Severity + Impact` | `nll` (Prior), `g`/Aufprallgeschwindigkeit, NSGA-II über (g, nll) |

> Annahme: `v` in der Notiz = relative Geschwindigkeit zum Ziel (`vr`); Ego-Geschwindigkeit kommt aus `w`.
> Falls anders gemeint: `sim.decision_signal` anpassen, der Rest bleibt unverändert.

## Bekannte Grenzen

- Sim und Sensormodell sind Platzhalter (kinematisch, 1D). Für SafeBench/CARLA: `Problem`-Unterklasse mit
  `decode`/`evaluate`, die Algorithmen bleiben unverändert.
- `ce_is` hat bei seltenen gültigen Fehlern (P_f ≈ 5e-4) erst ab ca. 4000 Evaluationen einen brauchbaren
  Schätzer (siehe `results_ce_long/`); bei 500 Evaluationen dominiert Rauschen.
- Toleranzen in `robustness.Tolerances` sind Platzhalter, an Datenblätter anpassen.

## See-Think-Act-Kette mit Weltmodell (`chain.py`, `world.py`)

`world.py` ist ein 1:1-Port des Weltmodells `world.rs` aus einem (nicht öffentlichen) Rust-See-Think-Act-Stack (Empfangszeitstempel je Quelle,
Altersgrenze `max_age`, konservatives Minimum, „keine Messung ≠ freier Weg"). `chain.py` verbindet Sensoren
(einfach: Bias/Rauschen/Latenz, Fehlerinjektion `stall`/`repeat`) → Weltmodell → RSS/AEB → Fahrzeug.
Probleme: `lead_brake_chain`, `chain_a02|a05|a10` (max_age 0.2/0.5/1.0 s), 10D inkl. Einfrier-Parametern.

    sbench staleness --n 4000 --out results_chain   # max_age vs. gültige Kollisionen / Blindzeit

Sensormodelle (Radar/Lidar/Kamera, FAU/fiveD, OSI/FMU) werden später hinter dieselben Nachrichten
(`LidarScan`, `RangeArray`) gehängt. Tests: `pytest`.

## Sensor-Suite (Stufe 1), Abdeckung und Traceability

    sbench suitestudy --n 800     # Szenario x Sensor-Suite x Umgebung, Ablation, Phantombremsen
    sbench trace --n 250          # R157-Traceability + Rückrechnung (ΔT*, Δs*, R*)
    python -m sbench.export_xlsx <Sensormodelle_Gesamtkette.xlsx> out.xlsx

* `suite.py` – phänomenologische Sensoren (Radar/Kamera/Lidar), Umgebungen, Gesamtkette inkl. Weltmodell
* `suite_study.py`, `budget.py`, `trace_study.py`, `export_xlsx.py`
* `data/r157_alks.json` – strukturierte Anforderungen aus UN R157 (ALKS), übernommen aus einem nicht öffentlichen Compliance-Projekt; Zuordnung Anforderung → Szenario ist Ingenieursentscheidung
* Ergebnisse und Grenzen: `docs/ERGEBNISSE.md`. Alle Sensorparameter sind Platzhalter.

## ATS (autonomes Shuttle, Level 4) als Gesamtsystem-Suchproblem

    sbench run --problem ats_ped --budget 300 --seeds 8 --constrained     # 12D: Szenario + Umgebung + Messabweichung
    sbench run --problem "ats_ped:v=6.9,brake=6,a_e=4"                     # ODD-/Systemvariante (v in m/s, brake = max. Verzögerung)
    sbench reference --problem ats_ped --n 40000                           # MC-Referenz für P_f
    sbench atsreport results_ats/ats_ped_run.json                          # Charakterisierung + Robustheitsverdikt
    sbench devimpact --problem ats_ped                                     # Anteil der Fehlerrate durch zulässige Abweichung
    python -m sbench.worstcase ats_ped 3000                                # Vorabschätzung Worst-Case-Ecke vs. 32 Ecken
    python -m sbench.probe_study                                           # Einzelsonden (7 statt 32 Auswertungen), Prior vs. Grenzfälle
    python -m sbench.oracle_gap 6000                                       # Gap 0: Oracle an der Plattform-Bremsgrenze

`ats.py` leitet den Suchraum aus `AtsOdd` ab und simuliert das Gesamtsystem (Sensorsuite → Weltmodell → RSS/AEB → Fahrzeug mit Latenz).
Alle ATS-, Sensor- und Bremsparameter sind Annahmen; öffentlich belegt ist nur die 2getthere-Designanforderung „60 km/h oder höher".
