# Code-Verifikation

Erzeugt mit `verification/run.sh`. Prüft den Code, nicht die Modellaussagen (die stehen getrennt in `experiments/`).

## Tests

| Kategorie | bestanden | fehlgeschlagen |
| --- | --- | --- |
| Differential/Metamorph | 8 | 0 |
| Golden-Regression | 10 | 0 |
| Property/Fuzz (Hypothesis, derandomisiert) | 10 | 0 |
| Statistik-Bausteine (Experimentcode) | 5 | 0 |
| Unit | 44 | 0 |

## Mutation-Tests (mutmut)

Score = getötet / alle bewerteten Mutanten; Abstürze (segfault) zählen nicht als getötet.

| Modul | getötet | überlebt | ohne Test | Absturz | Score |
| --- | --- | --- | --- | --- | --- |
| budget | 0 | 0 | 104 | 0 | 0.0 |
| probe_study | 139 | 65 | 162 | 0 | 0.38 |
| robustness | 357 | 178 | 0 | 0 | 0.667 |
| sim | 287 | 0 | 0 | 245 | 0.539 |
| world | 113 | 32 | 10 | 0 | 0.729 |
