"""Rückrechnung: vom kritischen Szenario zur tolerierbaren Sensor-/Laufzeitabweichung.

Für ein Szenario u (Kollision ohne Abweichung = nein) wird bestimmt, wie viel Abweichung die Kette
gerade noch verträgt:
  ΔT*   zusätzliche Ende-zu-Ende-Latenz (Perzeption + Fusion + Übertragung) bis zur Kollision
  Δs*   Distanz-Überschätzung (Bias, +) bis zur Kollision
  R*    minimal nötige Erkennungsreichweite (ideal rauschfreier Sensor, gegebene Latenz)
Bisektion auf dem Kollisionsverdikt (monoton in der Abweichung, per Test geprüft).
Ein Szenario, das schon ohne Abweichung kollidiert, hat Budget 0 („logik-/physikbegrenzt").
"""
from __future__ import annotations

from dataclasses import replace

from .sim import SensorConfig
from .suite import IDEAL, ENVS, simulate_suite


def _bisect(fail, lo: float, hi: float, tol: float) -> float:
    """Größter Wert x∈[lo,hi] mit fail(x)=False, vorausgesetzt fail(lo)=False. hi→ inf wenn nie Fehler."""
    if not fail(hi):
        return float("inf")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if not fail(mid) else (lo, mid)
    return lo


def latency_budget(problem, u, hi: float = 1.5, tol: float = 0.005) -> float:
    sc = problem.decode(u)
    fail = lambda d: problem.simulate(sc, SensorConfig(delay=d)).collision
    return 0.0 if fail(0.0) else _bisect(fail, 0.0, hi, tol)


def distance_bias_budget(problem, u, hi: float = 40.0, tol: float = 0.1) -> float:
    sc = problem.decode(u)
    fail = lambda b: problem.simulate(sc, SensorConfig(s_bias=b)).collision
    return 0.0 if fail(0.0) else _bisect(fail, 0.0, hi, tol)


def range_requirement(problem, u, latency: float = 0.0, hi: float = 250.0, tol: float = 0.5) -> float:
    """Minimal nötige Erkennungsreichweite R*. inf = auch mit hi nicht vermeidbar (Logikgrenze)."""
    sc = problem.decode(u)
    fail = lambda R: simulate_suite(sc, (replace(IDEAL, range_max=R, latency=latency),), ENVS["klar_tag"]).collision
    if fail(hi):
        return float("inf")
    lo, h = 0.0, hi
    while h - lo > tol:
        mid = 0.5 * (lo + h)
        lo, h = (lo, mid) if not fail(mid) else (mid, h)
    return h
