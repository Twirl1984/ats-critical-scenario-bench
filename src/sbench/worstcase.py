"""Vorabschätzung statt Brute Force: Worst-Case-Ecke der Toleranzbox aus der Sensitivität des Gesamtmodells.

Idee: g(u, θ) = Kritikalitätsmaß (min. TTC, <0 bei Kollision) hängt von den Abweichungen θ ab
(s_bias, w_scale, vr_bias, a_bias, delay).  Lineare Näherung  g(θ) ≈ g(0) + ∇θ g · θ  → die Worst-Case-Ecke ist
θ* = −sign(∇θ g) · tol.  Kosten: 11 Simulationen (Nominal, 4 zentrale Differenzen, 1 Vorwärtsdifferenz für delay, 1 Ecke) statt 33.
Geprüft wird, ob θ* die echte Kollisionsecke (aus allen 32 Ecken) trifft, und wo die Linearisierung versagt
(AEB-Stufen und Tote Zonen machen g stückweise, nicht glatt).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from .problems import make_problem
from .sim import SensorConfig

KEYS = ("s_bias", "w_scale", "vr_bias", "a_bias", "delay")


def _cfg(vals: dict) -> SensorConfig:
    return SensorConfig(**vals)


def study(name: str = "ats_ped", n: int = 3000, g_max: float = 2.5, seed: int = 4) -> dict:
    prob = make_problem(name)
    t = prob.system.tol
    tol = {"s_bias": t.s_bias, "w_scale": t.w_scale, "vr_bias": t.vr_bias, "a_bias": t.a_bias, "delay": t.delay}
    rng = np.random.default_rng(seed)
    U = rng.beta(prob.beta_a, prob.beta_b, size=(n, prob.dim))
    rows = []
    for u in U:
        if not all(prob.validity(u)):
            continue
        case = prob.decode(u)
        nom = type(case)(case.scenario, case.env, SensorConfig())
        g0 = prob.simulate(nom).g
        if g0 <= 0 or g0 > g_max:
            continue  # nur nominell bestandene, aber knappe Szenarien
        # Brute Force: 32 Ecken (delay ∈ {0, tol})
        best = (g0, None)
        gs = {}
        for sg in itertools.product((-1, 1), repeat=4):
            for dl in (0.0, 1.0):
                th = {"s_bias": sg[0] * tol["s_bias"], "w_scale": sg[1] * tol["w_scale"], "vr_bias": sg[2] * tol["vr_bias"],
                      "a_bias": sg[3] * tol["a_bias"], "delay": dl * tol["delay"]}
                g = prob.simulate(nom, _cfg(th)).g
                gs[(sg, dl)] = g
        g_brute = min(gs.values())
        # Vorabschätzung: Gradient je Parameter mit Vorwärtsdifferenz in +tol-Richtung (delay nur +)
        grad = {}
        for k in KEYS:
            gp = prob.simulate(nom, _cfg({k: tol[k]})).g
            gm = prob.simulate(nom, _cfg({k: -tol[k]})).g if k != "delay" else g0
            grad[k] = (gp - gm) / (2 if k != "delay" else 1)
        th = {k: (-np.sign(grad[k]) if grad[k] != 0 else 1.0) * tol[k] for k in KEYS}
        th["delay"] = tol["delay"] if grad["delay"] < 0 else 0.0
        g_pred = prob.simulate(nom, _cfg(th)).g
        rows.append({"g0": g0, "g_brute": g_brute, "g_pred": g_pred, "flip_brute": g_brute <= 0, "flip_pred": g_pred <= 0,
                     "dominant": max(grad, key=lambda k: abs(grad[k]))})
    r = len(rows)
    fb = [x for x in rows if x["flip_brute"]]
    res = {
        "problem": name, "n_near_critical": r, "n_flip_brute": len(fb),
        "n_flip_pred": sum(x["flip_pred"] for x in rows),
        "hit_rate_on_brute_flips": float(np.mean([x["flip_pred"] for x in fb])) if fb else None,
        "false_alarm": int(sum(x["flip_pred"] and not x["flip_brute"] for x in rows)),
        "median_gap_pred_vs_brute": float(np.median([x["g_pred"] - x["g_brute"] for x in rows])) if r else None,
        "cost_ratio_sims_per_scenario": "11 statt 33",
        "dominant_counts": {k: sum(x["dominant"] == k for x in rows) for k in KEYS},
    }
    return res


if __name__ == "__main__":
    import sys

    out = study(sys.argv[1] if len(sys.argv) > 1 else "ats_ped", int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
    print(json.dumps(out, indent=1, default=float), flush=True)
    Path("results_ats").mkdir(exist_ok=True)
    Path("results_ats", f"worstcase_{out['problem'].replace(':', '_').replace('=', '')}.json").write_text(json.dumps(out, indent=1, default=float))
