"""Zweiter Worst-Case-Test: Einzelsonden statt Sekanten, getrennt nach Zufalls- und Grenzfällen.

Je Szenario: nominell + eine Sonde je Abweichungsparameter (+Toleranz, Latenz nur +) → Vorzeichen →
eine vorhergesagte Ecke, zusammen 7 statt 32 Auswertungen. Referenz ist der Scan aller 32 Ecken
(nicht das wahre Maximum in der Box). Zusätzlich die rein lineare Schätzung aus den Sonden
(g_lin = g0 − Σ|Δg_k|, Latenz nur wenn sie verschlechtert; 6 Auswertungen, keine Ecke gerechnet).

„Flip" = nominell bestanden (g0 > 0), in mindestens einer Ecke Fehler (g ≤ 0).

    python -m sbench.probe_study                     # drei ODD-Probleme, Prior + Grenzfälle
    python -m sbench.probe_study --n-prior 50 --n-edge 40 --max-draws 4000   # schneller Lauf
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .problems import make_problem
from .robustness import Tolerances
from .sim import SensorConfig

KEYS = ("s_bias", "w_scale", "vr_bias", "a_bias", "delay")
PROBLEMS = {"lead_brake_odd": "Auffahren", "cut_in_odd": "Einscheren", "stopped_obstacle_odd": "stehendes Hindernis"}


def _corner(tol: dict, signs: dict) -> SensorConfig:
    th = {k: signs[k] * tol[k] for k in KEYS if k != "delay"}
    th["delay"] = tol["delay"] if signs["delay"] > 0 else 0.0
    return SensorConfig(**th)


def evaluate_case(prob, u, tol: dict) -> dict:
    sc = prob.decode(u)
    g0 = prob.simulate(sc).g
    corners = {}
    for sg in itertools.product((-1, 1), repeat=4):
        for dl in (0, 1):
            signs = dict(zip(KEYS[:4], sg), delay=dl)
            corners[tuple(sg) + (dl,)] = prob.simulate(sc, _corner(tol, signs)).g
    worst_key = min(corners, key=corners.get)
    # Einzelsonden (+Toleranz) → Vorzeichen der Ecke
    dg = {k: prob.simulate(sc, SensorConfig(**{k: tol[k]})).g - g0 for k in KEYS}
    signs = {k: (-1 if dg[k] > 0 else 1) for k in KEYS[:4]}
    signs["delay"] = 1 if dg["delay"] < 0 else 0
    key = tuple(signs[k] for k in KEYS[:4]) + (signs["delay"],)
    g_pred = prob.simulate(sc, _corner(tol, signs)).g
    g_lin = g0 - sum(abs(dg[k]) for k in KEYS[:4]) + min(0.0, dg["delay"])
    g_min = corners[worst_key]
    return {"g0": g0, "g_min": g_min, "g_pred": g_pred, "g_lin": g_lin,
            "flip": bool(g0 > 0 and g_min <= 0), "pred_hit": bool(g_pred <= 0), "lin_hit": bool(g_lin <= 0),
            "same_corner": bool(key == worst_key or np.isclose(g_pred, g_min))}


def _summary(rows: list[dict]) -> dict:
    flips = [r for r in rows if r["flip"]]
    ok = [r for r in rows if r["g0"] > 0]
    return {
        "n_valid": len(rows),
        "n_nominal_fail": int(sum(r["g0"] <= 0 for r in rows)),
        "n_flips": len(flips),
        "found_by_prediction": sum(r["pred_hit"] for r in flips),
        "found_by_linear": sum(r["lin_hit"] for r in flips),
        "false_alarm_prediction": sum(r["pred_hit"] and not r["flip"] for r in ok),
        "false_alarm_linear": sum(r["lin_hit"] and not r["flip"] for r in ok),
        "corner_is_worst_corner": float(np.mean([r["same_corner"] for r in rows])) if rows else None,
    }


def study(n_prior: int = 150, n_edge: int = 120, g_edge: float = 1.0, max_draws: int = 24000, seed: int = 7) -> dict:
    tol = Tolerances().as_dict()
    out = {"tolerances": tol, "g_edge": g_edge, "problems": {}}
    for name, label in PROBLEMS.items():
        prob = make_problem(name)
        rng = np.random.default_rng(seed)
        prior, edge, draws = [], [], 0
        while (len(prior) < n_prior or len(edge) < n_edge) and draws < max_draws:
            u = rng.beta(prob.beta_a, prob.beta_b)
            draws += 1
            if not all(prob.validity(u)):
                continue
            if len(prior) < n_prior:
                prior.append(evaluate_case(prob, u, tol))
            if len(edge) < n_edge and prob.simulate(prob.decode(u)).g <= g_edge:
                edge.append(evaluate_case(prob, u, tol))
        out["problems"][name] = {"label": label, "draws": draws,
                                 "prior": _summary(prior), "edge": _summary(edge)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-prior", type=int, default=150)
    ap.add_argument("--n-edge", type=int, default=120)
    ap.add_argument("--g-edge", type=float, default=1.0)
    ap.add_argument("--max-draws", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results_ats/probe_study.json")
    a = ap.parse_args(argv)
    res = study(a.n_prior, a.n_edge, a.g_edge, a.max_draws, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
