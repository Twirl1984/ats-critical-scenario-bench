"""Altersgrenze des Weltmodells (max_age) gegen Sensor-Einfrieren: Sicherheit vs. Blindbremsen.

Gleiche Szenario-Stichprobe (Common Random Numbers) für jede Altersgrenze, Szenario aus dem Prior, Fehlerparameter (Beginn/Dauer/Modus/Sensor) gleichverteilt.
Ergebnis je max_age: gültige Kollisionen (gesamt / mit Fehler / nach Modus) und Zeit im „Blindflug".
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .problems import LeadBrakeChain


def _one_age(args):
    max_age, n, seed = args
    prob = LeadBrakeChain(max_age)
    rng = np.random.default_rng(seed)
    U = rng.beta(prob.beta_a, prob.beta_b, size=(n, prob.dim))
    U[:, 6:] = rng.uniform(size=(n, 4))  # Fehlerparameter gleichverteilt (Prior macht Fehler zu selten)
    rows = []
    for u in U:
        sc = prob.decode(u)
        res = prob.simulate(sc)
        valid = False
        if res.g <= 0:
            valid = all(prob.validity(u))
        rows.append((res.g <= 0, valid, sc.faults.dur >= 0.1, sc.faults.mode, res.extra["blind_s"]))
    a = np.array([(r[0], r[1], r[2], r[3] == "repeat", r[4]) for r in rows], float)
    coll, valid, fault, repeat, blind = a.T
    fault = fault.astype(bool)
    repeat = repeat.astype(bool)
    f_stall, f_rep = fault & ~repeat, fault & repeat
    return {
        "max_age": max_age, "n": n, "fault_share": float(fault.mean()),
        "valid_coll": float(valid.mean()),
        "valid_coll_nofault": float(valid[~fault].mean()),
        "valid_coll_fault": float(valid[fault].mean()),
        "valid_coll_stall": float(valid[f_stall].mean()) if f_stall.any() else None,
        "valid_coll_repeat": float(valid[f_rep].mean()) if f_rep.any() else None,
        "blind_mean_s": float(blind.mean()), "blind_ge_0.5s": float((blind >= 0.5).mean()),
    }


def staleness_sweep(ages=(0.1, 0.2, 0.3, 0.5, 1.0, 2.0), n=4000, seed=7, workers=2, out="results") -> list[dict]:
    with Pool(workers) as pool:
        rows = pool.map(_one_age, [(a, n, seed) for a in ages])
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "staleness_sweep.json").write_text(json.dumps(rows, indent=1))
    return rows


def render(rows: list[dict]) -> str:
    f = lambda v, p=2: "–" if v is None else f"{100 * v:.{p}f}"
    head = "| max_age [s] | gültige Kollisionen % | davon ohne Fehler % | mit Fehler % | stall % | repeat % | Ø Blindzeit [s] | Blind ≥0,5 s % |"
    lines = [head, "|" + "---|" * 8]
    for r in rows:
        lines.append(f"| {r['max_age']} | {f(r['valid_coll'])} | {f(r['valid_coll_nofault'])} | {f(r['valid_coll_fault'])} | "
                     f"{f(r['valid_coll_stall'])} | {f(r['valid_coll_repeat'])} | {r['blind_mean_s']:.3f} | {f(r['blind_ge_0.5s'], 1)} |")
    return "\n".join(lines)
