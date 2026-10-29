"""Abdeckungsstudie: Was bildet die Kette mit Stufe-1-Sensoren pro kritischem Szenario ab,
und welches Sensor-Phänomen/Upgrade begrenzt?

Je (Problem, Umgebung): gleiche Szenarien aus dem Prior (CRN) für alle Sensor-Suiten.
  rate      = Anteil *gültiger* Kollisionen (plausibel + per Oracle vermeidbar) in %
  excess    = rate(Suite) − rate(ideal)           → sensor-bedingter Anteil
  Ablation  = Phänomen idealisieren (nur für Szenarien mit gültiger Kollision re-simuliert)
  Reichweitenfaktor = kleinster Faktor auf r50/range_max, der excess auf ≤ 0.3 pp drückt
"""
from __future__ import annotations

import json
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .problems import make_problem
from .sim import Scenario
from .suite import ENVS, SUITES, simulate_suite

PROBLEMS = ("lead_brake", "cut_in", "stopped_obstacle")
ABL_SUITES = ("radar", "radar+kamera", "radar+kamera+lidar")
FACTORS = (1.25, 1.5, 2.0, 3.0)


def _mods(suite):
    def each(**kw):
        return tuple(replace(sp, **kw) for sp in suite)

    return {
        "Erkennung/Reichweite ideal": (each(r50=1e6, pd0=1.0, range_max=1e6), None),
        "Umweltverlust weg": (suite, ENVS["klar_tag"]),
        "Rauschen weg": (tuple(replace(sp, sig0=0, sig1=0, sig2=0, sig_rate=0.0 if sp.sig_rate is not None else None)
                               for sp in suite), None),
        "Latenz 0": (each(latency=0.0), None),
        "Sichtfeld ideal": (tuple(replace(sp, p_det=min(sp.p_det, 0.4)) for sp in suite), None),
        "Stationärfilter weg": (each(stat_mult=1.0), None),
    }


def _job(args):
    pname, ename, n, seed = args
    prob = make_problem(pname)
    env = ENVS[ename]
    rng = np.random.default_rng(seed)
    U = rng.beta(prob.beta_a, prob.beta_b, size=(n, prob.dim))
    scs = [prob.decode(u) for u in U]
    valid = np.array([all(prob.validity(u)) for u in U])

    def rate(suite, e=env, idx=None):
        idx = range(n) if idx is None else idx
        coll = np.zeros(n, bool)
        for i in idx:
            coll[i] = simulate_suite(scs[i], suite, e, seed=i).collision
        return coll & valid

    out = {"problem": pname, "env": ename, "n": n, "n_valid_pool": int(valid.sum()), "suites": {}}
    base = {}
    for sn, suite in SUITES.items():
        base[sn] = rate(suite)
        out["suites"][sn] = {"rate": 100 * base[sn].mean(), "n_fail": int(base[sn].sum())}
    r0 = out["suites"]["ideal"]["rate"]
    for sn, suite in SUITES.items():
        d = out["suites"][sn]
        d["excess"] = d["rate"] - r0
        if sn in ABL_SUITES and d["n_fail"] > 0:
            idx = np.flatnonzero(base[sn])
            abl = {}
            for mname, (s2, e2) in _mods(suite).items():
                after = rate(s2, e2 or env, idx)
                abl[mname] = 100 * (d["n_fail"] - after[idx].sum()) / n  # vermiedene Fehler in pp
            d["ablation_pp"] = abl
        if sn != "ideal" and d["excess"] > 0.5:
            need = None
            for f in FACTORS:
                s2 = tuple(replace(sp, r50=sp.r50 * f, range_max=sp.range_max * f) for sp in suite)
                r = 100 * rate(s2).mean()
                if r - r0 <= 0.3:
                    need = f
                    break
            d["range_factor_needed"] = need if need is not None else f">{FACTORS[-1]}"
    return out


def false_brake(n: int = 500) -> dict:
    sc = Scenario("lead_brake", 30.0, 0.9, 33.0, 1e5, 0.0, 0.0)  # kein Ziel in Reichweite
    res = {}
    for sn, suite in SUITES.items():
        dr = [simulate_suite(sc, suite, seed=i).extra for i in range(n)]
        res[sn] = {"false_brake_pct": 100 * float(np.mean([d["v_drop"] > 2.0 for d in dr])),
                   "ghost_per_run": float(np.mean([d["ghosts"] for d in dr]))}
    return res


def run_study(n=800, workers=2, out="results_suite", seed=11):
    jobs = [(p, e, n, seed) for p in PROBLEMS for e in ENVS]
    with Pool(workers) as pool:
        rows = pool.map(_job, jobs)
    data = {"n": n, "rows": rows, "false_brake": false_brake()}
    Path(out).mkdir(exist_ok=True)
    Path(out, "suite_study.json").write_text(json.dumps(data, indent=1, default=float))
    return data
