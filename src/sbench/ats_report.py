"""Auswertung gefundener kritischer ATS-Szenarien: Charakterisierung, Messfehler-Relevanz, Robustheitsverdikt."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .problems import make_problem
from .robustness import Tolerances, scenario_robustness
from .sim import SensorConfig


def analyse(run_json: str, out: str = "results_ats", top: int = 25) -> dict:
    data = json.loads(Path(run_json).read_text())
    prob = make_problem(data["config"]["problem"])
    seen, items = [], []
    for r in data["runs"]:
        for f in r["failures"]:
            u = np.array(f["u"])
            if all(prob.validity(u)) and not any(np.linalg.norm(u - s) < 0.02 for s in seen):
                seen.append(u)
                items.append(u)
    rows = []
    for u in items:
        case = prob.decode(u)
        p = prob.params(u)
        nominal = prob.simulate(type(case)(case.scenario, case.env, SensorConfig()), SensorConfig(), seed=0).collision  # ohne Abweichung/Rauschen
        rows.append({"env": case.env.name.split("+")[0], "v_ego": p["v_ego"], "p_vis": p.get("p_vis"), "mu": p["mu"],
                     "nominal_fail": bool(nominal), "dev": {k: p[k] for k in p if k.startswith("dev_")}})
    n = len(rows)
    env_c = Counter(r["env"] for r in rows)
    needs_dev = sum(not r["nominal_fail"] for r in rows)
    rng = np.random.default_rng(0)
    pick = [items[i] for i in rng.permutation(n)[:top]]
    tol = prob.system.tol
    rob = [scenario_robustness(prob, u, tol, n_mc=60) for u in pick]
    verdicts = Counter(r["verdict"] for r in rob)
    dom = Counter(r["dominant_sensor"] for r in rob)
    summary = {
        "problem": prob.name, "n_unique_valid_failures": n, "env_counts": dict(env_c),
        "share_fail_without_deviation": 1 - needs_dev / max(n, 1), "share_only_with_deviation": needs_dev / max(n, 1),
        "v_ego_quantiles": {k: float(np.quantile([r["v_ego"] for r in rows], q)) for k, q in (("p10", .1), ("p50", .5), ("p90", .9))} if n else {},
        "p_vis_median": float(np.median([r["p_vis"] for r in rows])) if n and rows[0]["p_vis"] is not None else None,
        "robustness_verdicts": dict(verdicts), "dominant_deviation": dict(dom), "tolerances": tol.__dict__ if hasattr(tol, "__dict__") else str(tol),
    }
    Path(out).mkdir(exist_ok=True)
    Path(out, f"{prob.name.replace(':', '_').replace('=', '')}_findings.json").write_text(json.dumps({"summary": summary, "rows": rows, "robustness": rob}, default=float))
    return summary


def deviation_impact(name: str, n: int = 600, seed: int = 3) -> dict:
    """Wie viel der Fehlerrate geht auf zulässige Messabweichung/Zeitabweichung zurück?
    (a) nominal: Abweichung 0, kein Rauschen   (b) Prior-Abweichung + Rauschen   (c) irgendeine der 32 Toleranz-Ecken kollidiert."""
    import itertools

    from .sim import SensorConfig

    prob = make_problem(name)
    rng = np.random.default_rng(seed)
    U = rng.beta(prob.beta_a, prob.beta_b, size=(n, prob.dim))
    t = prob.system.tol
    a = b = c = 0
    valid = 0
    for u in U:
        if not all(prob.validity(u)):
            continue
        valid += 1
        case = prob.decode(u)
        nom = type(case)(case.scenario, case.env, SensorConfig())
        fa = prob.simulate(nom).collision
        fb = prob.simulate(case, seed=1).collision
        fc = fa or fb
        if not fc:
            for sg in itertools.product((-1, 1), repeat=4):
                d = SensorConfig(s_bias=sg[0] * t.s_bias, w_scale=sg[1] * t.w_scale, vr_bias=sg[2] * t.vr_bias, a_bias=sg[3] * t.a_bias, delay=t.delay)
                if prob.simulate(nom, d).collision:
                    fc = True
                    break
        a += fa; b += fb; c += fc
    return {"problem": name, "n": n, "n_valid": valid, "rate_nominal": a / n, "rate_prior_dev_noise": b / n, "rate_any_tolerance_corner": c / n}
