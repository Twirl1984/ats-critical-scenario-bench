"""Traceability UN R157 (ALKS, Datei aus dem compliance-Repo) → Bench-Szenario → Sensorphänomen,
plus Rückrechnung der Budgets (ΔT*, Δs*, R*) auf ODD-Szenarien.

Das Mapping Anforderung→Szenario ist eine Ingenieurentscheidung (explizit unten), keine Ableitung aus dem Text.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .budget import distance_bias_budget, latency_budget, range_requirement
from .problems import make_problem

R157_JSON = Path(__file__).resolve().parents[2] / "data" / "r157_alks.json"

# req_id -> (Bench-Problem | None, relevante Sensorphänomene, Abdeckung, Bemerkung)
TRACE = {
    "R157-5.1.1": ("alle ODD-Probleme", "Erkennung, Rauschen, Latenz", "teilweise", "nur Längsdynamik; lane_keeping/emergency_manoeuvre-Anteile quer nicht abbildbar"),
    "R157-5.2.3.3": ("lead_brake_odd", "Distanz-/Raten-Rauschen, Latenz", "teilweise", "Zeitlücke 1,0–1,6 s als Regelziel der ACC-Logik; im Bench nur indirekt (Min-Gap)"),
    "R157-5.2.5": ("lead_brake_odd, cut_in_odd, stopped_obstacle_odd", "Erkennung, Sichtfeld, Latenz", "teilweise", "cut_out_avoidance und pedestrian_crossing: nicht abbildbar (kein Quer-/Fußgängermodell, ODD: keine Fußgänger)"),
    "R157-5.2.5.1": ("lead_brake_odd", "Reichweite, Rate (Doppler), Latenz", "ja", "Lead-Verzögerung ≤ 6 m/s² als Prior-Grenze"),
    "R157-5.2.5.2": ("cut_in_odd", "Sichtfeld (p_det), Latenz, Rate", "teilweise", "min_lateral_visibility 0,72 s / ttc_margin 0,35 s nur über Fortschritts-Schwelle p_det abgebildet"),
    "R157-5.3.1": ("lead_brake_odd, stopped_obstacle_odd", "Latenz, Auslöseschwelle", "teilweise", "EM-Schwelle 5 m/s² vs. AEB-Stufe 1 = −4 m/s²: Stufenlogik gegen Anforderung prüfen"),
    "R157-7.1.1": ("stopped_obstacle_odd", "Erkennungsreichweite R*", "ja", "46 m Mindestreichweite gegen berechnetes R* je Szenario"),
    "R157-7.1.2": ("cut_in_odd", "Sichtfeld seitlich", "teilweise", "seitliche Reichweite nur als Zeitpunkt der Erkennung (p_det)"),
}
NOT_COVERED = ["R157-5.2.1 (Spurhaltung quer)", "R157-5.2.3.1 (Höchstgeschwindigkeit)", "R157-5.4.1/5.4.2 (Übernahmeaufforderung)",
               "R157-5.5.1 (MRM bis Stillstand)", "R157-6.1.1 (Fahrerverfügbarkeit)", "R157-6.2.2 (Off-Mode)", "R157-6.3.1 (Lenkübersteuerung)"]
CHAIN_LATENCY = {"ideal": 0.0, "lidar": 0.05 + 0.1, "radar": 0.08 + 0.05, "kamera": 0.12 + 0.033}  # Latenz + halbe Periode (Annahme!)


def _one(args):
    pname, u, lat = args
    p = make_problem(pname)
    return (latency_budget(p, u), distance_bias_budget(p, u), range_requirement(p, u, latency=lat))


def budgets(pname: str, n: int = 250, lat: float = 0.2, seed: int = 5, workers: int = 2) -> dict:
    p = make_problem(pname)
    rng = np.random.default_rng(seed)
    U = rng.beta(p.beta_a, p.beta_b, size=(n, p.dim))
    keep = [u for u in U if all(p.validity(u))]
    with Pool(workers) as pool:
        res = pool.map(_one, [(pname, u, lat) for u in keep])
    dT, ds, R = (np.array([r[i] for r in res]) for i in range(3))
    q = lambda a: {k: float(np.quantile(a[np.isfinite(a)], v)) if np.isfinite(a).any() else None
                   for k, v in (("p05", .05), ("p50", .5), ("p95", .95))}
    return {
        "problem": pname, "n_valid": len(keep), "lat_for_R": lat,
        "dT_quantiles": q(dT), "ds_quantiles": q(ds), "R_quantiles": q(R),
        "share_dT_below": {str(t): float(np.mean(dT < t)) for t in (0.1, 0.2, 0.3, 0.5)},
        "share_already_failing": float(np.mean(dT == 0.0)),
        "share_R_above": {str(r): float(np.mean(R > r)) for r in (30, 46, 60, 100)},
        "share_R_unavoidable_by_logic": float(np.mean(~np.isfinite(R))),
    }


def stopped_range_curve(mu: float = 0.8, lat: float = 0.2) -> list[dict]:
    p = make_problem("stopped_obstacle_odd")
    rows = []
    for v in (5, 8, 10, 12, 14, 16.67):
        u = (np.array([v, 10.0, mu]) - p.lows) / (p.highs - p.lows)
        rows.append({"v_ego": v, "R_ideal": range_requirement(p, u, 0.0), "R_lat": range_requirement(p, u, lat),
                     "R_physik": v * (0.2 + 0.1) + v * v / (2 * mu * 9.81)})
    return rows


def run_trace(out="results_suite", n=250):
    reqs = json.loads(R157_JSON.read_text())["requirements"]
    by = {r["req_id"]: r for r in reqs}
    trace = [{"req_id": k, "title": by[k]["title"], "risk": by[k]["risk_level"], "test_scenarios": by[k]["test_scenarios"],
              "thresholds": by[k].get("thresholds"), "bench": v[0], "phenomena": v[1], "coverage": v[2], "note": v[3]}
             for k, v in TRACE.items()]
    data = {"trace": trace, "not_covered": NOT_COVERED, "chain_latency_assumed": CHAIN_LATENCY,
            "budgets": [budgets(p, n) for p in ("lead_brake_odd", "cut_in_odd", "stopped_obstacle_odd")],
            "range_curve": stopped_range_curve()}
    Path(out).mkdir(exist_ok=True)
    Path(out, "trace_study.json").write_text(json.dumps(data, indent=1, default=float))
    return data
