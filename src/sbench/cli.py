"""CLI:  sbench run | reference | robustness"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .algos import ALGOS
from .problems import make_problem
from .robustness import Tolerances, scenario_robustness
from .runner import run_benchmark
from .staleness import render, staleness_sweep


def _ref_chunk(args):
    name, n, seed = args
    p = make_problem(name)
    rng = np.random.default_rng(seed)
    U = rng.beta(p.beta_a, p.beta_b, size=(n, p.dim))
    fails, valid = 0, 0
    for u in U:
        if p.evaluate(u)[0] <= 0:
            fails += 1
            valid += all(p.validity(u))
    return fails, valid


def cmd_reference(a):
    """Referenz-Fehlerwahrscheinlichkeit unter dem Prior per Monte-Carlo (Gegenprobe für ce_is)."""
    chunks = [(a.problem, a.n // a.workers, 1000 + i) for i in range(a.workers)]
    with Pool(a.workers) as pool:
        res = pool.map(_ref_chunk, chunks)
    n = sum(c[1] for c in chunks)
    f, v = sum(r[0] for r in res), sum(r[1] for r in res)
    out = {"problem": a.problem, "n": n, "pf_raw": f / n, "pf_valid": v / n,
           "pf_valid_se": float(np.sqrt(v / n * (1 - v / n) / n))}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.problem}_reference.json").write_text(json.dumps(out))
    print(json.dumps(out, indent=2))


def cmd_run(a):
    algos = list(ALGOS) if a.algos == "all" else a.algos.split(",")
    res = run_benchmark(a.problem, algos, a.budget, a.seeds, a.workers, a.out, constrained=a.constrained)
    print((Path(a.out) / f"{a.problem}_report.md").read_text())


def cmd_robustness(a):
    data = json.loads(Path(a.run_json).read_text())
    problem = make_problem(data["config"]["problem"])
    seen, items = [], []
    for r in data["runs"]:
        for f in r["failures"]:
            u = np.array(f["u"])
            if all(problem.validity(u)) and not any(np.linalg.norm(u - s) < 0.02 for s in seen):
                seen.append(u)
                items.append((r["algo"], u))
    rng = np.random.default_rng(0)
    rng.shuffle(items)
    items = items[: a.top]
    tol = Tolerances()
    rows = []
    t0 = time.time()
    for algo, u in items:
        rep = scenario_robustness(problem, u, tol, n_mc=a.mc)
        rep.update(algo=algo, u=u.tolist())
        rows.append(rep)
    verdicts = Counter(r["verdict"] for r in rows)
    dom = Counter(r["dominant_sensor"] for r in rows)
    flips = sum(r["corner_flip"] for r in rows)
    summary = {"n_scenarios": len(rows), "verdicts": dict(verdicts), "dominant_sensor": dict(dom),
               "corner_flip": flips, "wall_s": time.time() - t0, "tolerances": Tolerances().as_dict()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{problem.name}_robustness.json").write_text(json.dumps({"summary": summary, "scenarios": rows}))
    print(json.dumps(summary, indent=2))


def cmd_staleness(a):
    rows = staleness_sweep(n=a.n, workers=a.workers, out=a.out)
    print(render(rows))


def cmd_suitestudy(a):
    from .suite_study import run_study

    run_study(n=a.n, workers=a.workers, out=a.out)
    print("fertig:", a.out)


def cmd_trace(a):
    from .trace_study import run_trace

    run_trace(out=a.out, n=a.n)
    print("fertig:", a.out)


def cmd_atsreport(a):
    from .ats_report import analyse

    print(json.dumps(analyse(a.run_json, a.out, a.top), indent=2, default=float))


def cmd_devimpact(a):
    from .ats_report import deviation_impact

    r = deviation_impact(a.problem, a.n)
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / (a.problem.replace(":", "_").replace("=", "") + "_devimpact.json")).write_text(json.dumps(r))
    print(json.dumps(r, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--problem", default="lead_brake")
    r.add_argument("--algos", default="all", help="'all' oder kommagetrennt: " + ",".join(ALGOS))
    r.add_argument("--budget", type=int, default=300)
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--constrained", action="store_true", help="ungültige Fehler zählen im Suchsignal nicht")
    r.add_argument("--out", default="results")
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser("reference")
    f.add_argument("--problem", default="lead_brake")
    f.add_argument("--n", type=int, default=20000)
    f.add_argument("--workers", type=int, default=2)
    f.add_argument("--out", default="results")
    f.set_defaults(fn=cmd_reference)
    b = sub.add_parser("robustness")
    b.add_argument("run_json")
    b.add_argument("--top", type=int, default=20)
    b.add_argument("--mc", type=int, default=100)
    b.add_argument("--out", default="results")
    b.set_defaults(fn=cmd_robustness)
    st = sub.add_parser("staleness", help="Altersgrenze des Weltmodells vs. Sensor-Einfrieren")
    st.add_argument("--n", type=int, default=3000)
    st.add_argument("--workers", type=int, default=2)
    st.add_argument("--out", default="results")
    st.set_defaults(fn=cmd_staleness)
    ss = sub.add_parser("suitestudy", help="Abdeckungsstudie Szenario x Sensor-Suite x Umgebung")
    ss.add_argument("--n", type=int, default=800)
    ss.add_argument("--workers", type=int, default=2)
    ss.add_argument("--out", default="results_suite")
    ss.set_defaults(fn=cmd_suitestudy)
    tr = sub.add_parser("trace", help="R157-Traceability + Budget-Rückrechnung (ΔT*, Δs*, R*)")
    tr.add_argument("--n", type=int, default=250)
    tr.add_argument("--out", default="results_suite")
    tr.set_defaults(fn=cmd_trace)
    ar = sub.add_parser("atsreport", help="ATS: Charakterisierung der gefundenen kritischen Szenarien")
    ar.add_argument("run_json")
    ar.add_argument("--top", type=int, default=25)
    ar.add_argument("--out", default="results_ats")
    ar.set_defaults(fn=cmd_atsreport)
    di = sub.add_parser("devimpact", help="ATS: Anteil der Fehlerrate durch zulässige Messabweichung")
    di.add_argument("--problem", default="ats_ped")
    di.add_argument("--n", type=int, default=600)
    di.add_argument("--out", default="results_ats")
    di.set_defaults(fn=cmd_devimpact)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
