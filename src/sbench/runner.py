"""Benchmark-Runner: Algorithmen × Seeds, parallel, mit JSON-/CSV-/Markdown-Ausgabe."""
from __future__ import annotations

import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .algos import ALGOS, BudgetExhausted, Evaluator, make_algo
from .metrics import run_metrics
from .problems import make_problem


def run_one(task: tuple) -> dict:
    problem_name, algo_name, budget, seed, constrained = task
    problem = make_problem(problem_name)
    algo = make_algo(algo_name)
    ev = Evaluator(problem, budget, constrained=constrained)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    try:
        extras = algo.run(ev, rng)
    except BudgetExhausted:
        extras = {}
    runtime = time.time() - t0
    U, G, NLL = ev.arrays()
    m = run_metrics(problem, U, G, NLL, extras)
    m.update(algo=algo_name, seed=seed, runtime_s=runtime, problem=problem_name)
    # Fehlerpunkte für das Robustheitsmodul mitschreiben
    fail = np.flatnonzero(G <= 0)
    m["failures"] = [{"u": U[i].tolist(), "g": float(G[i])} for i in fail[:200]]
    return m


def run_benchmark(problem, algos, budget, seeds, workers=2, out_dir="results", constrained=False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(problem, a, budget, s, constrained) for a in algos for s in range(seeds)]
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            runs = pool.map(run_one, tasks, chunksize=1)
    else:
        runs = [run_one(t) for t in tasks]
    result = {
        "config": dict(problem=problem, algos=list(algos), budget=budget, seeds=seeds, constrained=constrained,
                       wall_s=time.time() - t0),
        "runs": runs,
    }
    (out / f"{problem}_run.json").write_text(json.dumps(result))
    summary = summarize(runs)
    _write_csv(out / f"{problem}_summary.csv", summary)
    (out / f"{problem}_report.md").write_text(render_report(result["config"], summary))
    try:
        _plot_curves(out / f"{problem}_curves.png", runs, budget, problem)
    except Exception as exc:  # Plot ist optional
        print(f"Plot übersprungen: {exc}")
    return result


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(runs: list[dict]) -> list[dict]:
    rows = []
    for algo in dict.fromkeys(r["algo"] for r in runs):
        rs = [r for r in runs if r["algo"] == algo]
        firsts = [r["first_valid_fail"] for r in rs if r["first_valid_fail"] is not None]
        rows.append({
            "algo": algo,
            "found_valid_%": 100.0 * len(firsts) / len(rs),
            "valid_fails": _mean([r["n_valid_fail"] for r in rs]),
            "all_fails": _mean([r["n_fail"] for r in rs]),
            "implausible_%": 100.0 * _mean([r["frac_implausible"] for r in rs]),
            "unavoidable_%": 100.0 * _mean([r["frac_unavoidable"] for r in rs]),
            "first_valid_med": float(np.median(firsts)) if firsts else None,
            "clusters": _mean([r["n_clusters_valid"] for r in rs]),
            "nn_dist": _mean([r["nn_dist_valid"] for r in rs]),
            "g_min": _mean([r["g_min_valid"] for r in rs]),
            "mean_nll": _mean([r["mean_nll_valid"] for r in rs]),
            "pf_valid": _mean([r["pf_est_valid"] for r in rs]),
            "runtime_s": _mean([r["runtime_s"] for r in rs]),
        })
    return rows


def _fmt(v, nd=2):
    return "–" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) and abs(v) < 0.01 and v != 0 else f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_report(cfg: dict, rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    lines = [
        f"# Benchmark {cfg['problem']}",
        "",
        f"Budget {cfg['budget']} Evaluationen, {cfg['seeds']} Seeds, constrained={cfg['constrained']}, Laufzeit {cfg['wall_s']:.0f}s.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    lines += [
        "",
        "**Lesehilfe:** *valid* = Fehler (g ≤ 0), der physikalisch plausibel (Zielverzögerung ≤ µ·g) und vom Oracle "
        "vermeidbar ist. `implausible_%`/`unavoidable_%` = Anteil der gefundenen Fehler, die das nicht sind. "
        "`clusters` = Anzahl Fehlermuster (Complete-Linkage, Schwelle 0.3). `pf_valid` = IS-Schätzung der "
        "Fehlerwahrscheinlichkeit unter dem Prior (nur ce_is).",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_curves(path: Path, runs, budget, problem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for algo in dict.fromkeys(r["algo"] for r in runs):
        curves = np.array([r["curve_valid"] for r in runs if r["algo"] == algo], float)
        ax.plot(np.arange(1, curves.shape[1] + 1), curves.mean(0), label=algo)
    ax.set_xlabel("Evaluationen")
    ax.set_ylabel("Ø gültige Fehlerszenarien (kumulativ)")
    ax.set_title(f"{problem}: Effizienz der Szenariosuche")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
