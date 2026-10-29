"""Vorab registrierter Hypothesentest (docs/PREREGISTRATION_H1.md):
Sensitivität eines gelernten Schlupf-Teilmodells vs. tolerierbare Latenz im geschlossenen Regelkreis.

    python -m sbench.hyp_slip                      # bestätigender Lauf → results_verify/hyp_slip.json
    python -m sbench.hyp_slip --n-scen 6 --quick   # Smoke-Lauf (nicht bestätigend)
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .problems import make_problem
from .robustness import Tolerances
from .sim import DT, SensorConfig, slip

A_LO, A_HI, V_LO, V_HI = -10.0, 2.0, 0.0, 20.0
DA, DV = 0.05, 0.125
AMP = ("s_bias", "w_scale", "vr_bias", "a_bias")


# ------------------------------------------------------------------ Schätzer als Tabelle (bilinear)
class TabSlip:
    def __init__(self, fn):
        self.na = int(round((A_HI - A_LO) / DA)) + 1
        self.nv = int(round((V_HI - V_LO) / DV)) + 1
        a = A_LO + DA * np.arange(self.na)
        v = V_LO + DV * np.arange(self.nv)
        A, V = np.meshgrid(a, v, indexing="ij")
        self.tab = np.asarray(fn(A.ravel(), V.ravel()), float).reshape(self.na, self.nv)
        self.rows = self.tab.tolist()

    def __call__(self, a: float, v: float) -> float:
        x = (min(max(a, A_LO), A_HI) - A_LO) / DA
        y = (min(max(v, V_LO), V_HI) - V_LO) / DV
        i, j = min(int(x), self.na - 2), min(int(y), self.nv - 2)
        fx, fy = x - i, y - j
        r0, r1 = self.rows[i], self.rows[i + 1]
        return (r0[j] * (1 - fx) + r1[j] * fx) * (1 - fy) + (r0[j + 1] * (1 - fx) + r1[j + 1] * fx) * fy


def _phys_vec(a, v):
    return np.array([slip(x, y) for x, y in zip(a, v)])


def train_estimators(quick: bool = False) -> list[dict]:
    from sklearn.neural_network import MLPRegressor

    rng = np.random.default_rng(0)
    a = rng.uniform(-9.5, 2.0, 400)
    v = rng.uniform(0.0, 20.0, 400)
    s = _phys_vec(a, v)
    grid = list(itertools.product((1e-6, 1e-4, 1e-2, 1.0), ((16,), (64, 64)), (0.0, 0.002, 0.01)))
    if quick:
        grid = grid[::6]
    out = []
    for k, (alpha, hid, sig) in enumerate(grid):
        y = s + np.random.default_rng(100 + k).normal(0, sig, s.size)
        m = MLPRegressor(hidden_layer_sizes=hid, activation="tanh", solver="lbfgs", alpha=alpha, max_iter=3000, random_state=k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(np.c_[a / 10, v / 20], y * 10)
        fn = (lambda mm: (lambda A, V: mm.predict(np.c_[np.asarray(A) / 10, np.asarray(V) / 20]) / 10))(m)
        out.append({"id": f"mlp{k:02d}", "alpha": alpha, "hidden": list(hid), "sigma": sig, "tab": TabSlip(fn)})
    return out


def estimator_metrics(tab: TabSlip) -> dict:
    a = np.linspace(-9.5, 2.0, 61)
    v = np.linspace(0.0, 20.0, 41)
    err, sens = [], []
    h = 0.05
    for x in a:
        for y in v:
            err.append(tab(x, y) - slip(x, y))
            sens.append(abs(tab(x + h, y) - tab(x - h, y)) / (2 * h))
    return {"rmse": float(np.sqrt(np.mean(np.square(err)))), "J": float(np.quantile(sens, 0.95))}


# ------------------------------------------------------------------ Optimierung im Regelkreis
def _g(prob, sc, theta: dict, k_delay: int, slip_fn) -> float:
    return prob.simulate(sc, SensorConfig(delay=k_delay * DT, **theta), slip_fn=slip_fn).g


def delta_t_star(prob, sc, tol: dict, slip_fn, k_max: int = 50) -> tuple[float, dict]:
    g0 = _g(prob, sc, {}, 0, slip_fn)
    theta = {}
    for k in AMP:
        gp = _g(prob, sc, {k: tol[k]}, 0, slip_fn)
        theta[k] = -tol[k] if gp > g0 else tol[k]
    if _g(prob, sc, theta, 0, slip_fn) <= 0:
        return 0.0, theta
    lo, hi = 0, k_max + 1  # g(lo) > 0; hi = erster Wert mit g ≤ 0 (k_max+1 = nie gefunden)
    if _g(prob, sc, theta, k_max, slip_fn) > 0:
        return k_max * DT, theta
    hi = k_max
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _g(prob, sc, theta, mid, slip_fn) > 0:
            lo = mid
        else:
            hi = mid
    return lo * DT, theta


def select_scenarios(n: int, g_max: float = 1.5, seed: int = 21) -> list:
    prob = make_problem("cut_in_odd")
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        u = rng.beta(prob.beta_a, prob.beta_b)
        if not all(prob.validity(u)):
            continue
        g = prob.simulate(prob.decode(u)).g
        if 0 < g <= g_max:
            out.append(u)
    return out


_STATE: dict = {}


def _init(tabs, U):
    _STATE["tabs"], _STATE["U"] = tabs, U
    _STATE["prob"] = make_problem("cut_in_odd")


def _job(i):
    prob, tol = _STATE["prob"], Tolerances().as_dict()
    tab = _STATE["tabs"][i]
    return [delta_t_star(prob, prob.decode(u), tol, tab)[0] for u in _STATE["U"]]


# ------------------------------------------------------------------ Statistik
def _rank(x):
    from scipy.stats import rankdata

    return rankdata(x)


def spearman(x, y) -> float:
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _resid(y, z):
    Z = np.c_[np.ones(len(z)), z]
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ beta


def partial_spearman(x, y, z) -> float:
    return float(np.corrcoef(_resid(_rank(x), _rank(z)), _resid(_rank(y), _rank(z)))[0, 1])


def perm_p_less(stat_fn, x, y, n_perm: int, seed: int, *extra) -> float:
    rng = np.random.default_rng(seed)
    obs = stat_fn(x, y, *extra)
    if not np.isfinite(obs):  # z. B. konstante Zielgröße: kein Test möglich, nie verwerfen
        return 1.0
    cnt = sum(stat_fn(rng.permutation(x), y, *extra) <= obs for _ in range(n_perm))
    return (cnt + 1) / (n_perm + 1)


def perm_p_partial(J, M, R, n_perm: int, seed: int) -> float:
    rj, rm = _resid(_rank(J), _rank(R)), _resid(_rank(M), _rank(R))
    with np.errstate(all="ignore"):
        obs = np.corrcoef(rj, rm)[0, 1]
    if not np.isfinite(obs):
        return 1.0
    rng = np.random.default_rng(seed)
    cnt = sum(np.corrcoef(rng.permutation(rj), rm)[0, 1] <= obs for _ in range(n_perm))
    return (cnt + 1) / (n_perm + 1)


def boot_ci(fn, arrays, n: int = 10000, seed: int = 2) -> list[float]:
    rng = np.random.default_rng(seed)
    k = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        with np.errstate(all="ignore"):
            v = fn(*[np.asarray(a)[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def holm(ps: dict, alpha: float = 0.05) -> dict:
    order = sorted(ps, key=ps.get)
    out, m, stop = {}, len(ps), False
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        rej = (not stop) and ps[k] <= thr
        stop = stop or not rej
        out[k] = {"p": ps[k], "holm_threshold": thr, "reject_H0": bool(rej)}
    return out


def run(n_scen: int = 60, quick: bool = False, workers: int = 4, n_perm: int = 20000) -> dict:
    import scipy
    import sklearn
    from scipy.stats import wilcoxon

    ests = train_estimators(quick)
    phys = TabSlip(_phys_vec)
    U = select_scenarios(n_scen)
    tabs = [phys] + [e["tab"] for e in ests]
    with Pool(workers, initializer=_init, initargs=(tabs, U)) as pool:
        dts = pool.map(_job, range(len(tabs)))
    phys_dt = np.array(dts[0])
    rows = []
    for e, d in zip(ests, dts[1:]):
        rows.append({"id": e["id"], "alpha": e["alpha"], "hidden": e["hidden"], "sigma": e["sigma"],
                     **estimator_metrics(e["tab"]), "M": float(np.median(d)), "dT": [float(x) for x in d]})
    pm = estimator_metrics(phys)
    J = np.array([r["J"] for r in rows])
    M = np.array([r["M"] for r in rows])
    R = np.array([r["rmse"] for r in rows])
    imax = int(np.argmax(J))
    diff = phys_dt - np.array(rows[imax]["dT"])
    try:
        p3 = float(wilcoxon(phys_dt, rows[imax]["dT"], zero_method="wilcox", alternative="greater").pvalue)
    except ValueError:  # alle Differenzen 0
        p3 = 1.0
    ps = {"H1": perm_p_less(spearman, J, M, n_perm, 0), "H2": perm_p_partial(J, M, R, n_perm, 1), "H3": p3}
    return {
        "preregistration": "experiments/H1_slip_sensitivity/PREREGISTRATION.md", "confirmatory": not quick and n_scen == 60,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__},
        "n_estimators": len(rows), "n_scenarios": len(U),
        "physical": {**pm, "M": float(np.median(phys_dt)), "dT": [float(x) for x in phys_dt]},
        "estimators": rows,
        "effects": {
            "spearman_J_M": spearman(J, M), "spearman_J_M_ci95": boot_ci(spearman, [J, M]),
            "partial_spearman_J_M_given_rmse": partial_spearman(J, M, R),
            "partial_ci95": boot_ci(partial_spearman, [J, M, R]),
            "H3_estimator": rows[imax]["id"], "H3_median_paired_diff_s": float(np.median(diff)),
            "H3_n_nonzero": int(np.sum(diff != 0)),
            "exploratory_spearman_rmse_M": spearman(R, M),
        },
        "tests": holm(ps),
    }



# ------------------------------------------------------------------ H1b: kontrollierte Sensitivität bei festem RMSE
H1B_EPS = 0.002 * 2 ** 0.5
H1B_OMEGA = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
H1B_PHI = (0.0, 2 * np.pi / 3, 4 * np.pi / 3)


def h1b_estimators() -> list[dict]:
    out = []
    for w in H1B_OMEGA:
        for k, ph in enumerate(H1B_PHI):
            fn = (lambda w_, ph_: (lambda A, V: _phys_vec(A, V) + H1B_EPS * np.sin(w_ * np.asarray(A) + ph_)))(w, ph)
            out.append({"id": f"w{w:g}_p{k}", "omega": w, "phi": float(ph), "tab": TabSlip(fn)})
    return out


def run_h1b(n_scen: int = 60, workers: int = 4, n_perm: int = 20000, quick: bool = False) -> dict:
    from scipy.stats import wilcoxon

    ests = h1b_estimators()
    if quick:
        ests = ests[::6]
    phys = TabSlip(_phys_vec)
    U = select_scenarios(n_scen)
    with Pool(workers, initializer=_init, initargs=([phys] + [e["tab"] for e in ests], U)) as pool:
        dts = pool.map(_job, range(len(ests) + 1))
    phys_dt = np.array(dts[0])
    rows = [{"id": e["id"], "omega": e["omega"], "phi": e["phi"], **estimator_metrics(e["tab"]),
             "mean_dT": float(np.mean(d)), "dT": [float(x) for x in d]} for e, d in zip(ests, dts[1:])]
    J = np.array([r["J"] for r in rows])
    Mn = np.array([r["mean_dT"] for r in rows])
    R = np.array([r["rmse"] for r in rows])
    ref = next((r for r in rows if r["omega"] == 16.0 and r["phi"] == 0.0), rows[-1])
    try:
        p2 = float(wilcoxon(phys_dt, ref["dT"], zero_method="wilcox", alternative="greater").pvalue)
    except ValueError:
        p2 = 1.0
    ps = {"H1b-1": perm_p_less(spearman, J, Mn, n_perm, 0), "H1b-2": p2}
    by_w = {f"{w:g}": float(np.mean([r["mean_dT"] for r in rows if r["omega"] == w])) for w in H1B_OMEGA}
    return {
        "preregistration": "experiments/H1b_controlled_sensitivity/PREREGISTRATION.md",
        "confirmatory": not quick and n_scen == 60, "n_estimators": len(rows), "n_scenarios": len(U),
        "physical": {**estimator_metrics(phys), "mean_dT": float(np.mean(phys_dt)), "dT": [float(x) for x in phys_dt]},
        "estimators": rows,
        "effects": {"spearman_J_meandT": spearman(J, Mn), "ci95": boot_ci(spearman, [J, Mn]),
                    "mean_dT_by_omega": by_w, "H1b-2_estimator": ref["id"],
                    "H1b-2_mean_paired_diff_s": float(np.mean(phys_dt - np.array(ref["dT"]))),
                    "exploratory_spearman_rmse_meandT": spearman(R, Mn)},
        "tests": holm(ps),
    }

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scen", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--h1b", action="store_true", help="Folgetest H1b statt H1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.h1b:
        res = run_h1b(a.n_scen, a.workers, 2000 if a.quick else 20000, a.quick)
        a.out = a.out or "experiments/H1b_controlled_sensitivity/result.json"
    else:
        res = run(a.n_scen, a.quick, a.workers, 2000 if a.quick else 20000)
        a.out = a.out or "experiments/H1_slip_sensitivity/result.json"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({k: res[k] for k in ("n_estimators", "n_scenarios", "effects", "tests")}, indent=1))


if __name__ == "__main__":
    main()
