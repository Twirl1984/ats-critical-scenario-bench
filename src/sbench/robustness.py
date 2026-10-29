"""Robustheit gegen Messabweichung (Sensitivität) und zeitliche Abweichung.

Entspricht den Notizen:
  Δ = ∂f/∂w·Δw + ∂f/∂v·Δv + ∂f/∂a·Δa + ∂f/∂slip·Δslip      → ``jacobian`` / ``linear_bounds``
  f_mess vs. f_theor mit ±Δ-Band                              → ``tube_*``
  ΔT, Δf → „Zeit- und Amplitudenschlauch"                     → ``tube_required_dF`` / ``tube_required_dT``
  Szenario-Verdikt unter Sensorabweichungen (flackert es?)    → ``scenario_robustness``
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .sim import DT, SENSOR_FIELDS, SensorConfig, decision_signal

X_NAMES = ("w", "vr", "s", "a")


@dataclass(frozen=True)
class Tolerances:
    """Toleranzen der Sensorik (± Halbbreite). Platzhalterwerte – an reale Datenblätter anpassen."""

    s_bias: float = 0.3  # m
    w_scale: float = 0.005  # relativ
    vr_bias: float = 0.2  # m/s
    a_bias: float = 0.2  # m/s²
    delay: float = 0.04  # s  (ΔT)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in SENSOR_FIELDS}


# ---------------------------------------------------------------- analytische Sensitivität
def jacobian(fn, x: np.ndarray, rel_h: float = 1e-4) -> np.ndarray:
    """Zentrale Differenzen ∂fn/∂xᵢ."""
    x = np.asarray(x, float)
    J = np.zeros(len(x))
    for i in range(len(x)):
        h = rel_h * max(abs(x[i]), 1.0)
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        J[i] = (fn(*xp) - fn(*xm)) / (2 * h)
    return J


def linear_bounds(J: np.ndarray, dx: np.ndarray) -> tuple[float, float]:
    """Worst-Case Σ|Jᵢ|Δxᵢ und statistisch (RSS) √Σ(JᵢΔxᵢ)²."""
    t = np.abs(J) * np.abs(dx)
    return float(t.sum()), float(np.sqrt((t**2).sum()))


def input_tolerances(x: np.ndarray, tol: Tolerances) -> np.ndarray:
    """Δx für x = (w, vr, s, a) aus den Sensortoleranzen."""
    w, vr, s, a = x
    return np.array([tol.w_scale * abs(w), tol.vr_bias, tol.s_bias, tol.a_bias])


def monte_carlo_delta(fn, x, dx, n=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    f0 = fn(*x)
    d = np.array([fn(*(x + rng.uniform(-1, 1, len(x)) * dx)) - f0 for _ in range(n)])
    return {"f0": f0, "max_abs": float(np.abs(d).max()), "p95_abs": float(np.quantile(np.abs(d), 0.95)),
            "std": float(d.std()), "samples": d}


# ---------------------------------------------------------------- Zeit-/Amplitudenschlauch
def _shift_matrix(f_true: np.ndarray, max_shift: int) -> np.ndarray:
    """Spalte j = f_true(t + τⱼ), τⱼ ∈ [-max_shift, max_shift]·dt; Ränder werden gehalten (Edge-Hold)."""
    n = len(f_true)
    padded = np.pad(np.asarray(f_true, float), max_shift, mode="edge")
    shifts = np.arange(-max_shift, max_shift + 1)
    return np.stack([padded[max_shift + sh: max_shift + sh + n] for sh in shifts], axis=1)


def tube_required_dF(f_true, f_meas, dT: float, dt: float = DT) -> float:
    """Minimal nötige Amplitudenbreite Δf, damit f_mess im Schlauch (±ΔT, ±Δf) um f_theor liegt."""
    k = int(round(dT / dt))
    M = _shift_matrix(np.asarray(f_true), k)
    return float(np.max(np.min(np.abs(M - np.asarray(f_meas)[:, None]), axis=1)))


def tube_required_dT(f_true, f_meas, dF: float, dt: float = DT, max_s: float = 1.0) -> float:
    """Minimal nötiger Zeitversatz ΔT bei gegebener Amplitudenbreite Δf (inf, wenn > max_s)."""
    k = int(round(max_s / dt))
    M = _shift_matrix(np.asarray(f_true), k)
    ok = np.abs(M - np.asarray(f_meas)[:, None]) <= dF
    shifts = np.abs(np.arange(-k, k + 1)) * dt
    per_t = np.where(ok, shifts[None, :], np.inf).min(axis=1)
    return float(per_t.max())


def tube_tradeoff(f_true, f_meas, dT_grid=(0.0, 0.02, 0.04, 0.08, 0.16, 0.32), dt: float = DT) -> list[tuple[float, float]]:
    """Kurve (ΔT → nötiges Δf): wie viel Zeitversatz „kauft" wie viel Amplitudentoleranz."""
    return [(float(T), tube_required_dF(f_true, f_meas, T, dt)) for T in dT_grid]


# ---------------------------------------------------------------- Szenario-Robustheit
def _sensor_from(theta: dict, noise: float = 0.0) -> SensorConfig:
    return SensorConfig(noise=noise, **theta)


def scenario_robustness(problem, u, tol: Tolerances = Tolerances(), n_mc: int = 150, seed: int = 0) -> dict:
    """Wie stabil ist das Verdikt (Fehler ja/nein) und der g-Wert unter Sensorabweichungen?"""
    rng = np.random.default_rng(seed)
    sc = problem.decode(u)
    base = problem.simulate(sc, trace=True)
    g0 = base.g
    tol_d = tol.as_dict()

    # (1) Ecken der Toleranzbox (2^5)
    corners = []
    for signs in itertools.product((-1, 1), repeat=len(SENSOR_FIELDS)):
        theta = {k: s * tol_d[k] for k, s in zip(SENSOR_FIELDS, signs)}
        theta["delay"] = abs(theta["delay"])  # Latenz nur ≥ 0
        corners.append(problem.simulate(sc, _sensor_from(theta)).g)
    # (2) Monte-Carlo in Box + nominelles Rauschen
    mc = []
    for i in range(n_mc):
        theta = {k: rng.uniform(-1, 1) * tol_d[k] for k in SENSOR_FIELDS}
        theta["delay"] = abs(theta["delay"])
        mc.append(problem.simulate(sc, _sensor_from(theta, noise=1.0), seed=int(rng.integers(1_000_000))).g)
    mc = np.array(mc)
    p_fail = float(np.mean(mc <= 0))
    nominal_fail = g0 <= 0
    if p_fail >= 0.9:
        verdict = "robust_fail"
    elif p_fail <= 0.1:
        verdict = "robust_pass"
    else:
        verdict = "fragile"

    # (3) Geschlossene-Kreis-Sensitivität dg/dθ·tol (Ranking der dominanten Fehlerquelle)
    closed_loop = {}
    for k in SENSOR_FIELDS:
        gp = problem.simulate(sc, _sensor_from({k: tol_d[k]})).g
        gm = problem.simulate(sc, _sensor_from({k: -tol_d[k] if k != "delay" else 0.0})).g
        closed_loop[k] = float(gp - gm) / (2 if k != "delay" else 1)

    # (4) Offene-Kreis-Sensitivität des Entscheidungssignals am kritischen Zeitpunkt
    tr = base.trace
    open_loop = None
    if tr is not None and len(tr["t"]) > 2:
        above = np.flatnonzero(tr["f_true"] >= 1.0)  # Auslöseschwelle der AEB (Stufe 1)
        k_crit = int(above[0]) if len(above) else int(np.argmax(tr["f_true"]))
        x = np.array([tr["w"][k_crit], tr["vr"][k_crit], tr["s"][k_crit], tr["a"][k_crit]])
        dfn = problem.decision_fn() if hasattr(problem, "decision_fn") else decision_signal
        J = jacobian(dfn, x)
        dx = input_tolerances(x, tol)
        worst, rss = linear_bounds(J, dx)
        mcd = monte_carlo_delta(dfn, x, dx, n=500, seed=seed)
        fdot = np.gradient(tr["f_true"], tr["t"])
        time_term = float(np.abs(fdot[max(k_crit - 1, 0): k_crit + 2]).max() * tol.delay)  # ḟ am Auslösepunkt · ΔT
        meas = problem.simulate(sc, _sensor_from(tol_d), trace=True).trace  # Sensorik am Toleranzrand
        n_common = min(len(tr["f_true"]), len(meas["f_meas"]))
        tube = tube_tradeoff(meas["f_true"][:n_common], meas["f_meas"][:n_common])
        open_loop = {
            "t_crit": float(tr["t"][k_crit]), "f_crit": float(tr["f_true"][k_crit]),
            "jacobian": dict(zip(X_NAMES, J.tolist())), "dx": dict(zip(X_NAMES, dx.tolist())),
            "df_linear_worst": worst, "df_linear_rss": rss, "df_mc_max": mcd["max_abs"], "df_mc_p95": mcd["p95_abs"],
            "linear_covers_mc": bool(mcd["max_abs"] <= worst * 1.05 + 1e-9),
            "df_time_term": time_term, "df_total_worst": worst + time_term, "tube_tradeoff_dT_dF": tube,
        }
    return {
        "g_nominal": float(g0), "nominal_fail": bool(nominal_fail), "verdict": verdict,
        "p_fail_mc": p_fail, "g_corner_min": float(min(corners)), "g_corner_max": float(max(corners)),
        "corner_flip": bool(min(corners) <= 0 < max(corners)), "g_mc_p5": float(np.quantile(mc, 0.05)),
        "g_mc_p95": float(np.quantile(mc, 0.95)), "closed_loop_dg": closed_loop,
        "dominant_sensor": max(closed_loop, key=lambda k: abs(closed_loop[k])), "open_loop": open_loop,
    }
