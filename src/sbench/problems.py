"""Suchprobleme: Parameterraum, Prior (Eintrittswahrscheinlichkeit), Zielfunktion g(u), Validität.

Alle Algorithmen arbeiten im Einheitswürfel u ∈ [0,1]^d. g(u) ≤ 0 bedeutet Kollision.
Der Prior ist ein Produkt aus Beta-Verteilungen (Closed-Form-Dichte → Importance Sampling).
"""
from __future__ import annotations

import math

import numpy as np
from dataclasses import replace as _replace

from .chain import ChainConfig, ChainScenario, Faults, simulate_chain
from .sim import G, Scenario, SensorConfig, simulate


def _beta_logpdf(u: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    u = np.clip(u, 1e-9, 1 - 1e-9)
    lnB = np.array([math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y) for x, y in zip(a, b)])
    return (a - 1) * np.log(u) + (b - 1) * np.log1p(-u) - lnB


class Problem:
    name = "base"
    names: list[str]
    lows: np.ndarray
    highs: np.ndarray
    beta_a: np.ndarray
    beta_b: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.names)

    def decode(self, u) -> Scenario:
        raise NotImplementedError

    def params(self, u) -> dict:
        x = self.lows + np.asarray(u) * (self.highs - self.lows)
        return dict(zip(self.names, x))

    def nll(self, u) -> float:
        return float(-_beta_logpdf(np.asarray(u, float), self.beta_a, self.beta_b).sum())

    def simulate(self, sc, sensor: SensorConfig = SensorConfig(), ctrl: str = "aeb", trace: bool = False, seed: int = 0):
        """Einheitlicher Simulationseinstieg (Robustheitsmodul und Metriken gehen hierüber)."""
        return simulate(sc, sensor, ctrl=ctrl, trace=trace, seed=seed)

    def evaluate(self, u) -> tuple[float, float]:
        return self.simulate(self.decode(u)).g, self.nll(u)

    # Validität ------------------------------------------------------------------------
    def plausible(self, u) -> bool:
        """Physikalisch plausibel: Zielverzögerung ≤ Reibgrenze."""
        p = self.params(u)
        return bool(p[self._decel_key] <= 0.98 * p["mu"] * G)

    def avoidable(self, u) -> bool:
        """Vermeidbar durch ein Oracle (perfekte Sensorik, Vollbremsung ab Ereignis)."""
        return not self.simulate(self.decode(u), ctrl="oracle").collision

    def validity(self, u) -> tuple[bool, bool]:
        return self.plausible(u), self.avoidable(u)


class LeadBrake(Problem):
    name = "lead_brake"
    _decel_key = "decel"
    names = ["v_ego", "gap", "v_lead", "decel", "t_brake", "mu"]
    lows = np.array([15.0, 10.0, 5.0, 2.0, 0.5, 0.3])
    highs = np.array([40.0, 100.0, 35.0, 9.0, 8.0, 1.0])
    beta_a = np.array([3.0, 2.0, 2.5, 2.0, 1.5, 5.0])
    beta_b = np.array([2.5, 2.0, 2.5, 4.0, 1.5, 1.5])

    def decode(self, u) -> Scenario:
        p = self.params(u)
        return Scenario("lead_brake", p["v_ego"], p["mu"], p["v_lead"], p["gap"], p["t_brake"], p["decel"])


class LeadBrakeChain(LeadBrake):
    """lead_brake mit voller See-Think-Act-Kette (Weltmodell mit Altersgrenze) und Sensor-Fehlerinjektion.

    Zusätzliche Suchparameter: Beginn/Dauer eines Sensor-Einfrierens, Modus (stall/repeat) und
    betroffener Sensor (Lidar/Radar/beide). Der Prior macht Einfrieren selten (Dauer ~ Beta(1,12)).
    """

    names = LeadBrake.names + ["freeze_t0", "freeze_dur", "freeze_mode", "freeze_target"]
    lows = np.concatenate([LeadBrake.lows, [0.5, 0.0, 0.0, 0.0]])
    highs = np.concatenate([LeadBrake.highs, [8.0, 3.0, 1.0, 1.0]])
    beta_a = np.concatenate([LeadBrake.beta_a, [1.5, 1.0, 1.0, 1.0]])
    beta_b = np.concatenate([LeadBrake.beta_b, [1.5, 12.0, 1.0, 1.0]])

    def __init__(self, max_age: float = 0.5, name: str = "lead_brake_chain"):
        self.name = name
        self.cfg = ChainConfig(max_age=max_age)

    def decode(self, u) -> ChainScenario:
        p = self.params(u)
        base = Scenario("lead_brake", p["v_ego"], p["mu"], p["v_lead"], p["gap"], p["t_brake"], p["decel"])
        target = ("lidar", "radar", "both")[min(int(p["freeze_target"] * 3), 2)]
        faults = Faults(p["freeze_t0"], p["freeze_dur"], "repeat" if p["freeze_mode"] >= 0.5 else "stall", target)
        return ChainScenario(base, faults)

    def simulate(self, sc, sensor: SensorConfig = SensorConfig(), ctrl: str = "aeb", trace: bool = False, seed: int = 0):
        return simulate_chain(sc, sensor, self.cfg, ctrl=ctrl, trace=trace, seed=seed)

    def with_cfg(self, **kw) -> "LeadBrakeChain":
        import copy
        from dataclasses import replace

        new = copy.copy(self)
        new.cfg = replace(self.cfg, **kw)
        return new


class CutIn(Problem):
    name = "cut_in"
    _decel_key = "decel_after"
    names = ["v_ego", "v_cut", "d_cut", "dur", "t_cut", "decel_after", "mu"]
    lows = np.array([15.0, 5.0, 3.0, 0.8, 0.5, 0.0, 0.3])
    highs = np.array([40.0, 35.0, 60.0, 4.0, 6.0, 8.0, 1.0])
    beta_a = np.array([3.0, 2.5, 2.0, 2.0, 1.5, 1.5, 5.0])
    beta_b = np.array([2.5, 2.5, 2.0, 2.0, 1.5, 3.0, 1.5])

    def decode(self, u) -> Scenario:
        p = self.params(u)
        x_t0 = p["d_cut"] + (p["v_ego"] - p["v_cut"]) * p["t_cut"]
        return Scenario("cut_in", p["v_ego"], p["mu"], p["v_cut"], x_t0, p["t_cut"], p["decel_after"], p["dur"])


class StoppedObstacle(Problem):
    """Stehendes Hindernis (z. B. Stauende) voraus: nur v_ego, Sichtweite/Abstand, Reibwert."""

    name = "stopped_obstacle"
    _decel_key = "v_ego"
    names = ["v_ego", "h0", "mu"]  # h0 = anfängliche Zeitlücke zum Hindernis [s] → Abstand = v·h0 (immer im Zeithorizont)
    lows = np.array([15.0, 1.0, 0.3])
    highs = np.array([40.0, 8.0, 1.0])
    beta_a = np.array([3.0, 2.0, 5.0])
    beta_b = np.array([2.5, 2.0, 1.5])

    def decode(self, u) -> Scenario:
        p = self.params(u)
        return Scenario("lead_brake", p["v_ego"], p["mu"], 0.0, p["v_ego"] * p["h0"], 0.0, 0.0, v_set=p["v_ego"])  # Ego fährt im Beharrungszustand

    def plausible(self, u):
        return True


# ------------------------------------------------------------------ ODD-eingeschränkte Varianten (ALKS, UN R157)
V_ODD = 60 / 3.6  # 60 km/h (r157_alks.json: speed_0_to_60_kmh); Wert pro Variante änderbar


class _HoldSpeed:
    """ODD-Varianten: Ego fährt anfangs mit ACC-Wunschgeschwindigkeit = v_ego (Beharrung, keine Beschleunigung ins Unbekannte)."""

    def decode(self, u):
        sc = super().decode(u)
        return _replace(sc, v_set=sc.v_e0)


class LeadBrakeODD(_HoldSpeed, LeadBrake):
    name = "lead_brake_odd"
    lows = np.array([5.0, 5.0, 0.0, 2.0, 0.5, 0.3])
    highs = np.array([V_ODD, 60.0, V_ODD, 6.0, 8.0, 1.0])  # Zielverzögerung ≤ 6 m/s² (R157-5.2.5.1)
    beta_a = np.array([2.5, 2.0, 2.0, 2.0, 1.5, 5.0])
    beta_b = np.array([2.0, 2.0, 2.0, 2.5, 1.5, 1.5])


class CutInODD(_HoldSpeed, CutIn):
    name = "cut_in_odd"
    lows = np.array([5.0, 0.0, 3.0, 0.8, 0.5, 0.0, 0.3])
    highs = np.array([V_ODD, V_ODD, 40.0, 4.0, 6.0, 6.0, 1.0])  # Verzögerung ≤ 6 m/s² (R157-5.2.5.2)
    beta_a = np.array([2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 5.0])
    beta_b = np.array([2.0, 2.5, 2.0, 2.0, 1.5, 3.0, 1.5])


class StoppedObstacleODD(StoppedObstacle):
    name = "stopped_obstacle_odd"
    lows = np.array([5.0, 1.0, 0.3])
    highs = np.array([V_ODD, 10.0, 1.0])
    beta_a = np.array([2.5, 2.0, 5.0])
    beta_b = np.array([2.0, 2.0, 1.5])


class ToyBall(Problem):
    """Analytisch prüfbares 2D-Problem: Fehler = Kugel um (0.8, 0.8). Nur für Tests."""

    name = "toy_ball"
    _decel_key = "x"
    names = ["x", "y"]
    lows = np.zeros(2)
    highs = np.ones(2)
    beta_a = np.array([2.0, 2.0])
    beta_b = np.array([2.0, 2.0])
    radius = 0.2
    center = np.array([0.8, 0.8])

    def evaluate(self, u):
        d = float(np.linalg.norm(np.asarray(u) - self.center))
        return d - self.radius, self.nll(u)

    def plausible(self, u):
        return True

    def avoidable(self, u):
        return True

    def true_pf(self, n: int = 2_000_000, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        u = rng.beta(self.beta_a, self.beta_b, size=(n, 2))
        return float(np.mean(np.linalg.norm(u - self.center, axis=1) <= self.radius))


PROBLEMS = {
    "lead_brake": LeadBrake,
    "cut_in": CutIn,
    "stopped_obstacle": StoppedObstacle,
    "lead_brake_odd": LeadBrakeODD,
    "cut_in_odd": CutInODD,
    "stopped_obstacle_odd": StoppedObstacleODD,
    "toy_ball": ToyBall,
    "lead_brake_chain": lambda: LeadBrakeChain(0.5, "lead_brake_chain"),
    "chain_a02": lambda: LeadBrakeChain(0.2, "chain_a02"),  # Altersgrenze 0.2 s
    "chain_a05": lambda: LeadBrakeChain(0.5, "chain_a05"),
    "chain_a10": lambda: LeadBrakeChain(1.0, "chain_a10"),  # Standardwert aus world.rs
}


def make_problem(name: str) -> Problem:
    """Namen: 'lead_brake' …; ATS: 'ats_ped' oder 'ats_stopped', optional '@<suite>' (z. B. 'ats_ped@radar+kamera')."""
    if name.startswith("ats_"):
        from dataclasses import replace

        from .ats import AtsOdd, AtsPedestrian, AtsStopped, AtsSystem

        head, _, suite = name.partition("@")
        base, _, opts = head.partition(":")
        odd, sy = AtsOdd(), AtsSystem(suite=suite or "radar+kamera+lidar")
        for kv in filter(None, opts.split(",")):  # z. B. ats_ped:v=6.9,a_e=4@radar+kamera
            k, v = kv.split("=")
            if k == "v":
                odd = replace(odd, v_max=float(v))
            elif k == "a_e":
                sy = replace(sy, a_e=float(v))
            elif k == "brake":
                sy = replace(sy, brake_scale=float(v) / 9.0)  # max. Notverzögerung in m/s²
            elif k == "age":
                sy = replace(sy, max_age=float(v))
            else:
                raise KeyError(k)
        p = {"ats_ped": AtsPedestrian, "ats_stopped": AtsStopped}[base](odd, sy)
        p.name = name
        return p
    return PROBLEMS[name]()
