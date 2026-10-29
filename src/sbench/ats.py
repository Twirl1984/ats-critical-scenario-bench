"""ATS (automatisiertes Transportsystem, z. B. autonomer People Mover/Shuttle Level 4) als Suchproblem.

Der Suchraum wird aus einer ODD-Beschreibung (``AtsOdd``) abgeleitet und enthält neben den Szenarioparametern
  * die Umgebungsbedingung (diskret, über den Prior gewichtet) und
  * die Messabweichungen/Zeitabweichungen innerhalb der Toleranzbox (Messfehleranalyse, ``Tolerances``).
Gesucht werden also kritische Kombinationen *Szenario × Umgebung × zulässige Sensorabweichung* für das Gesamtsystem
(Sensorsuite → Weltmodell → RSS/AEB → Fahrzeug mit Latenzen, siehe ``suite.simulate_suite``).

ALLE ZAHLEN SIND ANNAHMEN. Öffentlich belegt ist nur (2getthere, Seite „3rd generation GRT"): Designanforderung
„60 km/h oder höher"; Sensorik, ODD und Bremsdaten sind öffentlich nicht angegeben und müssen vom Systemhersteller kommen.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .problems import Problem, _beta_logpdf
from .robustness import Tolerances
from .sim import G, Scenario, SensorConfig, SimResult, decision_signal, ego_speed_from_wheel, rss_ratio, simulate
from .suite import ENVS, SUITES, Env, simulate_suite


@dataclass(frozen=True)
class AtsOdd:
    name: str = "Campus/Urban-Shuttle (Annahme)"
    v_max: float = 11.1  # m/s (40 km/h) – Annahme
    pedestrians: bool = True
    occlusion: float = 0.5  # größter Querfortschritt, bei dem ein Fußgänger noch verdeckt sein kann
    ped_speed: tuple = (0.6, 2.5)  # m/s
    ped_distance: tuple = (5.0, 50.0)  # m Abstand zum Querungspunkt beim Losgehen
    mu: tuple = (0.3, 1.0)
    envs: tuple = (("klar_tag", 0.55), ("regen_nacht", 0.20), ("nebel", 0.10), ("gegenlicht", 0.15))  # Prior – Annahme
    lane_cross_m: float = 3.0  # zu querende Breite


@dataclass(frozen=True)
class AtsSystem:
    """Gesamtsystem: Sensorsuite + Weltmodell + Bremsfähigkeit. Bremsdaten: Annahme."""

    suite: str = "radar+kamera+lidar"
    max_age: float = 0.5
    a_e: float = 3.0  # in der Auslegung angenommene Ego-Verzögerung (RSS) [m/s²]
    brake_scale: float = 4.0 / 9.0  # skaliert die AEB-Stufen (−4/−7/−9) auf max. 4 m/s² (Annahme)
    tol: Tolerances = field(default_factory=lambda: Tolerances(s_bias=0.3, w_scale=0.005, vr_bias=0.2, a_bias=0.2, delay=0.05))


PED_MULT = {"radar": 0.3, "camera": 0.8, "lidar": 0.8}  # Zielklasse Fußgänger (kleiner Querschnitt) – Annahme
DEV = ("s_bias", "w_scale", "vr_bias", "a_bias", "delay")


@dataclass(frozen=True)
class AtsCase:
    scenario: Scenario
    env: Env
    dev: SensorConfig  # zulässige Sensor-/Zeitabweichung dieses Falls


def _ats_suite(name: str):
    suite = SUITES[name]
    return tuple(replace(sp, p_det=0.0) if sp.kind == "ideal" else sp for sp in suite)


class _AtsBase(Problem):
    target_ped = False

    def __init__(self, odd: AtsOdd = AtsOdd(), system: AtsSystem = AtsSystem()):
        self.odd, self.system = odd, system
        self._init_space()

    # ---- Raum: Szenario | Umgebung | Abweichungen
    def _scen_space(self):
        raise NotImplementedError

    def _init_space(self):
        n, lo, hi, a, b = self._scen_space()
        self._ns = len(n)
        self.names = list(n) + ["env"] + [f"dev_{d}" for d in DEV]
        self.lows = np.array(list(lo) + [0.0] + [-1, -1, -1, -1, 0])
        self.highs = np.array(list(hi) + [1.0] + [1, 1, 1, 1, 1])
        self.beta_a = np.array(list(a) + [1.0] + [2.0] * 5)
        self.beta_b = np.array(list(b) + [1.0] + [2.0] * 5)
        self.name = f"{self._tag}@{self.system.suite}"

    def _env(self, x: float) -> Env:
        acc = 0.0
        for nm, pr in self.odd.envs:
            acc += pr
            if x <= acc + 1e-12:
                break
        env = ENVS[nm]
        if self.target_ped:
            env = Env(env.name + "+fussg", {k: env.m(k) * m for k, m in PED_MULT.items()})
        return env

    def _dev(self, p: dict) -> SensorConfig:
        t = self.system.tol
        return SensorConfig(s_bias=p["dev_s_bias"] * t.s_bias, w_scale=p["dev_w_scale"] * t.w_scale, vr_bias=p["dev_vr_bias"] * t.vr_bias,
                            a_bias=p["dev_a_bias"] * t.a_bias, delay=p["dev_delay"] * t.delay, noise=1.0)

    def scenario(self, p: dict) -> Scenario:
        raise NotImplementedError

    def decode(self, u) -> AtsCase:
        p = self.params(u)
        env_x = float(np.asarray(u)[self._ns])
        return AtsCase(self.scenario(p), self._env(env_x), self._dev(p))

    # ---- Simulation des Gesamtsystems
    def simulate(self, case: AtsCase, sensor: SensorConfig = SensorConfig(), ctrl: str = "aeb", trace: bool = False, seed: int = 0) -> SimResult:
        if ctrl == "oracle":
            return simulate(case.scenario, ctrl="oracle")
        d = case.dev
        ego = SensorConfig(s_bias=d.s_bias + sensor.s_bias, s_scale=d.s_scale + sensor.s_scale, w_scale=d.w_scale + sensor.w_scale,
                           vr_bias=d.vr_bias + sensor.vr_bias, a_bias=d.a_bias + sensor.a_bias, delay=d.delay + sensor.delay,
                           noise=max(d.noise, sensor.noise))
        sy = self.system
        return simulate_suite(case.scenario, _ats_suite(sy.suite), case.env, ego, sy.max_age, "aeb", trace, seed,
                              a_e=sy.a_e, brake_scale=sy.brake_scale)

    def decision_fn(self):
        a_e = self.system.a_e
        return lambda w, vr, s, a: rss_ratio(ego_speed_from_wheel(w, a), vr, s, a_e)

    def plausible(self, u) -> bool:
        return True

    def avoidable(self, u) -> bool:
        return not simulate(self.decode(u).scenario, ctrl="oracle").collision

    def with_system(self, **kw):
        return type(self)(self.odd, replace(self.system, **kw))


class AtsPedestrian(_AtsBase):
    """Fußgänger quert (ggf. verdeckt) vor dem Shuttle."""

    _tag = "ats_ped"
    target_ped = True

    def _scen_space(self):
        o = self.odd
        names = ["v_ego", "d_ped", "v_ped", "t_cross", "p_vis", "mu"]
        lo = [1.5, o.ped_distance[0], o.ped_speed[0], 0.5, 0.0, o.mu[0]]
        hi = [o.v_max, o.ped_distance[1], o.ped_speed[1], 6.0, o.occlusion, o.mu[1]]
        return names, lo, hi, [2.5, 2.0, 2.0, 1.5, 1.0, 5.0], [2.0, 2.0, 2.0, 1.5, 2.0, 1.5]

    def scenario(self, p):
        dur = self.odd.lane_cross_m / p["v_ped"]
        x_t0 = p["d_ped"] + p["v_ego"] * p["t_cross"]
        return Scenario("cut_in", p["v_ego"], p["mu"], 0.0, x_t0, p["t_cross"], 0.0, dur, v_set=p["v_ego"], p_in=0.25, p_out=0.85, p_vis=p["p_vis"])


class AtsStopped(_AtsBase):
    """Stehendes Hindernis im Fahrweg (Fahrzeug, Baustelle, Gegenstand)."""

    _tag = "ats_stopped"

    def _scen_space(self):
        o = self.odd
        return ["v_ego", "h0", "mu"], [1.5, 1.0, o.mu[0]], [o.v_max, 10.0, o.mu[1]], [2.5, 2.0, 5.0], [2.0, 2.0, 1.5]

    def scenario(self, p):
        return Scenario("lead_brake", p["v_ego"], p["mu"], 0.0, p["v_ego"] * p["h0"], 0.0, 0.0, v_set=p["v_ego"])
