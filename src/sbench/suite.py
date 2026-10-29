"""Sensor-Suite Stufe 1 (phänomenologisch) + Umgebungsbedingungen + Simulation der Gesamtkette.

Stufe 1 laut Blatt „Modellstufen": statistisches Modell je Sensor mit
  * Erkennungswahrscheinlichkeit über Distanz  Pd(s) = pd0 / (1 + exp((s - r50)/w))
  * Distanzrauschen σ(s) = sig0 + sig1·s + sig2·s²   (Kamera: Pinhole → sig2)
  * Doppler-Rate (nur Radar), Latenz, Taktrate, Geisterziele (Rate pro Scan)
  * Seitliche Sichtbarkeit: ab welchem Fortschritt p des Einscherens das Ziel erkannt wird (Sichtfeld)
  * Stationärziel-Filter (Radar): Pd-Faktor für stehende Ziele bei schneller Ego-Fahrt
Umgebung skaliert r50 je Sensorart.  ALLE ZAHLEN SIND PLATZHALTER (Annahmen), keine belegten Werte –
sie müssen je Sensor aus Datenblatt/Messdaten identifiziert werden (Stufe-1-Identifikation).
Fusion: unverändert world.rs-Semantik (konservatives Minimum aller frischen Quellen, kein Tracker).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .chain import _Link
from .sim import (DT, G, T_END, V_SET, A_E, TAU_ACT, TTC_CAP, Scenario, SensorConfig, SimResult, acc_cmd, aeb_stage,
                  ego_speed_from_wheel, rss_ratio, simulate, wheel_speed)
from .world import Aged, LidarScan, RangeArray, RangeReading, VehicleWorldModel


@dataclass(frozen=True)
class SensorSpec:
    name: str
    kind: str  # radar | camera | lidar | ideal
    hz: float
    latency: float
    range_max: float
    r50: float  # Distanz mit Pd = pd0/2 (Standardziel, Tag, klar)
    width: float
    pd0: float
    sig0: float = 0.0
    sig1: float = 0.0
    sig2: float = 0.0
    sig_rate: float | None = None  # None = kein Doppler
    ghost: float = 0.0  # Geisterziel-Wahrscheinlichkeit je Scan
    p_det: float = 0.4  # Fortschritt des Einscherens, ab dem das Ziel gesehen wird (Sichtfeld)
    stat_mult: float = 1.0  # Pd-Faktor für stehende Ziele (Radar-Filter)

    def pd(self, s: float, mult: float = 1.0, stationary: bool = False) -> float:
        if s > self.range_max:
            return 0.0
        x = (s - self.r50 * mult) / self.width
        p = self.pd0 / (1.0 + math.exp(min(max(x, -50), 50)))
        return p * (self.stat_mult if stationary else 1.0)

    def sigma(self, s: float) -> float:
        return self.sig0 + self.sig1 * s + self.sig2 * s * s


RADAR = SensorSpec("radar", "radar", 20, 0.08, 200, 160, 12, 0.99, sig0=0.25, sig1=0.002, sig_rate=0.10,
                   ghost=2e-4, p_det=0.45, stat_mult=0.35)
CAMERA = SensorSpec("camera", "camera", 30, 0.12, 120, 90, 10, 0.97, sig2=6e-4, ghost=1e-4, p_det=0.35)
LIDAR = SensorSpec("lidar", "lidar", 10, 0.05, 100, 70, 8, 0.995, sig0=0.05, p_det=0.40)
IDEAL = SensorSpec("ideal", "ideal", 50, 0.0, 1e9, 1e9, 1.0, 1.0, sig_rate=0.0, p_det=0.40)

SUITES: dict[str, tuple[SensorSpec, ...]] = {
    "ideal": (IDEAL,),
    "radar": (RADAR,),
    "kamera": (CAMERA,),
    "lidar": (LIDAR,),
    "radar+kamera": (RADAR, CAMERA),
    "radar+lidar": (RADAR, LIDAR),
    "radar+kamera+lidar": (RADAR, CAMERA, LIDAR),
}


@dataclass(frozen=True)
class Env:
    name: str
    mult: dict  # Faktor auf r50 je Sensorart (Annahmen!)

    def m(self, kind: str) -> float:
        return self.mult.get(kind, 1.0)


ENVS = {
    "klar_tag": Env("klar_tag", {}),
    "regen_nacht": Env("regen_nacht", {"radar": 0.95, "camera": 0.45, "lidar": 0.6}),
    "nebel": Env("nebel", {"radar": 1.0, "camera": 0.35, "lidar": 0.35}),
    "gegenlicht": Env("gegenlicht", {"camera": 0.5}),
    "dunkles_ziel": Env("dunkles_ziel", {"lidar": 0.6}),
}


class MultiSourceWorldModel(VehicleWorldModel):
    """world.rs-Weltmodell + Kamera-Quelle (Objektebene). Gleiche Semantik: min über frische Quellen."""

    def __init__(self, max_age: float):
        super().__init__(max_age)
        self.camera: Aged | None = None

    def on_ranges(self, arr, at):
        super().on_ranges(arr, at)
        for s in arr.sensors:
            if s.id == "cam":
                self.camera = Aged(s.range_m, at)

    def obstacle_m(self, now):
        vals = [x.fresh(now, self.max_age) for x in (self.front_clearance, self.sonar_fwd, self.camera) if x]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    def stale_sources(self, now):
        out = super().stale_sources(now)
        if self.camera and self.camera.fresh(now, self.max_age) is None:
            out.append("camera")
        return out


def _measure(spec: SensorSpec, rng, env: Env, s, vr, vis_p, stationary, noise, p_vis=0.0, ego: SensorConfig = SensorConfig()):
    """Eine Messung. Rückgabe (Distanz, Rate|None, ist_geist). Distanz=range_max → „frei"."""
    if rng is not None and rng.random() < spec.ghost * noise:
        return float(rng.uniform(8, 50)), float(-rng.uniform(3, 15)), True
    detected = vis_p >= max(spec.p_det, p_vis) and s > 0
    if detected:
        pd = spec.pd(s, env.m(spec.kind), stationary)
        detected = pd >= 1.0 or (rng.random() < pd if rng is not None else pd >= 0.5)
    if not detected:
        return float(min(spec.range_max, 1e4)), None, False
    sg = spec.sigma(s) * noise
    d = s * (1.0 + ego.s_scale) + ego.s_bias + (rng.normal(0, sg) if rng is not None and sg > 0 else 0.0)
    rate = None
    if spec.sig_rate is not None:
        rate = vr + ego.vr_bias + (rng.normal(0, spec.sig_rate) * noise if rng is not None else 0.0)
    return max(d, 0.1), rate, False


def simulate_suite(sc: Scenario, suite=SUITES["radar+kamera+lidar"], env: Env = ENVS["klar_tag"],
                   ego: SensorConfig = SensorConfig(), max_age: float = 0.5, ctrl: str = "aeb",
                   trace: bool = False, seed: int = 0, blind_decel: float = -3.0, grace: float = 0.5,
                   hold_s: float = 0.8, a_e: float = A_E, brake_scale: float = 1.0) -> SimResult:
    if ctrl == "oracle":
        return simulate(sc, ctrl="oracle", trace=trace)
    ideal = all(sp.kind == "ideal" for sp in suite)
    rng = None if ideal and ego.noise == 0 else np.random.default_rng(seed)
    noise = 0.0 if ideal else 1.0
    n = int(T_END / DT)
    a_phys = sc.mu * G
    world = MultiSourceWorldModel(max_age)
    link = _Link()
    nxt = {sp.name: 0.0 for sp in suite}
    x_e, v_e, a_act, x_t = 0.0, sc.v_e0, 0.0, sc.x_t0
    latched, min_ttc, min_gap = 0.0, TTC_CAP, 1e9
    t_trig = -1e9  # Zeitpunkt der letzten Verschärfung des AEB-Latch (Mindesthaltezeit hold_s)
    collision, impact = False, 0.0
    blind_s, ghosts, v_min = 0.0, 0, sc.v_e0
    prev_obs, rate_fd = None, 0.0
    stationary_target = sc.kind == "lead_brake" and sc.v_t0 < 2.0 and sc.decel == 0.0
    tr = {k: [] for k in ("t", "s", "ve", "vt", "a", "w", "vr", "f_true", "f_meas", "ttc")} if trace else None

    for k in range(n):
        t = k * DT
        v_t = sc.target_speed(t)
        p = sc.progress(t)
        overlap = sc.p_in <= p < sc.p_out
        s = x_t - x_e
        if overlap and s <= 0.0:
            collision, impact = True, max(v_e - v_t, 0.0)
            break
        vr = v_t - v_e
        w_true = wheel_speed(v_e, a_act)
        stat = stationary_target and v_e > 15.0

        for sp in suite:
            if t + 1e-9 >= nxt[sp.name]:
                nxt[sp.name] += 1.0 / sp.hz
                d, rate, ghost = _measure(sp, rng, env, s, vr, p, stat, noise, sc.p_vis, ego)
                ghosts += ghost
                at = t + sp.latency + ego.delay
                if sp.kind == "lidar":
                    link.send(at, "lidar", LidarScan(t, "l", -0.1, 0.1, 0.05, 0.2, sp.range_max, [d, d, d]))
                elif sp.kind == "camera":
                    link.send(at, "ranges", RangeArray(t, "c", [RangeReading("cam", d, 0.8, None)]))
                else:  # radar / ideal
                    link.send(at, "ranges", RangeArray(t, "r", [RangeReading("fwd", d, 0.9, rate)]))
        for _, _, kind, msg in link.due(t):
            (world.on_lidar if kind == "lidar" else world.on_ranges)(msg, t)

        nz = noise
        w_m = w_true * (1.0 + ego.w_scale) + (rng.normal(0, 0.3) * nz if rng is not None else 0.0)
        a_m = a_act + ego.a_bias + (rng.normal(0, 0.1) * nz if rng is not None else 0.0)
        v_hat = ego_speed_from_wheel(w_m, a_m)

        d_obs = world.obstacle_m(t)
        f_meas = 0.0
        if d_obs is None:
            if t < grace:
                a_cmd = acc_cmd(v_hat, 0.0, 0.0, False, sc.v_set or V_SET)
            else:
                blind_s += DT
                a_cmd = blind_decel
        else:
            free = d_obs >= max(sp.range_max for sp in suite) - 1e-6
            if free:  # nicht durch „frei" differenzieren (Artefakt): Verlauf zurücksetzen
                prev_obs, rate_fd, rate = None, 0.0, 0.0
            else:
                rate = world.relative_speed(t)
                if rate is None:
                    if prev_obs is not None and t > prev_obs[1]:
                        rate_fd = 0.5 * rate_fd + 0.5 * (d_obs - prev_obs[0]) / (t - prev_obs[1])
                    rate = rate_fd
                prev_obs = (d_obs, t)
            f_meas = 0.0 if free else rss_ratio(v_hat, rate, d_obs, a_e)
            stage = aeb_stage(f_meas) * brake_scale
            if stage < latched:
                t_trig = t
            latched = min(latched, stage)
            if (free or (f_meas < 0.6 and rate >= 0.0)) and t - t_trig >= hold_s:
                latched = 0.0
            a_cmd = latched if latched < 0.0 else acc_cmd(v_hat, d_obs, rate, not free, sc.v_set or V_SET)
        a_cmd = min(max(a_cmd, -a_phys), 2.0)

        closing = v_e - v_t
        ttc = TTC_CAP
        if overlap and s > 0.0:
            min_gap = min(min_gap, s)
            if closing > 0.1:
                ttc = min(s / closing, TTC_CAP)
                min_ttc = min(min_ttc, ttc)
        if tr is not None:
            f_true = rss_ratio(v_e, vr, s, a_e)
            for key, val in zip(tr.keys(), (t, s, v_e, v_t, a_act, w_true, vr, f_true, f_meas, ttc)):
                tr[key].append(val)
        a_act += (a_cmd - a_act) * DT / TAU_ACT
        a_act = max(a_act, -a_phys)
        v_e = max(v_e + a_act * DT, 0.0)
        v_min = min(v_min, v_e)
        if v_e == 0.0 and a_act < 0.0:
            a_act = 0.0
        x_e += v_e * DT
        x_t += v_t * DT
        if v_e <= 0.01 and t > sc.event_time:
            break

    g = -(impact + 0.1) if collision else min_ttc
    if tr is not None:
        tr = {k: np.asarray(v) for k, v in tr.items()}
    res = SimResult(g=g, collision=collision, impact=impact, min_ttc=min_ttc, min_gap=min_gap, trace=tr)
    res.extra = {"blind_s": blind_s, "ghosts": ghosts, "v_drop": sc.v_e0 - v_min}
    return res


def modify(suite, **changes):
    """Parameter aller Sensoren der Suite ändern (für Ablation / Upgrade)."""
    return tuple(replace(sp, **changes) for sp in suite)
