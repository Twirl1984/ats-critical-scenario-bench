"""See-Think-Act-Kette: Sensoren → Weltmodell (SEE) → Planer (THINK) → Fahrzeug (ACT).

Die Sensoren sind hier bewusst einfach (Bias, Rauschen, Latenz, Reichweite, Fehlerinjektion).
Echte Sensormodelle (Radar/Lidar/Kamera, FAU/fiveD, OSI/FMU) werden später hinter derselben
Nachrichtenschnittstelle (``LidarScan``/``RangeArray`` aus ``world.py``) eingehängt.

Fehlerinjektion (Kernfrage aus world.rs):
  stall   – Sensor liefert nichts mehr; die Altersgrenze des Weltmodells fängt das ab.
  repeat  – Treiber veröffentlicht weiter, aber mit eingefrorenem Inhalt und *frischem* Empfangszeitpunkt;
            die Altersgrenze greift hier nicht, nur Querprüfung gegen eine zweite Quelle.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import count

import numpy as np

from .sim import (DT, G, T_END, V_SET, TAU_ACT, TTC_CAP, Scenario, SensorConfig, SimResult, acc_cmd, aeb_stage,
                  ego_speed_from_wheel, rss_ratio, simulate, wheel_speed)
from .world import LidarScan, RangeArray, RangeReading, VehicleWorldModel


@dataclass(frozen=True)
class ChainConfig:
    max_age: float = 0.5  # Altersgrenze des Weltmodells [s] (Rust-Vorlage: 1.0)
    lidar_hz: float = 10.0
    radar_hz: float = 20.0
    lidar_latency: float = 0.05
    radar_latency: float = 0.05
    lidar_range: float = 100.0
    radar_range: float = 150.0
    blind_decel: float = -3.0  # Verhalten ohne belastbare Freiraumaussage („kein Blindflug")
    startup_grace: float = 0.5  # s bis zur ersten Messung
    lidar_sigma: float = 0.05  # m, bei sensor.noise = 1
    radar_sigma: float = 0.30  # m
    radar_rate_sigma: float = 0.10  # m/s


@dataclass(frozen=True)
class Faults:
    t0: float = 0.0
    dur: float = 0.0  # < 0.1 s = kein Fehler
    mode: str = "stall"  # "stall" | "repeat"
    target: str = "lidar"  # "lidar" | "radar" | "both"

    def active(self, t: float, sensor: str) -> bool:
        return self.dur >= 0.1 and self.t0 <= t < self.t0 + self.dur and self.target in (sensor, "both")


@dataclass(frozen=True)
class ChainScenario:
    base: Scenario
    faults: Faults = Faults()

    @property
    def event_time(self) -> float:
        return self.base.event_time


class _Link:
    """Nachrichtenstrecke mit Latenz (Delivery-Queue)."""

    def __init__(self):
        self._q: list = []
        self._n = count()

    def send(self, deliver_at: float, kind: str, payload) -> None:
        heapq.heappush(self._q, (deliver_at, next(self._n), kind, payload))

    def due(self, now: float):
        while self._q and self._q[0][0] <= now + 1e-9:
            yield heapq.heappop(self._q)


def simulate_chain(
    csc: ChainScenario,
    sensor: SensorConfig = SensorConfig(),
    cfg: ChainConfig = ChainConfig(),
    ctrl: str = "aeb",
    trace: bool = False,
    seed: int = 0,
) -> SimResult:
    sc = csc.base
    if ctrl == "oracle":
        return simulate(sc, ctrl="oracle", trace=trace)

    rng = np.random.default_rng(seed) if sensor.noise > 0 else None
    n = int(T_END / DT)
    a_phys = sc.mu * G
    world = VehicleWorldModel(max_age=cfg.max_age)
    link = _Link()
    last_msg: dict[str, object] = {}
    lidar_period = max(1, round(1.0 / (cfg.lidar_hz * DT)))
    radar_period = max(1, round(1.0 / (cfg.radar_hz * DT)))
    extra_lat = sensor.delay

    x_e, v_e, a_act, x_t = 0.0, sc.v_e0, 0.0, sc.x_t0
    latched = 0.0
    min_ttc, min_gap = TTC_CAP, 1e9
    collision, impact = False, 0.0
    blind_s = stale_s = 0.0
    prev_obs: tuple[float, float] | None = None
    rate_fd = 0.0
    tr = {k: [] for k in ("t", "s", "ve", "vt", "a", "w", "vr", "f_true", "f_meas", "ttc")} if trace else None

    for k in range(n):
        t = k * DT
        v_t = sc.target_speed(t)
        p = sc.progress(t)
        visible, overlap = p >= 0.4, sc.p_in <= p < sc.p_out
        s = x_t - x_e
        if overlap and s <= 0.0:
            collision, impact = True, max(v_e - v_t, 0.0)
            break
        vr = v_t - v_e
        w_true = wheel_speed(v_e, a_act)
        nz = sensor.noise

        # ---------------- SEE: Sensoren erzeugen Nachrichten
        if k % lidar_period == 0 and not (csc.faults.active(t, "lidar") and csc.faults.mode == "stall"):
            if csc.faults.active(t, "lidar") and "lidar" in last_msg:
                scan = last_msg["lidar"]  # repeat: eingefrorener Inhalt, frischer Empfangszeitpunkt
            else:
                if visible and s > 0:
                    d = s * (1 + sensor.s_scale) + sensor.s_bias
                    d += rng.normal(0, cfg.lidar_sigma) * nz if rng is not None else 0.0
                else:
                    d = 1e9
                scan = LidarScan(t, "lidar_front", -0.1, 0.1, 0.05, 0.2, cfg.lidar_range, [d, d, d])
                last_msg["lidar"] = scan
            link.send(t + cfg.lidar_latency + extra_lat, "lidar", scan)
        if k % radar_period == 0 and not (csc.faults.active(t, "radar") and csc.faults.mode == "stall"):
            if csc.faults.active(t, "radar") and "radar" in last_msg:
                arr = last_msg["radar"]
            else:
                if visible and s > 0:
                    d = s * (1 + sensor.s_scale) + sensor.s_bias
                    rr = vr + sensor.vr_bias
                    if rng is not None:
                        d += rng.normal(0, cfg.radar_sigma) * nz
                        rr += rng.normal(0, cfg.radar_rate_sigma) * nz
                    rd = RangeReading("fwd", min(d, cfg.radar_range), 0.9, rr)
                else:
                    rd = RangeReading("fwd", cfg.radar_range, 0.9, 0.0)  # frei bis Reichweite
                arr = RangeArray(t, "radar_fwd", [rd])
                last_msg["radar"] = arr
            link.send(t + cfg.radar_latency + extra_lat, "radar", arr)
        for _, _, kind, msg in link.due(t):
            (world.on_lidar if kind == "lidar" else world.on_ranges)(msg, t)

        # Ego-Messung (Raddrehzahl, IMU)
        w_m = w_true * (1.0 + sensor.w_scale) + (rng.normal(0, 0.3) * nz if rng is not None else 0.0)
        a_m = a_act + sensor.a_bias + (rng.normal(0, 0.1) * nz if rng is not None else 0.0)
        v_hat = ego_speed_from_wheel(w_m, a_m)

        # ---------------- THINK
        d_obs = world.obstacle_m(t)
        stale = bool(world.stale_sources(t))
        stale_s += DT * stale
        f_meas = 0.0
        if d_obs is None:
            if t < cfg.startup_grace:
                a_cmd = acc_cmd(v_hat, 0.0, 0.0, False, sc.v_set or V_SET)
            else:
                blind_s += DT
                a_cmd = cfg.blind_decel
                latched = min(latched, 0.0)
        else:
            rate = world.relative_speed(t)
            if rate is None:  # Rückfall: Differenzenquotient über frische Distanzen
                if prev_obs is not None and t > prev_obs[1]:
                    rate_fd = 0.5 * rate_fd + 0.5 * (d_obs - prev_obs[0]) / (t - prev_obs[1])
                rate = rate_fd
            prev_obs = (d_obs, t)
            f_meas = rss_ratio(v_hat, rate, d_obs)
            latched = min(latched, aeb_stage(f_meas))
            if f_meas < 0.6 and rate >= 0.0:
                latched = 0.0
            a_cmd = latched if latched < 0.0 else acc_cmd(v_hat, d_obs, rate, True, sc.v_set or V_SET)

        # ---------------- ACT
        a_cmd = min(max(a_cmd, -a_phys), 2.0)
        closing = v_e - v_t
        ttc = TTC_CAP
        if overlap and s > 0.0:
            min_gap = min(min_gap, s)
            if closing > 0.1:
                ttc = min(s / closing, TTC_CAP)
                min_ttc = min(min_ttc, ttc)
        if tr is not None:
            f_true = rss_ratio(v_e, vr, s) if visible else 0.0
            for key, val in zip(tr.keys(), (t, s, v_e, v_t, a_act, w_true, vr, f_true, f_meas, ttc)):
                tr[key].append(val)
        a_act += (a_cmd - a_act) * DT / TAU_ACT
        a_act = max(a_act, -a_phys)
        v_e = max(v_e + a_act * DT, 0.0)
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
    res.extra = {"blind_s": blind_s, "stale_s": stale_s}
    return res
