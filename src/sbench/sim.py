"""Leichtgewichtige Längsdynamik-Simulation mit Sensorpfad.

Ego-Fahrzeug mit ACC + gestufter AEB (RSS-basiertes Kriterium) gegen ein Zielfahrzeug
(Vorausfahrender bremst / Cut-in). Der Sensorpfad bildet die Größen aus den Notizen ab:

    f(w, vr, s, a)   mit   v_ego aus Raddrehzahl w und Schlupf = slip(a, v)

Sensorfehler (Bias, Skalierung, Rauschen, Latenz) sind in ``SensorConfig`` parametrierbar.
Alle Modellparameter sind Platzhalter und gedacht zum Austauschen.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

G = 9.81
R_W = 0.31  # Radradius [m]
DT = 0.02
T_END = 12.0
TTC_CAP = 10.0

# Entscheidungslogik (RSS-Kriterium)
RHO = 0.5  # Reaktionszeit [s]
A_E = 6.0  # angenommene Ego-Verzögerung [m/s²]
A_L = 6.0  # angenommene Verzögerung des Vorausfahrenden [m/s²]
S_MIN = 2.0  # Sicherheitspuffer [m]
V_SET = 33.0
TAU_ACT = 0.2  # Aktuator-Verzögerung (Bremse) [s]


# ----------------------------------------------------------------------------- Sensorik
def slip(a: float, v: float) -> float:
    """Schlupf als Funktion von Beschleunigung und Geschwindigkeit (Platzhaltermodell)."""
    return 0.10 * (a / G) * (1.0 + 2.0 / (1.0 + v / 5.0))


def wheel_speed(v: float, a: float) -> float:
    """Raddrehzahl [rad/s] zu Fahrzeuggeschwindigkeit v und Beschleunigung a."""
    return v * (1.0 + slip(a, v)) / R_W


def ego_speed_from_wheel(w: float, a: float, iters: int = 6, slip_fn=None) -> float:
    """Fahrzeuggeschwindigkeit aus Raddrehzahl, Schlupf-kompensiert (Fixpunkt, da slip=f(a,v)).

    ``slip_fn(a, v)`` ersetzt das physikalische Schlupfmodell im *Schätzer* (z. B. ein gelerntes Netz);
    die Strecke selbst rechnet weiter mit ``slip``.
    """
    sf = slip if slip_fn is None else slip_fn
    v = w * R_W
    for _ in range(iters):
        v = w * R_W / (1.0 + sf(a, v))
    return max(v, 0.0)


def rss_ratio(v: float, vr: float, s: float, a_e: float = A_E, a_l: float = A_L) -> float:
    """Verhältnis RSS-Sicherheitsabstand / Ist-Abstand (>1: Abstand unterschritten)."""
    vl = max(v + vr, 0.0)
    d = v * RHO + v * v / (2.0 * a_e) - vl * vl / (2.0 * a_l) + S_MIN
    return max(d, 0.0) / max(s, 0.1)


def decision_signal(w: float, vr: float, s: float, a: float, slip_fn=None) -> float:
    """Das Entscheidungssignal f(w, vr, s, a), auf dem die AEB-Auslösung basiert."""
    return rss_ratio(ego_speed_from_wheel(w, a, slip_fn=slip_fn), vr, s)


def aeb_stage(f: float) -> float:
    if f >= 2.0:
        return -9.0
    if f >= 1.4:
        return -7.0
    if f >= 1.0:
        return -4.0
    return 0.0


def acc_cmd(v: float, s: float, vr: float, target: bool, v_set: float = V_SET) -> float:
    a_cruise = 0.4 * (v_set - v)
    if target:
        gap_des = 3.0 + 1.8 * v
        a_follow = 0.25 * (s - gap_des) + 0.8 * vr
        a = min(a_cruise, a_follow)
    else:
        a = a_cruise
    return min(max(a, -3.0), 2.0)


@dataclass(frozen=True)
class SensorConfig:
    s_bias: float = 0.0  # Abstandsbias [m]
    s_scale: float = 0.0  # relativer Abstandsfehler
    w_scale: float = 0.0  # relativer Raddrehzahlfehler (Radius/Encoder)
    vr_bias: float = 0.0  # Relativgeschwindigkeit-Bias [m/s]
    a_bias: float = 0.0  # IMU-Bias [m/s²]
    delay: float = 0.0  # Latenz [s]
    noise: float = 0.0  # Rausch-Faktor (1.0 = nominal)


SENSOR_FIELDS = ("s_bias", "w_scale", "vr_bias", "a_bias", "delay")


# ----------------------------------------------------------------------------- Szenario
def _ramp_v(v0: float, a: float, tau: float, r: float = 0.3) -> float:
    if tau <= 0.0:
        return v0
    if tau < r:
        v = v0 - a * tau * tau / (2.0 * r)
    else:
        v = v0 - a * (tau - r / 2.0)
    return max(v, 0.0)


@dataclass(frozen=True)
class Scenario:
    kind: str  # "lead_brake" | "cut_in"
    v_e0: float
    mu: float
    v_t0: float
    x_t0: float
    t_ev: float  # Bremsbeginn bzw. Beginn des Spurwechsels
    decel: float  # Zielverzögerung (bei cut_in: nach dem Einscheren)
    dur: float = 0.0  # Dauer Querbewegung (cut_in)
    v_set: float | None = None  # ACC-Wunschgeschwindigkeit; None = globaler V_SET
    p_in: float = 0.5  # Querfortschritt, ab dem das Ziel im Fahrschlauch ist (Kollisionsrelevanz)
    p_out: float = float('inf')  # Querfortschritt, ab dem es den Schlauch wieder verlässt (Fußgänger)
    p_vis: float = 0.0  # Verdeckung: frühester Querfortschritt, ab dem irgendein Sensor das Ziel sehen kann

    def target_speed(self, t: float) -> float:
        if self.kind == "lead_brake":
            return _ramp_v(self.v_t0, self.decel, t - self.t_ev)
        return _ramp_v(self.v_t0, self.decel, t - (self.t_ev + self.dur))

    def progress(self, t: float) -> float:
        if self.kind == "lead_brake":
            return 1.0
        return min(max((t - self.t_ev) / max(self.dur, 1e-6), 0.0), 1.0)

    @property
    def event_time(self) -> float:
        return self.t_ev if self.kind == "lead_brake" else self.t_ev + min(0.4, self.p_in) * self.dur


@dataclass
class SimResult:
    g: float
    collision: bool
    impact: float
    min_ttc: float
    min_gap: float
    trace: dict | None = None
    extra: dict | None = None


def simulate(
    sc: Scenario,
    sensor: SensorConfig = SensorConfig(),
    ctrl: str = "aeb",
    trace: bool = False,
    seed: int = 0,
    oracle_a_max: float | None = None,
    slip_fn=None,
) -> SimResult:
    """Simuliert ein Szenario. g = -(Aufprallgeschwindigkeit+0.1) bei Kollision, sonst min. TTC.

    ``oracle_a_max``: Bremsgrenze der Plattform für das Oracle (None = Reibgrenze µ·g, bisheriges Verhalten).
    ``slip_fn``: Schlupfmodell im Geschwindigkeitsschätzer (None = physikalisches Modell).
    """
    n = int(T_END / DT)
    rng = np.random.default_rng(seed) if sensor.noise > 0 else None
    nd = int(round(sensor.delay / DT))
    buf: deque = deque(maxlen=nd + 1)
    a_phys = sc.mu * G
    tau = TAU_ACT if ctrl == "aeb" else 0.05

    x_e, v_e, a_act, x_t = 0.0, sc.v_e0, 0.0, sc.x_t0
    latched = 0.0
    min_ttc, min_gap = TTC_CAP, 1e9
    collision, impact = False, 0.0
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

        # Messung
        if rng is not None:
            nz = sensor.noise
            ns, nv = rng.normal(0, 0.15 + 0.003 * abs(s)) * nz, rng.normal(0, 0.15) * nz
            nw, na = rng.normal(0, 0.3) * nz, rng.normal(0, 0.1) * nz
        else:
            ns = nv = nw = na = 0.0
        s_m = s * (1.0 + sensor.s_scale) + sensor.s_bias + ns
        vr_m = vr + sensor.vr_bias + nv
        w_m = w_true * (1.0 + sensor.w_scale) + nw
        a_m = a_act + sensor.a_bias + na
        buf.append((visible, w_m, a_m, s_m, vr_m))
        m_vis, m_w, m_a, m_s, m_vr = buf[0]

        f_true = rss_ratio(v_e, vr, s) if visible else 0.0
        f_meas = decision_signal(m_w, m_vr, m_s, m_a, slip_fn) if m_vis else 0.0

        if ctrl == "oracle":
            a_brk = a_phys if oracle_a_max is None else min(a_phys, oracle_a_max)
            a_cmd = -a_brk if t >= sc.event_time else acc_cmd(v_e, s, vr, visible, sc.v_set or V_SET)
        else:
            stage = aeb_stage(f_meas)
            latched = min(latched, stage)
            if f_meas < 0.6 and m_vr >= 0.0:
                latched = 0.0
            if latched < 0.0:
                a_cmd = latched
            else:
                v_hat = ego_speed_from_wheel(m_w, m_a, slip_fn=slip_fn)
                a_cmd = acc_cmd(v_hat, m_s, m_vr, m_vis, sc.v_set or V_SET)
        a_cmd = min(max(a_cmd, -a_phys), 2.0)

        # TTC / Mindestabstand
        closing = v_e - v_t
        ttc = TTC_CAP
        if overlap and s > 0.0:
            min_gap = min(min_gap, s)
            if closing > 0.1:
                ttc = min(s / closing, TTC_CAP)
                min_ttc = min(min_ttc, ttc)

        if tr is not None:
            for key, val in zip(tr.keys(), (t, s, v_e, v_t, a_act, w_true, vr, f_true, f_meas, ttc)):
                tr[key].append(val)

        a_act += (a_cmd - a_act) * DT / tau
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
    return SimResult(g=g, collision=collision, impact=impact, min_ttc=min_ttc, min_gap=min_gap, trace=tr)
