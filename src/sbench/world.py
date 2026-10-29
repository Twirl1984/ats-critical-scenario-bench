"""SEE: konsolidiertes Weltmodell aus allen Sensorquellen.

Python-Portierung von ``world.rs`` (auv-project/src/see-think-rust). Zentrale Eigenschaft bleibt:
Jede Quelle trägt einen Empfangszeitpunkt. Ein Sensor, der einfriert, liefert keinen Fehler,
er wiederholt stillschweigend seinen letzten Wert. Ohne Altersgrenze meldet das Weltmodell dann
weiterhin freie Fahrt (AUV-SYS-026).

Unterschiede zur Rust-Vorlage (bewusst, durch ``VehicleWorldModel``):
  * Simulationszeit als float [s] statt ``Instant``.
  * Straßenfahrzeug: Ein gültiger Scan ohne Echo bedeutet „frei bis range_max", nicht „keine Messung".
    (Beim AUV gilt das Gegenteil: ohne Echo wird gestoppt, siehe ``WorldModel.on_lidar``.)
  * Zusätzlich Relativgeschwindigkeit aus Radar-Doppler (``RangeReading.rate_mps``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

T = TypeVar("T")

MAX_SENSOR_AGE = 1.0  # s, Standard wie in world.rs


@dataclass(frozen=True)
class Aged(Generic[T]):
    """Ein Messwert mit Empfangszeitpunkt."""

    value: T
    at: float

    def fresh(self, now: float, max_age: float) -> Optional[T]:
        return self.value if now - self.at <= max_age else None

    def age(self, now: float) -> float:
        return now - self.at


# ------------------------------------------------------------------------- Nachrichten (msgs.rs)
@dataclass
class Pose:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0


@dataclass
class LidarScan:
    t: float
    frame: str
    angle_min: float
    angle_max: float
    angle_inc: float
    range_min: float
    range_max: float
    ranges: list


@dataclass
class RangeReading:
    id: str
    range_m: float
    confidence: float = 0.0
    rate_mps: Optional[float] = None  # Erweiterung: Radial­geschwindigkeit (Doppler), <0 = Annäherung


@dataclass
class RangeArray:
    t: float
    frame: str
    sensors: list = field(default_factory=list)


# ------------------------------------------------------------------------- Weltmodell
class WorldModel:
    """Port von ``WorldModel`` (AUV-Semantik: Scan ohne gültigen Wert = keine Messung)."""

    SECTOR = 0.35  # rad, ~20 Grad

    def __init__(self, max_age: float = MAX_SENSOR_AGE):
        self.max_age = max_age
        self.pose = Pose()
        self.front_clearance: Optional[Aged] = None
        self.sonar_fwd: Optional[Aged] = None
        self.altitude: Optional[Aged] = None
        self.last_camera = None

    # -- Eingänge
    def _scan_min(self, scan: LidarScan) -> float:
        m = math.inf
        for i, r in enumerate(scan.ranges):
            a = scan.angle_min + scan.angle_inc * i
            if abs(a) <= self.SECTOR and scan.range_min <= r <= scan.range_max:
                m = min(m, float(r))
        return m

    def on_lidar(self, scan: LidarScan, at: float) -> None:
        m = self._scan_min(scan)
        if math.isfinite(m):  # leerer Scan überschreibt nicht, der alte Wert altert weiter
            self.front_clearance = Aged(m, at)

    def on_ranges(self, arr: RangeArray, at: float) -> None:
        for s in arr.sensors:
            if s.id in ("fwd", "forward"):
                self.sonar_fwd = Aged(s.range_m, at)
            elif s.id in ("down", "altimeter"):
                self.altitude = Aged(s.range_m, at)

    # -- Ausgänge
    def obstacle_m(self, now: float) -> Optional[float]:
        """Konservativstes Votum aller *frischen* Quellen. None = keine belastbare Aussage."""
        lidar = self.front_clearance.fresh(now, self.max_age) if self.front_clearance else None
        sonar = self.sonar_fwd.fresh(now, self.max_age) if self.sonar_fwd else None
        if lidar is not None and sonar is not None:
            return min(lidar, sonar)
        return lidar if lidar is not None else sonar

    def altitude_m(self, now: float) -> Optional[float]:
        return self.altitude.fresh(now, self.max_age) if self.altitude else None

    def has_any_clearance_source(self) -> bool:
        return self.front_clearance is not None or self.sonar_fwd is not None

    def stale_sources(self, now: float) -> list[str]:
        out = []
        if self.front_clearance and self.front_clearance.fresh(now, self.max_age) is None:
            out.append("lidar")
        if self.sonar_fwd and self.sonar_fwd.fresh(now, self.max_age) is None:
            out.append("sonar")
        if self.altitude and self.altitude.fresh(now, self.max_age) is None:
            out.append("ultrasonic")
        return out


class VehicleWorldModel(WorldModel):
    """Straßenfahrzeug-Variante: „kein Echo" = frei bis range_max; Radar-Doppler als Annäherungsrate."""

    def __init__(self, max_age: float = MAX_SENSOR_AGE):
        super().__init__(max_age)
        self.rate: Optional[Aged] = None  # Relativgeschwindigkeit des Ziels (<0: nähert sich)

    def on_lidar(self, scan: LidarScan, at: float) -> None:
        m = self._scan_min(scan)
        if math.isfinite(m):
            self.front_clearance = Aged(m, at)
        elif scan.ranges:  # gültiger Scan, aber keine Echos → frei bis range_max
            self.front_clearance = Aged(float(scan.range_max), at)

    def on_ranges(self, arr: RangeArray, at: float) -> None:
        super().on_ranges(arr, at)
        for s in arr.sensors:
            if s.id in ("fwd", "forward") and s.rate_mps is not None:
                self.rate = Aged(s.rate_mps, at)

    def relative_speed(self, now: float) -> Optional[float]:
        return self.rate.fresh(now, self.max_age) if self.rate else None
