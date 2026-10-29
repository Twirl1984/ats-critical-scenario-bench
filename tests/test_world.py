"""Portierte Tests aus world.rs (AUV-Semantik) plus Fahrzeug-Variante."""
import math

from sbench.world import LidarScan, RangeArray, RangeReading, VehicleWorldModel, WorldModel


def scan(ranges):
    return LidarScan(t=0.0, frame="lidar_front", angle_min=-0.1, angle_max=0.1, angle_inc=0.05,
                     range_min=0.2, range_max=60.0, ranges=ranges)


def ranges(id_, v, rate=None):
    return RangeArray(t=0.0, frame="sonar_fwd", sensors=[RangeReading(id=id_, range_m=v, confidence=0.9, rate_mps=rate)])


def test_nimmt_konservativstes_votum():  # AC-002-1
    w = WorldModel()
    w.on_lidar(scan([4.0, 4.0, 4.0]), 0.0)
    w.on_ranges(ranges("fwd", 3.2), 0.0)
    assert w.obstacle_m(0.0) == 3.2


def test_verwirft_werte_ausserhalb_des_messbereichs():  # AC-002-2
    w = WorldModel()
    w.on_lidar(scan([120.0, 120.0, 120.0]), 0.0)
    assert w.obstacle_m(0.0) is None


def test_veralteter_sensor_faellt_aus_der_bewertung():  # AC-010-3 / GAP-003
    w = WorldModel()
    w.on_lidar(scan([50.0, 50.0, 50.0]), 0.0)
    assert w.obstacle_m(0.0) == 50.0
    assert w.obstacle_m(1.5) is None, "eingefrorener Lidar darf keine freie Fahrt mehr melden"


def test_frischer_sonar_traegt_bei_ausgefallenem_lidar():  # AC-010-1
    w = WorldModel()
    w.on_lidar(scan([50.0, 50.0, 50.0]), 0.0)
    w.on_ranges(ranges("fwd", 8.0), 1.5)
    assert w.obstacle_m(1.5) == 8.0


def test_meldet_veraltete_quellen():
    w = WorldModel()
    w.on_lidar(scan([5.0, 5.0, 5.0]), 0.0)
    w.on_ranges(ranges("fwd", 6.0), 0.0)
    stale = w.stale_sources(1.5)
    assert "lidar" in stale and "sonar" in stale


def test_leerer_scan_ueberschreibt_nicht():
    w = WorldModel()
    w.on_lidar(scan([5.0, 5.0, 5.0]), 0.0)
    w.on_lidar(scan([999.0, 999.0]), 0.0)
    assert w.obstacle_m(0.0) == 5.0


def test_sektor_filtert_seitliche_strahlen():
    s = LidarScan(0.0, "l", -0.5, 0.5, 0.25, 0.2, 60.0, [1.0, 9.0, 9.0, 9.0, 1.0])  # ±0.5 rad Ränder = 1 m
    w = WorldModel()
    w.on_lidar(s, 0.0)
    assert w.obstacle_m(0.0) == 9.0


def test_max_age_ist_parametrierbar():
    w = WorldModel(max_age=0.2)
    w.on_lidar(scan([10.0, 10.0, 10.0]), 0.0)
    assert w.obstacle_m(0.15) == 10.0 and w.obstacle_m(0.25) is None


def test_fahrzeugvariante_kein_echo_heisst_frei_bis_reichweite():
    w = VehicleWorldModel()
    w.on_lidar(scan([999.0, 999.0, 999.0]), 0.0)
    assert w.obstacle_m(0.0) == 60.0


def test_fahrzeugvariante_radarrate_altert():
    w = VehicleWorldModel(max_age=0.5)
    w.on_ranges(ranges("fwd", 30.0, rate=-5.0), 0.0)
    assert w.relative_speed(0.3) == -5.0 and w.relative_speed(0.6) is None
    assert not math.isnan(w.obstacle_m(0.3))
