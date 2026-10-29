"""Property-/Fuzz-Tests (Hypothesis, derandomisiert): Invarianten, die für *jede* Eingabe gelten müssen."""
import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sbench.problems import PROBLEMS, make_problem
from sbench.sim import (G, TTC_CAP, Scenario, SensorConfig, acc_cmd, aeb_stage, ego_speed_from_wheel, rss_ratio,
                        simulate, wheel_speed)
from sbench.world import Aged, LidarScan, RangeArray, RangeReading, WorldModel

fin = dict(allow_nan=False, allow_infinity=False)
scen = st.builds(
    lambda kind, v, mu, vt, x, tev, dec, dur, pv: Scenario(kind, v, mu, vt, x, tev, dec, dur, v_set=v, p_vis=pv),
    st.sampled_from(["lead_brake", "cut_in"]), st.floats(0.0, 45.0, **fin), st.floats(0.1, 1.2, **fin),
    st.floats(0.0, 45.0, **fin), st.floats(0.5, 200.0, **fin), st.floats(0.0, 8.0, **fin),
    st.floats(0.0, 10.0, **fin), st.floats(0.1, 5.0, **fin), st.floats(0.0, 0.9, **fin))
sens = st.builds(SensorConfig, s_bias=st.floats(-5, 5, **fin), s_scale=st.floats(-0.3, 0.3, **fin),
                 w_scale=st.floats(-0.1, 0.1, **fin), vr_bias=st.floats(-3, 3, **fin), a_bias=st.floats(-3, 3, **fin),
                 delay=st.floats(0, 1.0, **fin), noise=st.sampled_from([0.0, 1.0, 3.0]))


@given(scen, sens, st.sampled_from(["aeb", "oracle"]), st.integers(0, 2**31 - 1))
def test_fuzz_simulate_result_contract(sc, se, ctrl, seed):
    r = simulate(sc, se, ctrl=ctrl, seed=seed)
    assert math.isfinite(r.g) and r.g <= TTC_CAP
    assert r.collision == (r.g < 0)
    assert r.impact >= 0 and (r.collision or r.impact == 0)
    if r.collision:
        assert math.isclose(r.g, -(r.impact + 0.1))


@given(scen, sens, st.integers(0, 1000))
def test_simulate_deterministic(sc, se, seed):
    a, b = simulate(sc, se, seed=seed), simulate(sc, se, seed=seed)
    assert (a.g, a.collision, a.impact) == (b.g, b.collision, b.impact)


@given(st.floats(0.0, 45.0, **fin), st.floats(-9.5, 2.0, **fin))
def test_wheel_speed_roundtrip(v, a):
    assert abs(ego_speed_from_wheel(wheel_speed(v, a), a) - v) <= 1e-6 * max(1.0, v)


@given(st.floats(0, 45, **fin), st.floats(0, 5, **fin), st.floats(-20, 20, **fin), st.floats(0.5, 200, **fin))
def test_rss_ratio_monotone(v, dv, vr, s):
    # schnelleres Ego bei fester Geschwindigkeit des Vorausfahrenden → kritischer
    # (bei festem vr ist RSS nicht monoton in v, wenn der Vorausfahrende schneller ist: physikalisch korrekt)
    assert rss_ratio(v + dv, vr - dv, s) >= rss_ratio(v, vr, s) - 1e-12
    assert rss_ratio(v, vr, s + dv) <= rss_ratio(v, vr, s) + 1e-12   # mehr Abstand → unkritischer


@given(st.floats(-10, 10, **fin), st.floats(0, 5, **fin))
def test_aeb_stage_monotone_and_discrete(f, df):
    assert aeb_stage(f + df) <= aeb_stage(f)
    assert aeb_stage(f) in (0.0, -4.0, -7.0, -9.0)


@given(st.floats(0, 45, **fin), st.floats(-100, 300, **fin), st.floats(-40, 40, **fin), st.booleans())
def test_acc_cmd_bounded(v, s, vr, tgt):
    assert -3.0 <= acc_cmd(v, s, vr, tgt) <= 2.0


@given(st.floats(1.0, 40.0, **fin), st.floats(0.3, 1.0, **fin), st.floats(1.05, 3.0, **fin))
def test_oracle_stops_before_far_obstacle(v, mu, k):
    """Physik: Anhalteweg v²/(2µg) + Totzeit-Anteil < Abstand ⇒ das Oracle kollidiert nie."""
    d = k * (v * v / (2 * mu * G) + v * 0.1) + 1.0
    assert not simulate(Scenario("lead_brake", v, mu, 0.0, d, 0.0, 0.0, v_set=v), ctrl="oracle").collision


@given(st.sampled_from(sorted(PROBLEMS) + ["ats_ped", "ats_stopped"]), st.data())
def test_problem_params_in_bounds_and_nll_finite(name, data):
    p = make_problem(name)
    u = np.array(data.draw(st.lists(st.floats(1e-6, 1 - 1e-6, **fin), min_size=p.dim, max_size=p.dim)))
    x = np.array(list(p.params(u).values()))
    assert np.all(x >= p.lows - 1e-9) and np.all(x <= p.highs + 1e-9)
    assert math.isfinite(p.nll(u))


@given(st.floats(0, 10, **fin), st.floats(0, 3, **fin), st.floats(0.05, 2.0, **fin))
def test_aged_fresh_boundary(at, dt, max_age):
    a = Aged(1.0, at)
    fresh = a.fresh(at + dt, max_age) is not None
    if abs(dt - max_age) > 1e-6:          # abseits der Grenze: exakte Semantik
        assert fresh == (dt < max_age)
    if dt == max_age:                     # genau auf der Grenze: inklusive, wie Duration in world.rs
        assert fresh


@given(st.lists(st.one_of(st.floats(-1, 100, **fin), st.just(float("nan"))), min_size=0, max_size=12),
       st.floats(0, 5, **fin))
def test_world_model_empty_scan_is_no_measurement(ranges, now):
    """AUV-Semantik: ein Scan ohne gültigen Wert überschreibt nichts; ohne frische Quelle → None (nie 'frei')."""
    w = WorldModel(max_age=1.0)
    w.on_lidar(LidarScan(0.0, "f", -0.1, 0.1, 0.05, 0.2, 60.0, ranges), at=0.0)
    ob = w.obstacle_m(now)
    valid = [r for i, r in enumerate(ranges) if abs(-0.1 + 0.05 * i) <= 0.35 and 0.2 <= r <= 60.0]
    if not valid or now > 1.0:
        assert ob is None
    else:
        assert ob == min(valid)
    w.on_ranges(RangeArray(0.0, "s", [RangeReading("fwd", 3.0)]), at=now)
    assert w.obstacle_m(now) == (3.0 if ob is None else min(3.0, ob))
