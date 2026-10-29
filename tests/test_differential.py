"""Differenzielle und metamorphe Tests: zwei unabhängige Wege zur selben Größe müssen übereinstimmen."""
import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.optimize import brentq

import sbench.sim as sim
from sbench.probe_study import KEYS, _corner, evaluate_case
from sbench.problems import make_problem
from sbench.robustness import Tolerances
from sbench.sim import G, R_W, Scenario, SensorConfig, ego_speed_from_wheel, simulate, slip

fin = dict(allow_nan=False, allow_infinity=False)


@given(st.floats(0.0, 45.0, **fin), st.floats(-9.5, 2.0, **fin))
def test_fixed_point_vs_root_finder(v, a):
    """Fixpunkt-Iteration (6 Schritte) gegen Brent-Nullstellensuche für v·(1+slip(a,v)) = w·R."""
    w = sim.wheel_speed(v, a)
    if w <= 0:
        return
    v_ref = brentq(lambda x: x * (1 + slip(a, x)) - w * R_W, 0.0, 2 * w * R_W + 1.0, xtol=1e-12)
    assert abs(ego_speed_from_wheel(w, a) - v_ref) <= 1e-6 * max(1.0, v_ref)


def test_slip_fn_physical_equals_default():
    p = make_problem("cut_in_odd")
    for u in np.random.default_rng(3).beta(p.beta_a, p.beta_b, size=(30, p.dim)):
        sc = p.decode(u)
        a = simulate(sc, SensorConfig(a_bias=0.2, delay=0.04))
        b = simulate(sc, SensorConfig(a_bias=0.2, delay=0.04), slip_fn=slip)
        assert (a.g, a.collision) == (b.g, b.collision)


@given(st.floats(1.0, 30.0, **fin), st.floats(0.3, 1.0, **fin), st.floats(5.0, 120.0, **fin))
def test_oracle_limit_at_or_above_friction_is_identity(v, mu, d):
    sc = Scenario("lead_brake", v, mu, 0.0, d, 0.0, 0.0, v_set=v)
    a, b = simulate(sc, ctrl="oracle"), simulate(sc, ctrl="oracle", oracle_a_max=mu * G + 1.0)
    assert (a.g, a.collision) == (b.g, b.collision)


@given(st.floats(3.0, 20.0, **fin), st.floats(0.5, 1.0, **fin), st.floats(5.0, 60.0, **fin), st.floats(2.0, 6.0, **fin))
def test_oracle_limit_monotone(v, mu, d, a_max):
    """Weniger Bremsvermögen kann ein vermeidbares Szenario nur unvermeidbar machen, nie umgekehrt."""
    sc = Scenario("lead_brake", v, mu, 0.0, d, 0.0, 0.0, v_set=v)
    strong, weak = simulate(sc, ctrl="oracle"), simulate(sc, ctrl="oracle", oracle_a_max=a_max)
    assert not (strong.collision and not weak.collision)


def test_probe_corner_vs_brute_force_scan():
    """Die vorhergesagte Ecke ist eine der 32 Ecken: g_pred ≥ Minimum des vollständigen Scans."""
    p = make_problem("cut_in_odd")
    tol = Tolerances().as_dict()
    for u in np.random.default_rng(9).beta(p.beta_a, p.beta_b, size=(15, p.dim)):
        r = evaluate_case(p, u, tol)
        sc = p.decode(u)
        brute = min(simulate(sc, _corner(tol, dict(zip(KEYS[:4], sg), delay=dl))).g
                    for sg in itertools.product((-1, 1), repeat=4) for dl in (0, 1))
        assert abs(brute - r["g_min"]) < 1e-12 and r["g_pred"] >= brute - 1e-12


@pytest.mark.parametrize("name", ["lead_brake_odd", "cut_in_odd", "stopped_obstacle_odd"])
def test_metamorphic_step_size_convergence(monkeypatch, name):
    """Metamorph: halbe Schrittweite darf das Kollisionsurteil nur nahe der Grenze ändern (|g| klein)."""
    p = make_problem(name)
    U = np.random.default_rng(5).beta(p.beta_a, p.beta_b, size=(40, p.dim))
    g1 = [p.simulate(p.decode(u)).g for u in U]
    monkeypatch.setattr(sim, "DT", sim.DT / 2)
    g2 = [p.simulate(p.decode(u)).g for u in U]
    for a, b in zip(g1, g2):
        if (a < 0) != (b < 0):
            assert min(abs(a), abs(b)) < 1.0     # Urteilswechsel nur im Grenzband
