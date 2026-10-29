import numpy as np
import pytest

from sbench.algos import ALGOS, BudgetExhausted, Evaluator, make_algo
from sbench.problems import ToyBall, make_problem
from sbench.robustness import (Tolerances, jacobian, linear_bounds, scenario_robustness, tube_required_dF,
                               tube_required_dT)
from sbench.sim import SensorConfig, decision_signal, ego_speed_from_wheel, simulate, slip, wheel_speed


def test_wheel_speed_roundtrip():
    for v, a in [(5.0, -8.0), (30.0, -3.0), (20.0, 1.5), (0.5, 0.0)]:
        assert ego_speed_from_wheel(wheel_speed(v, a), a) == pytest.approx(v, rel=1e-6)


def test_sim_deterministic_and_benign_scenario_is_safe():
    p = make_problem("lead_brake")
    u = np.array([0.4, 0.9, 0.5, 0.1, 0.8, 0.9])  # großer Abstand, sanftes Bremsen, trocken
    assert simulate(p.decode(u)).g == simulate(p.decode(u)).g
    assert simulate(p.decode(u)).g > 0


def test_extreme_scenario_collides_and_is_unavoidable():
    p = make_problem("lead_brake")
    u = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # 40 m/s, 10 m Abstand, 9 m/s² bremsend, µ=0.3
    assert simulate(p.decode(u)).collision
    plaus, avoid = p.validity(u)
    assert not plaus and not avoid


def test_sensor_delay_worsens_outcome():
    p = make_problem("lead_brake")
    rng = np.random.default_rng(1)
    worse = 0
    for u in rng.random((40, p.dim)):
        g0, g1 = simulate(p.decode(u)).g, simulate(p.decode(u), SensorConfig(delay=0.3)).g
        assert g1 <= g0 + 1e-6 or g1 > 0  # Latenz verbessert nie ein Ergebnis wesentlich
        worse += g1 < g0 - 1e-6
    assert worse > 0


@pytest.mark.parametrize("name", list(ALGOS))
def test_algorithms_respect_budget(name):
    p = make_problem("lead_brake")
    ev = Evaluator(p, 60)
    try:
        make_algo(name).run(ev, np.random.default_rng(0))
    except BudgetExhausted:
        pass
    assert ev.n == 60
    assert np.all((ev.arrays()[0] >= 0) & (ev.arrays()[0] <= 1))


def test_ce_is_estimates_known_failure_probability():
    p = ToyBall()
    truth = p.true_pf()
    est = []
    for seed in range(8):
        ev = Evaluator(p, 400)
        extras = make_algo("ce_is").run(ev, np.random.default_rng(seed))
        _, G, _ = ev.arrays()
        est.append(np.mean(extras["weights"] * (G <= 0)))
    assert np.mean(est) == pytest.approx(truth, rel=0.4)


def test_jacobian_and_linear_bound_match_finite_difference():
    x = np.array([wheel_speed(25.0, -2.0), -3.0, 30.0, -2.0])
    J = jacobian(decision_signal, x)
    dx = np.array([0.01, 0.2, 0.3, 0.2])
    worst, rss = linear_bounds(J, dx)
    f0 = decision_signal(*x)
    corner = max(abs(decision_signal(*(x + s * dx)) - f0) for s in [np.array(c) for c in
                 [(1, 1, -1, 1), (1, -1, -1, 1), (-1, 1, 1, -1), (1, 1, -1, -1)]])
    assert rss <= worst + 1e-12
    assert corner <= worst * 1.1


def test_tube_trade_off_between_time_and_amplitude():
    t = np.arange(0, 5, 0.02)
    f_true = np.sin(2 * t)
    f_meas = np.interp(t - 0.1, t, f_true)  # reiner Zeitversatz 0.1 s (Rand gehalten)
    assert tube_required_dF(f_true, f_meas, 0.0) > 0.1
    assert tube_required_dF(f_true, f_meas, 0.12) < 0.02
    assert tube_required_dT(f_true, f_meas, 0.01) == pytest.approx(0.1, abs=0.03)


def test_scenario_robustness_report_structure():
    p = make_problem("lead_brake")
    rng = np.random.default_rng(3)
    for u in rng.random((3000, p.dim)):
        if simulate(p.decode(u)).g <= 0 and all(p.validity(u)):
            rep = scenario_robustness(p, u, Tolerances(), n_mc=20)
            assert rep["verdict"] in {"robust_fail", "robust_pass", "fragile"}
            assert rep["open_loop"]["df_linear_rss"] <= rep["open_loop"]["df_linear_worst"] + 1e-12
            return
    pytest.skip("kein gültiger Fehler in der Stichprobe")
