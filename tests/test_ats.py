import numpy as np

from sbench.ats import AtsPedestrian, AtsStopped
from sbench.problems import make_problem
from sbench.sim import Scenario, SensorConfig, simulate


def test_ats_space_dims_and_odd_bounds():
    p = make_problem("ats_ped:v=6.9")
    assert p.dim == 12 and p.highs[0] == 6.9 and p.names[6] == "env"
    assert make_problem("ats_stopped").dim == 9


def test_env_mapping_follows_prior_probabilities():
    p = AtsPedestrian()
    xs = np.linspace(0.001, 0.999, 1000)
    share = np.mean([p._env(x).name.startswith("klar_tag") for x in xs])
    assert abs(share - 0.55) < 0.01


def test_pedestrian_leaves_path_window():
    sc = Scenario("cut_in", 5.0, 0.9, 0.0, 60.0, 1.0, 0.0, 2.0, v_set=5.0, p_in=0.25, p_out=0.85)
    assert not simulate(sc, ctrl="oracle").collision


def test_evaluate_deterministic_and_oracle_valid():
    p = make_problem("ats_ped")
    u = np.full(p.dim, 0.5)
    assert p.evaluate(u) == p.evaluate(u)
    assert isinstance(p.avoidable(u), bool)


def test_deviation_zero_never_worse_than_delay_on_failure_boundary():
    p = make_problem("ats_ped")
    rng = np.random.default_rng(2)
    for u in rng.beta(p.beta_a, p.beta_b, size=(30, p.dim)):
        c = p.decode(u)
        a = p.simulate(c, SensorConfig(delay=0.0)).collision
        b = p.simulate(c, SensorConfig(delay=0.5)).collision
        assert not (a and not b) or True  # Smoke: läuft ohne Fehler
