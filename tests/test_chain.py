import numpy as np

from sbench.chain import Faults, ChainScenario, simulate_chain
from sbench.problems import LeadBrakeChain, make_problem
from sbench.sim import Scenario, simulate


def _sc(faults=Faults()):
    return ChainScenario(Scenario("lead_brake", 25.0, 0.9, 20.0, 40.0, 2.0, 6.0), faults)


def test_no_fault_matches_simple_sim_verdict():
    p = make_problem("chain_a05")
    rng = np.random.default_rng(1)
    agree = 0
    for _ in range(60):
        u = rng.beta(p.beta_a, p.beta_b)
        u[6:] = 0.0  # kein Fehler
        agree += p.simulate(p.decode(u)).collision == simulate(p.decode(u).base).collision
    assert agree >= 55


def test_stall_blocks_when_age_short_and_not_when_long():
    f = Faults(1.0, 2.0, "stall", "both")
    short = simulate_chain(_sc(f), cfg=LeadBrakeChain(0.2).cfg)
    long_ = simulate_chain(_sc(f), cfg=LeadBrakeChain(5.0).cfg)
    assert short.extra["blind_s"] > long_.extra["blind_s"]


def test_repeat_not_caught_by_age_gate():
    f = Faults(1.0, 2.0, "repeat", "both")
    r = simulate_chain(_sc(f), cfg=LeadBrakeChain(0.2).cfg)
    assert r.extra["blind_s"] == 0.0 and r.extra["stale_s"] == 0.0


def test_repeat_one_sensor_is_masked_by_min_vote_of_other():
    f = Faults(1.0, 3.0, "repeat", "lidar")
    assert not simulate_chain(_sc(f)).collision


def test_oracle_unaffected_by_faults():
    f = Faults(0.0, 3.0, "stall", "both")
    assert simulate_chain(_sc(f), ctrl="oracle").g == simulate(_sc().base, ctrl="oracle").g
