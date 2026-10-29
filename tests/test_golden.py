"""Regressionstest: Ergebnisse vor Einführung der Hooks (oracle_a_max, slip_fn) eingefroren, Standardverhalten unverändert."""
import json
from pathlib import Path

import numpy as np
import pytest

from sbench.problems import make_problem
from sbench.sim import SensorConfig

GOLD = json.loads((Path(__file__).parent / "golden" / "sim_golden.json").read_text())


@pytest.mark.parametrize("name", sorted(GOLD))
def test_golden_unchanged(name):
    p = make_problem(name)
    rng = np.random.default_rng(1234)
    for u, ref in zip(rng.beta(p.beta_a, p.beta_b, size=(25, p.dim)), GOLD[name]):
        sc = p.decode(u)
        r0, r1 = p.simulate(sc), p.simulate(sc, SensorConfig(delay=0.1, a_bias=0.2, s_bias=-0.3))
        got = [round(float(r0.g), 10), bool(r0.collision), round(float(r1.g), 10), bool(r1.collision), bool(p.avoidable(u))]
        assert got == ref
