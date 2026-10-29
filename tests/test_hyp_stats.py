"""Statistik-Bausteine des vorab registrierten Tests (ohne Ergebnis-Lauf)."""
import numpy as np

from sbench.hyp_slip import TabSlip, _phys_vec, holm, partial_spearman, perm_p_less, spearman
from sbench.sim import slip


def test_tabulated_physical_slip_close_to_exact():
    t = TabSlip(_phys_vec)
    for a, v in [(-9.0, 0.3), (-4.2, 7.7), (1.5, 19.0), (0.0, 0.0)]:
        assert abs(t(a, v) - slip(a, v)) < 2e-4


def test_spearman_and_partial():
    x = np.arange(10.0)
    assert abs(spearman(x, -x) + 1.0) < 1e-12
    z = np.random.default_rng(0).normal(size=10)
    assert abs(partial_spearman(x, x + 0.0, z) - 1.0) < 1e-12


def test_permutation_p_detects_strong_negative_and_not_null():
    x = np.arange(20.0)
    assert perm_p_less(spearman, x, -x, 999, 0) < 0.01
    assert perm_p_less(spearman, x, x, 999, 0) > 0.5


def test_holm_stepdown():
    r = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert r["a"]["reject_H0"] and not r["b"]["reject_H0"] and not r["c"]["reject_H0"]


def test_nan_statistic_never_rejects():
    from sbench.hyp_slip import perm_p_partial
    x = np.arange(10.0)
    const = np.ones(10)
    assert perm_p_less(spearman, x, const, 99, 0) == 1.0
    assert perm_p_partial(x, const, x[::-1].copy(), 99, 0) == 1.0
