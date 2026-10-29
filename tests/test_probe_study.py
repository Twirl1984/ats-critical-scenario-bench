import numpy as np

from sbench.probe_study import KEYS, _summary, evaluate_case
from sbench.problems import make_problem
from sbench.robustness import Tolerances


def test_evaluate_case_fields_and_consistency():
    p = make_problem("cut_in_odd")
    tol = Tolerances().as_dict()
    rng = np.random.default_rng(0)
    u = rng.beta(p.beta_a, p.beta_b)
    r = evaluate_case(p, u, tol)
    assert set(KEYS) == set(tol)
    # die vorhergesagte Ecke ist eine der 32 Ecken, also nie besser als deren Minimum
    assert r["g_pred"] >= r["g_min"] - 1e-9
    assert r["flip"] == (r["g0"] > 0 and r["g_min"] <= 0)


def test_summary_counts_flips_and_false_alarms():
    rows = [
        {"g0": 0.5, "g_min": -0.1, "g_pred": -0.1, "g_lin": 0.2, "flip": True, "pred_hit": True, "lin_hit": False, "same_corner": True},
        {"g0": 0.5, "g_min": 0.3, "g_pred": 0.3, "g_lin": -0.1, "flip": False, "pred_hit": False, "lin_hit": True, "same_corner": True},
        {"g0": -0.2, "g_min": -0.5, "g_pred": -0.4, "g_lin": -0.3, "flip": False, "pred_hit": True, "lin_hit": True, "same_corner": False},
    ]
    s = _summary(rows)
    assert s["n_flips"] == 1 and s["found_by_prediction"] == 1 and s["found_by_linear"] == 0
    assert s["false_alarm_prediction"] == 0 and s["false_alarm_linear"] == 1 and s["n_nominal_fail"] == 1
