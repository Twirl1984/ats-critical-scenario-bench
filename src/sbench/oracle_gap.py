"""Gap 0: Wie viele „gültige" Fehler wären mit einem Oracle an der Plattform-Bremsgrenze unvermeidbar?

Das Standard-Oracle bremst ab Ereignis mit µ·g. Hier wird es näherungsweise auf a_max begrenzt, indem nur für das
Oracle der Reibwert auf µ' = min(µ, a_max/g) gesetzt wird.

    python -m sbench.oracle_gap 6000
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .problems import make_problem
from .sim import G, simulate

CASES = (("ats_ped", 4.0), ("ats_ped:v=6.9,brake=6,a_e=4", 6.0))


def study(n: int = 6000, seed: int = 11) -> dict:
    out = {}
    for name, a_max in CASES:
        p = make_problem(name)
        rng = np.random.default_rng(seed)
        fails = valid = valid_lim = 0
        for u in rng.beta(p.beta_a, p.beta_b, size=(n, p.dim)):
            c = p.decode(u)
            if p.simulate(c).g > 0:
                continue
            fails += 1
            if simulate(c.scenario, ctrl="oracle").collision:
                continue
            valid += 1
            if not simulate(replace(c.scenario, mu=min(c.scenario.mu, a_max / G)), ctrl="oracle").collision:
                valid_lim += 1
        out[name] = {"n": n, "a_max": a_max, "fails": fails, "valid_mu_g": valid, "valid_platform_limit": valid_lim,
                     "pf_valid_mu_g": valid / n, "pf_valid_platform_limit": valid_lim / n}
    return out


if __name__ == "__main__":
    res = study(int(sys.argv[1]) if len(sys.argv) > 1 else 6000)
    Path("results_ats").mkdir(exist_ok=True)
    Path("results_ats/oracle_gap.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
