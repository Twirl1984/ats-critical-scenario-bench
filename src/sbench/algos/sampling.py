"""Baselines: Zufall, Gitter, Latin Hypercube."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import qmc

from .base import Algo


class RandomSearch(Algo):
    name = "random"

    def run(self, ev, rng):
        while ev.remaining > 0:
            ev(rng.random(ev.problem.dim))
        return {}


class GridSearch(Algo):
    """Vollfaktorielles Zellzentren-Gitter (Reihenfolge gemischt), Restbudget zufällig aufgefüllt."""

    name = "grid"

    def run(self, ev, rng):
        d = ev.problem.dim
        m = max(2, int(ev.budget ** (1.0 / d) + 1e-9))
        axes = (np.arange(m) + 0.5) / m
        pts = np.array(list(itertools.product(axes, repeat=d)))
        rng.shuffle(pts)
        for u in pts[: ev.budget]:
            ev(u)
        while ev.remaining > 0:
            ev(rng.random(d))
        return {}


class LHS(Algo):
    name = "lhs"

    def run(self, ev, rng):
        pts = qmc.LatinHypercube(d=ev.problem.dim, seed=rng).random(ev.budget)
        for u in pts:
            ev(u)
        return {}
