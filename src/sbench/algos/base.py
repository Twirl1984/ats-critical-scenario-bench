"""Gemeinsame Infrastruktur: budgetierter Evaluator und Algorithmus-Basisklasse."""
from __future__ import annotations

import numpy as np


class BudgetExhausted(Exception):
    pass


class Evaluator:
    """Zählt Simulationsaufrufe und protokolliert die Historie (U, g, nll)."""

    def __init__(self, problem, budget: int, constrained: bool = False):
        self.problem = problem
        self.budget = budget
        self.constrained = constrained
        self.U: list[np.ndarray] = []
        self.G: list[float] = []
        self.NLL: list[float] = []

    @property
    def n(self) -> int:
        return len(self.G)

    @property
    def remaining(self) -> int:
        return self.budget - self.n

    def __call__(self, u) -> tuple[float, float]:
        if self.n >= self.budget:
            raise BudgetExhausted
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        g, nll = self.problem.evaluate(u)
        if self.constrained and g <= 0 and not all(self.problem.validity(u)):
            g = 0.05  # ungültiger Fehler = „Beinahe-Fund": kein Treffer, aber weiter als kritisch erkennbar
        self.U.append(u)
        self.G.append(g)
        self.NLL.append(nll)
        return g, nll

    def arrays(self):
        return np.array(self.U), np.array(self.G), np.array(self.NLL)


class Algo:
    name = "base"

    def run(self, ev: Evaluator, rng: np.random.Generator) -> dict:
        """Führt die Suche bis zum Budgetende aus. Rückgabe: optionale Zusatzinfos (z. B. Gewichte)."""
        raise NotImplementedError
