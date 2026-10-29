"""Cross-Entropy Importance Sampling (+ optionales Surrogat-Screening).

Proposal q = Produkt aus Betas, per gewichteter Momentenanpassung an Elite-Samples adaptiert.
screen_k = 1  → reiner CE-IS, unverzerrte Schätzung P_f = mean(w · 1[g ≤ 0]), w = p/q.
screen_k > 1  → Random-Forest-Surrogat wählt unter k Kandidaten den kritischsten (mehr Funde pro
                Simulation, aber Schätzer verzerrt → es werden dann keine Gewichte geliefert).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.ensemble import RandomForestRegressor

from .base import Algo


def _logpdf(u, a, b):
    return beta_dist.logpdf(np.clip(u, 1e-6, 1 - 1e-6), a, b).sum(axis=1)


def _fit_beta(u, w):
    w = w / w.sum()
    m = np.clip((w[:, None] * u).sum(0), 0.03, 0.97)
    v = np.maximum((w[:, None] * (u - m) ** 2).sum(0), 2e-3)
    common = np.maximum(m * (1 - m) / v - 1.0, 1.0)
    return np.clip(m * common, 0.6, 60.0), np.clip((1 - m) * common, 0.6, 60.0)


class CrossEntropyIS(Algo):
    def __init__(self, screen_k: int = 1, batch: int = 25, rho: float = 0.2, smooth: float = 0.7,
                 min_screen: int = 40, defensive: float = 0.3):
        self.screen_k, self.batch, self.rho, self.smooth, self.min_screen = screen_k, batch, rho, smooth, min_screen
        self.defensive = defensive  # Anteil Prior-Samples im Proposal → Gewichte ≤ 1/defensive
        self.name = "ce_is" if screen_k == 1 else "surrogate_ce"

    def run(self, ev, rng):
        pa, pb = ev.problem.beta_a, ev.problem.beta_b
        qa, qb = pa.copy(), pb.copy()
        d = ev.problem.dim
        lam = self.defensive
        weights: list[float] = []
        rf = None

        def sample(n):  # Mischung λ·p + (1-λ)·q
            from_p = rng.random(n) < lam
            return np.where(from_p[:, None], rng.beta(pa, pb, size=(n, d)), rng.beta(qa, qb, size=(n, d)))

        while ev.remaining > 0:
            nb = min(self.batch, ev.remaining)
            screen = self.screen_k > 1 and len(ev.G) >= self.min_screen
            if screen:
                U_all, G_all, _ = ev.arrays()
                rf = RandomForestRegressor(60, min_samples_leaf=2, n_jobs=1, random_state=int(rng.integers(1_000_000_000)))
                rf.fit(U_all, np.clip(G_all, -10, 10))
                cand = sample(nb * self.screen_k).reshape(nb, self.screen_k, d)
                pred = rf.predict(cand.reshape(-1, d)).reshape(nb, self.screen_k)
                U = cand[np.arange(nb), pred.argmin(axis=1)]
            else:
                U = sample(nb)
            lp, lq = _logpdf(U, pa, pb), _logpdf(U, qa, qb)
            w = np.exp(lp - np.logaddexp(np.log(lam) + lp, np.log1p(-lam) + lq))
            g = np.array([ev(u)[0] for u in U])
            weights += list(w)
            gamma = max(0.0, float(np.quantile(g, self.rho)))
            elite = g <= gamma
            if elite.sum() >= 2:
                na, nb_ = _fit_beta(U[elite], w[elite])
                qa = self.smooth * na + (1 - self.smooth) * qa
                qb = self.smooth * nb_ + (1 - self.smooth) * qb
        out: dict = {}
        if self.screen_k == 1:
            out["weights"] = np.array(weights)  # ausgerichtet an der Eval-Reihenfolge
        return out
