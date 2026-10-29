"""Adversarial Policy Gradient (REINFORCE): lernt eine Gauß-Policy über Szenarioparameter."""
from __future__ import annotations

import numpy as np

from .base import Algo


class AdversarialPG(Algo):
    name = "adv_pg"

    def __init__(self, batch: int = 16, lr_mu: float = 0.4, lr_sigma: float = 0.1, sigma_min: float = 0.15):
        self.batch, self.lr_mu, self.lr_sigma, self.sigma_min = batch, lr_mu, lr_sigma, sigma_min

    def run(self, ev, rng):
        d = ev.problem.dim
        mu, log_s = np.zeros(d), np.zeros(d)
        while ev.remaining > 0:
            s = np.maximum(np.exp(log_s), self.sigma_min)
            z = mu + s * rng.standard_normal((min(self.batch, ev.remaining), d))
            u = 1.0 / (1.0 + np.exp(-z))
            r = np.array([-np.clip(ev(x)[0], -10.0, 10.0) for x in u])
            adv = (r - r.mean()) / (r.std() + 1e-9)
            eps = (z - mu) / s
            mu += self.lr_mu * (adv[:, None] * eps).mean(axis=0)
            log_s += self.lr_sigma * (adv[:, None] * (eps**2 - 1)).mean(axis=0)
            mu = np.clip(mu, -4, 4)
            log_s = np.clip(log_s, np.log(self.sigma_min), 1.0)
        return {}
