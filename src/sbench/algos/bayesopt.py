"""Bayessche Optimierung (GP + Expected Improvement) zur Minimierung von g."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .base import Algo


class BayesOpt(Algo):
    name = "bayesopt"

    def __init__(self, n_init: int = 20, n_cand: int = 1500, refit_every: int = 10):
        self.n_init, self.n_cand, self.refit_every = n_init, n_cand, refit_every

    def run(self, ev, rng):
        d = ev.problem.dim
        for u in qmc.LatinHypercube(d=d, seed=rng).random(min(self.n_init, ev.budget)):
            ev(u)
        gp, it = None, 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            while ev.remaining > 0:
                U, G, _ = ev.arrays()
                y = np.clip(G, -10.0, 10.0)
                yz = (y - y.mean()) / (y.std() + 1e-9)
                if gp is None or it % self.refit_every == 0:
                    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
                        length_scale=np.full(d, 0.4), length_scale_bounds=(0.05, 10.0), nu=2.5
                    ) + WhiteKernel(1e-3, (1e-6, 1e-1))
                    gp = GaussianProcessRegressor(kernel, random_state=int(rng.integers(1_000_000_000)))
                else:
                    gp = GaussianProcessRegressor(gp.kernel_, optimizer=None)
                gp.fit(U, yz)
                best_pts = U[np.argsort(yz)[:5]]
                local = np.clip(
                    best_pts[rng.integers(0, len(best_pts), self.n_cand // 3)]
                    + rng.normal(0, 0.05, (self.n_cand // 3, d)),
                    0, 1,
                )
                cand = np.vstack([rng.random((self.n_cand - len(local), d)), local])
                mu, sig = gp.predict(cand, return_std=True)
                z = (yz.min() - mu) / np.maximum(sig, 1e-9)
                ei = (yz.min() - mu) * norm.cdf(z) + sig * norm.pdf(z)
                ev(cand[int(np.argmax(ei))])
                it += 1
        return {}
