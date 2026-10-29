"""NSGA-II: Multi-Objective-Suche (Kritikalität g ↓ vs. Unwahrscheinlichkeit nll ↓)."""
from __future__ import annotations

import numpy as np

from .base import Algo


def nd_sort(F: np.ndarray) -> list[np.ndarray]:
    n = len(F)
    dominated_by = [[] for _ in range(n)]
    count = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                dominated_by[i].append(j)
            elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                count[i] += 1
    fronts, cur = [], [i for i in range(n) if count[i] == 0]
    while cur:
        fronts.append(np.array(cur))
        nxt = []
        for i in cur:
            for j in dominated_by[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        cur = nxt
    return fronts


def crowding(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(F[:, k])
        dist[order[[0, -1]]] = np.inf
        span = F[order[-1], k] - F[order[0], k]
        if span > 0 and n > 2:
            dist[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return dist


def _rank_crowd(F):
    rank, crowd = np.zeros(len(F), int), np.zeros(len(F))
    for r, fr in enumerate(nd_sort(F)):
        rank[fr] = r
        crowd[fr] = crowding(F[fr])
    return rank, crowd


class NSGA2(Algo):
    name = "nsga2"

    def __init__(self, pop: int = 40, eta_c: float = 15.0, eta_m: float = 20.0, pc: float = 0.9):
        self.pop, self.eta_c, self.eta_m, self.pc = pop, eta_c, eta_m, pc

    def _sbx(self, p1, p2, rng):
        c1, c2 = p1.copy(), p2.copy()
        if rng.random() < self.pc:
            u = rng.random(len(p1))
            beta = np.where(u <= 0.5, (2 * u) ** (1 / (self.eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1)))
            c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
            c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return np.clip(c1, 0, 1), np.clip(c2, 0, 1)

    def _mutate(self, c, rng):
        d = len(c)
        mask = rng.random(d) < 1.0 / d
        u = rng.random(d)
        delta = np.where(u < 0.5, (2 * u) ** (1 / (self.eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1)))
        return np.clip(np.where(mask, c + delta, c), 0, 1)

    def run(self, ev, rng):
        d = ev.problem.dim
        P = min(self.pop, max(8, ev.budget // 4))
        pop = rng.random((P, d))
        F = np.array([ev(u) for u in pop])
        while ev.remaining > 0:
            rank, crowd = _rank_crowd(F)

            def pick():
                i, j = rng.integers(0, P, 2)
                return i if (rank[i], -crowd[i]) < (rank[j], -crowd[j]) else j

            kids = []
            while len(kids) < P:
                c1, c2 = self._sbx(pop[pick()], pop[pick()], rng)
                kids += [self._mutate(c1, rng), self._mutate(c2, rng)]
            kids = kids[:P]
            kF = []
            for u in kids:
                if ev.remaining <= 0:
                    break
                kF.append(ev(u))
            if not kF:
                break
            allU, allF = np.vstack([pop, kids[: len(kF)]]), np.vstack([F, np.array(kF)])
            sel = []
            for fr in nd_sort(allF):
                if len(sel) + len(fr) <= P:
                    sel += list(fr)
                else:
                    cd = crowding(allF[fr])
                    sel += list(fr[np.argsort(-cd)[: P - len(sel)]])
                    break
            pop, F = allU[sel], allF[sel]
        return {}
