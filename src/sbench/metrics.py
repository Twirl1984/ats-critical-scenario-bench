"""Metriken je Suchlauf: Kritikalität, Diversität, Plausibilität, Effizienz, P_f-Schätzung."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

CLUSTER_THRESHOLD = 0.30  # Complete-Linkage-Schwelle auf u/√d (Einheitswürfel-Distanz)


def _diversity(U: np.ndarray) -> tuple[int, float]:
    if len(U) == 0:
        return 0, 0.0
    if len(U) == 1:
        return 1, 0.0
    X = U / np.sqrt(U.shape[1])
    n_clusters = int(fcluster(linkage(pdist(X), "complete"), CLUSTER_THRESHOLD, "distance").max())
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    return n_clusters, float(D.min(axis=1).mean())


def run_metrics(problem, U, G, NLL, extras: dict, thr: float = 0.0) -> dict:
    """U, G, NLL: Historie in Auswertungsreihenfolge."""
    U, G, NLL = np.asarray(U), np.asarray(G), np.asarray(NLL)
    n = len(G)
    fail_idx = np.flatnonzero(G <= thr)
    plaus = np.zeros(len(fail_idx), bool)
    avoid = np.zeros(len(fail_idx), bool)
    for k, i in enumerate(fail_idx):
        plaus[k], avoid[k] = problem.validity(U[i])
    valid = plaus & avoid
    valid_idx = fail_idx[valid]

    n_clusters, nn_dist = _diversity(U[valid_idx])
    out = {
        "n_evals": n,
        "n_fail": int(len(fail_idx)),
        "n_valid_fail": int(valid.sum()),
        "frac_implausible": float((~plaus).mean()) if len(fail_idx) else 0.0,
        "frac_unavoidable": float((~avoid).mean()) if len(fail_idx) else 0.0,
        "first_fail": int(fail_idx[0]) + 1 if len(fail_idx) else None,
        "first_valid_fail": int(valid_idx[0]) + 1 if len(valid_idx) else None,
        "g_min_valid": float(G[valid_idx].min()) if len(valid_idx) else None,
        "n_clusters_valid": n_clusters,
        "nn_dist_valid": nn_dist,
        "mean_nll_valid": float(NLL[valid_idx].mean()) if len(valid_idx) else None,
        "pf_est_raw": None,
        "pf_est_valid": None,
        "pf_se_valid": None,
        "curve_valid": np.cumsum(np.isin(np.arange(n), valid_idx)).tolist(),
    }
    w = extras.get("weights")
    if w is not None and len(w) == n:
        fail_all = np.zeros(n)
        fail_all[fail_idx] = 1.0
        fail_valid = np.zeros(n)
        fail_valid[valid_idx] = 1.0
        out["pf_est_raw"] = float(np.mean(w * fail_all))
        out["pf_est_valid"] = float(np.mean(w * fail_valid))
        out["pf_se_valid"] = float(np.std(w * fail_valid, ddof=1) / np.sqrt(n))
    return out
