"""Abbildung 'Preliminary evidence' (FAU Research Statement) aus gespeicherten Ergebnissen, deterministisch."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "docs" / "figures" / "fig_evidence.png"
pr = json.loads((R / "results_ats" / "probe_study.json").read_text())["problems"]
flips = sum(v[s]["n_flips"] for v in pr.values() for s in ("prior", "edge"))
pred = sum(v[s]["found_by_prediction"] for v in pr.values() for s in ("prior", "edge"))
lin = sum(v[s]["found_by_linear"] for v in pr.values() for s in ("prior", "edge"))
h = json.loads((R / "experiments" / "H1b_controlled_sensitivity" / "result.json").read_text())

plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 1.9), gridspec_kw={"width_ratios": [1, 1.35]})
names = ["brute force\n32 runs", "probe\n7 runs", "linear\n6 runs"]
vals = [flips, pred, lin]
bars = a.bar(names, vals, color=["#9aa5b1", "#1f5f8b", "#c9a227"], width=0.6)
for bb, v in zip(bars, vals):
    a.text(bb.get_x() + bb.get_width() / 2, v + 0.3, f"{v}/{flips}", ha="center", fontsize=7)
a.set_ylabel("tolerance flips found")
a.set_ylim(0, flips + 3)
n_sc = sum(v[s]["n_valid"] for v in pr.values() for s in ("prior", "edge"))
a.set_title(f"(a) worst tolerance corner, {n_sc} scenarios", fontsize=7.5, loc="left")
J = [e["J"] for e in h["estimators"]]
M = [e["mean_dT"] for e in h["estimators"]]
b.scatter(J, M, s=10, color="#1f5f8b", label="learned slip map, RMSE ≈ 0.002")
b.scatter([h["physical"]["J"]], [h["physical"]["mean_dT"]], s=28, marker="D", color="#c0392b", label="physical slip map")
b.set_xlabel("sensitivity J = p95 |∂ŝ/∂a| of the learned sub-model")
b.set_ylabel("tolerable latency ΔT* [s]")
b.set_ylim(0.30, 0.48)
t = h["tests"]["H1b-1"]
b.set_title(f"(b) closed loop, 60 cut-in edge cases: ρ = {h['effects']['spearman_J_meandT']:+.2f}, p = {t['p']:.2f}",
            fontsize=7.5, loc="left")
b.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(pad=0.3, w_pad=1.2)
fig.savefig(out, dpi=300)
print(out, flips, pred, lin)
