"""Kompakte, druckgroß lesbare Fassungen von Blockbild und Prozesskette (für 1-seitige Anschreiben)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
PHYS, AI, PLAN, PROC = "#dbe7f3", "#f6dfa4", "#eeeeee", "#e3efe1"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, title, body, fc, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#33475b",
                                lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=6.8, weight="bold")
    ax.text(x + w / 2, y + h - 0.47, body, ha="center", va="top", fontsize=5.7, linespacing=1.3)


def arr(ax, a, b, txt="", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#33475b", ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.08, txt, ha="center", va="bottom", fontsize=5.8)


fig, ax = plt.subplots(figsize=(6.5, 2.05))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.1)
ax.axis("off")
W, G0, Y, H = 2.42, 0.17, 1.5, 2.5
X = [0.02 + i * (W + G0) for i in range(5)]
box(ax, X[0], Y, W, H, "Scenario + deviations", "u ∈ U_ODD, prior p(u)\nθ: bias, scale,\nlatency τ, faults", PHYS)
box(ax, X[1], Y, W, H, "SEE", "y = h(x(t−τ))(1+ε)+b+n\nworld model:\nmin over fresh sources", PHYS)
box(ax, X[1] + 0.12, Y + 0.1, W - 0.24, 0.8, "AI: slip map ŝ(a,v)", "MLP in v̂ = ωR/(1+ŝ)", AI)
box(ax, X[2], Y, W, H, "THINK", "f = d_RSS(v̂, v_r)/s\nstaged braking\n−4/−7/−9 m/s², else ACC", PHYS)
box(ax, X[3], Y, W, H, "ACT + vehicle", "ȧ = (a_cmd − a)/τ_act\nv̇ = a,  |a| ≤ µg\nω = v(1 + s(a,v))/R", PHYS)
box(ax, X[4], Y, W, H, "Criticality + tube", "g = min TTC or −v_imp\nδf ≲ Σ|∂f/∂x|δx + |ḟ|δT\nθ* = −sign(∂g/∂θ)·tol", PHYS)
for i in range(4):
    arr(ax, (X[i] + W, Y + 1.7), (X[i + 1], Y + 1.7))
arr(ax, (X[3] + W / 2, Y), (X[1] + W / 2 + 0.6, Y), "", rad=-0.18, ls="--")
ax.text((X[1] + X[3]) / 2 + W / 2 + 0.3, 0.98, "x(t): closed loop", ha="center", fontsize=5.8, color="#33475b")
for x0, w0, lab, fc, ls in [(0.02, 6.4, "AI in the search: BayesOpt · NSGA-II · REINFORCE · CE-IS · RF", AI, "-"),
                            (6.58, 6.4, "Planned AI: perception · residual monitor · controller", PLAN, "--")]:
    ax.add_patch(FancyBboxPatch((x0, 0.05), w0, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc,
                                ec="#33475b", lw=0.8, ls=ls))
    ax.text(x0 + w0 / 2, 0.3, lab, ha="center", va="center", fontsize=5.9)
fig.tight_layout(pad=0.1)
fig.savefig(OUT / "fig_model_compact.png", dpi=300)

fig, ax = plt.subplots(figsize=(6.5, 1.25))
ax.set_xlim(0, 13)
ax.set_ylim(0, 2.5)
ax.axis("off")
steps = [("Requirements", "2022/1426,\nISO 21448", PLAN), ("ODD +\nscenarios", "functional →\nconcrete", PROC),
         ("Worst case\nfirst", "θ* from 7\nruns", PROC), ("Search", "BayesOpt,\nNSGA-II …", AI),
         ("Validation", "plausible,\navoidable, MC", PROC), ("Evidence", "P_f, ΔT*\nper requirement", PROC)]
w, g = 1.9, 0.3
for i, (t, b, fc) in enumerate(steps):
    x = 0.05 + i * (w + g) - (0 if i else 0)
    ax.add_patch(FancyBboxPatch((x, 0.75), w, 1.6, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#33475b",
                                lw=0.8, ls="--" if fc == PLAN else "-"))
    ax.text(x + w / 2, 2.27, t, ha="center", va="top", fontsize=6.6, weight="bold", linespacing=1.1)
    ax.text(x + w / 2, 1.42, b, ha="center", va="top", fontsize=5.9, linespacing=1.2)
    if i < len(steps) - 1:
        arr(ax, (x + w, 1.55), (x + w + g, 1.55))
ax.text(6.5, 0.32, "credibility alongside: code verification (tests, mutation) · pre-registered experiments · "
        "model validation · credible simulation process", ha="center", fontsize=5.9, style="italic", color="#33475b")
fig.tight_layout(pad=0.1)
fig.savefig(OUT / "fig_process_compact.png", dpi=300)
print("ok")
