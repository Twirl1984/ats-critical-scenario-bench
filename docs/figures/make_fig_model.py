"""Blockbild Gesamtmodell (mit KI-Stellen) und Prozesskette Evidenz/Validierung, deterministisch gezeichnet.

    python docs/figures/make_fig_model.py      → docs/figures/fig_model_block.png, fig_process_chain.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).parent
PHYS, AI, PLAN, PROC = "#dbe7f3", "#f6dfa4", "#eeeeee", "#e3efe1"
plt.rcParams.update({"font.size": 7.2, "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, title, body="", fc=PHYS, ls="-", lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec="#33475b", ls=ls, lw=lw))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=7.6, weight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=6.3, linespacing=1.35)


def arrow(ax, a, b, text="", rad=0.0, color="#33475b", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.1, text, ha="center", va="bottom", fontsize=6.3,
                color=color)


def model_block():
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.set_xlim(0, 22.4)
    ax.set_ylim(0, 11.2)
    ax.axis("off")
    # Szenario / ODD
    box(ax, 0.2, 7.6, 3.4, 3.2, "Scenario u ∈ U_ODD",
        "pedestrian crossing, cut-in,\nstopped obstacle; v, µ, occlusion\nenvironment e (rain, fog, glare)\nprior p(u) (Beta), likelihood −log p")
    # Abweichungen
    box(ax, 0.2, 3.9, 3.4, 3.2, "Admissible deviations θ ∈ Θ",
        "amplitude: bias b_i, scale ε_i\n(distance, wheel speed,\nrate, IMU)\ntime: latency τ_i, jitter\nfault injection: stall/repeat")
    # SEE
    box(ax, 4.2, 6.4, 4.3, 4.4, "SEE: sensors",
        "y_i(t) = h_i(x(t−τ_i))(1+ε_i) + b_i + n_i\n\nradar · camera · lidar\n(phenomenological, stage 1)\nIMU â, wheel speed ω̃\n\nghost targets, dropouts")
    box(ax, 4.2, 1.4, 4.3, 4.4, "World model",
        "freshness: age ≤ max_age\ns_obs = min over fresh sources\n'no measurement ≠ free path'\n\nego speed estimator\nv̂ = ω̃R / (1 + ŝ(â, v̂))")
    # KI-Block im Schätzer
    box(ax, 4.6, 1.6, 3.5, 1.25, "AI: learned slip map ŝ(a,v)", "MLP (tanh); H1/H1b tests", fc=AI)
    # THINK
    box(ax, 9.2, 6.4, 4.9, 4.4, "THINK: decision",
        "f = d_RSS(v̂, v_r) / s_obs\nd_RSS = v̂ρ + v̂²/2a_e − v_l²/2a_l + s_min\n\nAEB stages (latched)\nf ≥ 1.0 / 1.4 / 2.0 → −4 / −7 / −9 m/s²\nelse ACC: min(k_v(v_set−v̂),\nk_s(s−s_des)+k_r v_r)")
    # ACT
    box(ax, 9.2, 1.4, 4.9, 4.4, "ACT: actuator + vehicle",
        "ȧ = (a_cmd − a)/τ_act,  |a| ≤ min(µg, a_max)\nv̇ = a,  ẋ = v\nwheel: ω = v(1 + s(a,v))/R\ns(a,v) = 0.1(a/g)(1 + 2/(1 + v/5))\n(physical slip in the plant)")
    # Kritikalität / Suche
    box(ax, 15.0, 6.4, 4.0, 4.4, "Criticality g(u, θ)",
        "g = min TTC  (no collision)\ng = −(v_impact + 0.1)  (collision)\n\nvalid failure: plausible and\navoidable by the oracle\nP_f = ∫ 1{g ≤ 0} p(u)p(θ) du dθ")
    box(ax, 15.0, 1.4, 4.0, 4.4, "Sensitivity tube",
        "δf ≲ Σ |∂f/∂x_i| δx_i + |ḟ| δT\n\nworst corner θ* = −sign(∂g/∂θ)·tol\n(7 instead of 32 runs)\nΔT* = largest tolerable latency\nat θ* (bisection)")
    # KI-Spalte rechts
    box(ax, 19.5, 6.4, 2.8, 4.4, "AI in the search", "GP Bayesian optimisation\nNSGA-II (evolutionary)\nREINFORCE adversary\ncross-entropy IS\nrandom-forest surrogate", fc=AI)
    box(ax, 19.5, 1.4, 2.8, 4.4, "AI planned", "learned perception\n(detector / tracker)\nresidual monitor\nlearned controller\n(DeepONet-type)\n5G/GNSS localisation\nspoofing, jamming", fc=PLAN, ls="--")
    # Pfeile
    arrow(ax, (3.6, 9.2), (4.2, 9.2), "x(t)")
    arrow(ax, (3.6, 5.5), (4.2, 7.4), "θ")
    arrow(ax, (6.35, 6.4), (6.35, 5.8), "y_i")
    arrow(ax, (8.5, 4.4), (9.2, 7.6), "")
    ax.text(8.2, 6.0, "s_obs, v̂, v_r", fontsize=6.3, color="#33475b", ha="center")
    arrow(ax, (11.65, 6.4), (11.65, 5.8), "a_cmd")
    arrow(ax, (9.2, 2.2), (8.5, 2.2), "")
    ax.text(8.85, 1.85, "a, ω", ha="center", fontsize=6.3, color="#33475b")
    ax.plot([14.1, 14.55, 14.55], [3.6, 3.6, 8.6], color="#33475b", lw=0.9)
    ax.annotate("", xy=(15.0, 8.6), xytext=(14.55, 8.6), arrowprops=dict(arrowstyle="-|>", color="#33475b", lw=0.9))
    ax.text(14.75, 8.75, "x(t)", fontsize=6.3, ha="center")
    arrow(ax, (17.0, 6.4), (17.0, 5.8), "∂g/∂θ")
    arrow(ax, (19.5, 8.6), (19.0, 8.6), "")
    ax.text(11.0, 0.75, "feedback: the vehicle state x(t) closes the loop to the sensors", ha="center", fontsize=6.5,
            style="italic", color="#33475b")
    arrow(ax, (11.65, 1.4), (3.0, 1.0), "", rad=-0.12, color="#7f8c8d", ls="--")
    # Legende
    for i, (fc, lab, ls) in enumerate([(PHYS, "physical / rule-based model", "-"), (AI, "AI / ML component (implemented)", "-"),
                                       (PLAN, "planned", "--")]):
        ax.add_patch(FancyBboxPatch((0.3 + 4.2 * i, 0.05), 0.5, 0.35, boxstyle="round,pad=0.01", fc=fc, ec="#33475b", ls=ls))
        ax.text(0.95 + 4.2 * i, 0.22, lab, va="center", fontsize=6.6)
    ax.text(0.2, 11.05, "Overall model: see–think–act chain of a low-speed automated shuttle (all parameters assumed)",
            fontsize=8.5, weight="bold", va="top")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_model_block.png", dpi=220)


def process_chain():
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    steps = [
        ("1 Requirements", "EU 2022/1426 (Annex II/III)\nISO 22737, ISO 21448\nUN R157 as analogue\nschema: digital_loop", PLAN),
        ("2 ODD + scenarios", "functional → logical →\nconcrete (Menzel 2018)\nPEGASUS layers as tags\nprior p(u)", PROC),
        ("3 Search space", "u × e × θ\n12-D for the shuttle\ntolerance box Θ\n(assumed, to identify)", PROC),
        ("4 Worst case first", "sensitivities ∂g/∂θ\nprobe corner θ*\n(7 instead of 32 runs)\nΔT* by bisection", PROC),
        ("5 Search", "BayesOpt, NSGA-II,\nREINFORCE, CE-IS,\nsurrogate\n(AI, see model figure)", AI),
        ("6 Validation", "plausible + avoidable\nMonte Carlo, corner scan\nrobust_fail / fragile\nP_f with CI", PROC),
        ("7 Evidence", "P_f, flips, ΔT*, R*\nper requirement\n→ SOTIF argument\n(proposal, not proof)", PROC),
    ]
    w, h, gap = 2.65, 3.0, 0.32
    for i, (t, b, fc) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        box(ax, x, 4.3, w, 2.6, t, b, fc=fc, ls="--" if fc == PLAN else "-")
        if i < len(steps) - 1:
            arrow(ax, (x + w, 5.6), (x + w + gap, 5.6))
    # Credibility-Schiene
    cred = [
        ("Code verification", "unit, golden, property/fuzz,\ndifferential, mutation tests\n(tests/, verification/)", PROC),
        ("Pre-registered experiments", "hypothesis, statistics and\nstop rule committed before\nthe run (experiments/)", PROC),
        ("Model validation", "gap list, oracle limit,\nparameters from measurements,\nresidual monitor (planned)", PLAN),
        ("Credible simulation process", "SSP Traceability 1.0.0:\nanalysis → requirements → design →\nimplementation → execution →\nevaluation → fulfilment", PLAN),
    ]
    w2 = 4.95
    for i, (t, b, fc) in enumerate(cred):
        box(ax, 0.2 + i * (w2 + 0.22), 0.6, w2, 2.3, t, b, fc=fc, ls="--" if fc == PLAN else "-")
    ax.text(2.2, 3.45, "credibility of the evidence (runs alongside every step)", fontsize=7, style="italic", color="#33475b")
    arrow(ax, (19.6, 4.3), (19.6, 2.9), "", color="#7f8c8d", ls="--")
    arrow(ax, (1.5, 2.9), (1.5, 4.3), "", color="#7f8c8d", ls="--")
    ax.text(1.65, 3.45, "trace back", fontsize=6.3, color="#7f8c8d")
    ax.text(0.2, 7.5, "Process chain for evidence and validation (solid: implemented, dashed: planned)", fontsize=8.5,
            weight="bold", va="top")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_process_chain.png", dpi=220)


if __name__ == "__main__":
    model_block()
    process_chain()
    print("ok")
