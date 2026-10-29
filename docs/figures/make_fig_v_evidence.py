"""V-Modell mit Evidenz je Ebene (Konzept), deterministisch gezeichnet."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.set_xlim(0, 17.2)
ax.set_ylim(0, 8.8)
ax.axis("off")
L = [("Overall system / ODD", 7.6), ("Components (sensors, controller, actuator)", 5.9),
     ("Functions and models", 4.2), ("Code", 2.5)]
R = [("Vehicle-in-the-loop / field", "tubes checked against measurements;\nresidual monitor in operation", 7.6),
     ("System / networked HIL", "end-to-end latency sensor → brake;\ntiming evidence for ΔT", 5.9),
     ("HIL / PIL", "real ECUs and timing; worst-corner\ncases replayed at the bench", 4.2),
     ("SIL", "mass search, worst-case corner from\nsensitivities, P_f with CI", 2.5)]
ax.plot([4.2, 8.6, 13.0], [8.0, 1.2, 8.0], color="#33475b", lw=1.6)
for t, y in L:
    ax.text(4.2 + (8.0 - y) / 6.8 * 4.4 - 0.2, y, t, ha="right", va="center", fontsize=7.4)
for t, ev, y in R:
    x = 13.0 - (8.0 - y) / 6.8 * 4.4 + 0.25
    ax.text(x, y + 0.12, t, ha="left", va="bottom", fontsize=7.6, weight="bold")
    ax.text(x, y + 0.05, ev, ha="left", va="top", fontsize=6.6, color="#1f5f8b", linespacing=1.25)
for y in (7.6, 5.9, 4.2, 2.5):
    xl = 4.2 + (8.0 - y) / 6.8 * 4.4
    xr = 13.0 - (8.0 - y) / 6.8 * 4.4
    ax.annotate("", xy=(xr - 0.1, y), xytext=(xl + 0.1, y), arrowprops=dict(arrowstyle="<->", color="#9aa5b1", lw=0.7, ls="--"))
ax.text(8.6, 0.5, "cross-cutting: requirements traceability · safety (ISO 26262 / ISO 21448) · security "
        "(e.g. spoofing, jamming)", ha="center", fontsize=6.8, style="italic", color="#33475b")
ax.text(0.1, 8.65, "Evidence along the V: what each test level contributes", fontsize=8.8, weight="bold", va="top")
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_v_evidence.png", dpi=220)
print("ok")
