"""Phase error of each GPU precision option against orbits (ADR 0006).

Reads docs/adr/0006/g1_none_series.csv and g1_both_series.csv (written by
`cargo run -p fring-gpu --release --example g1_precision`) and writes
docs/adr/0006/g1_phase.png: RMS phase difference GPU - CPU for the core
orbit (e = 0.0024), without and with the moons, with the CPU's own error
(step h against h/10) and the PLAN section 10 tolerance.

Run: uv run --no-project --with matplotlib python3 docs/adr/0006/plot_g1.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
# Categorical slots 1 to 5 of the dataviz reference palette, fixed order.
COLOURS = {"a": "#2a78d6", "b": "#eb6834", "b+": "#1baf7a", "d": "#eda100", "c": "#e87ba4"}
LABELS = {
    "a": "A: f32",
    "b": "B: DS state",
    "b+": "B+: DS scalars",
    "d": "D: DS drift",
    "c": "C: DS throughout",
}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def load(moons):
    rows = list(csv.DictReader(open(HERE / f"g1_{moons}_series.csv")))
    out = {}
    for r in rows:
        if r["orbit_type"] != "core":
            continue
        key = ("cpu" if r["source"] == "cpu_own" else r["source"], r["option"])
        out.setdefault(key, ([], []))
        out[key][0].append(float(r["orbits"]))
        out[key][1].append(max(float(r["rms_dphase_rad"]), 1e-13))
    return out


fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True, facecolor=SURFACE)
TITLES = ["Saturn only", "With Prometheus and Pandora: 16 particles regular to\n1000 orbits (shadow test); all are chaotic by 10^4 orbits"]
for ax, moons, title in zip(axes, ["none", "both"], TITLES):
    data = load(moons)
    ax.set_facecolor(SURFACE)
    for opt in ["a", "b", "b+", "d", "c"]:
        x, y = data[("gpu", opt)]
        ax.plot(x, y, color=COLOURS[opt], lw=2, label=LABELS[opt] + " (GPU)")
        if moons == "none":  # direct labels; with moons the line ends meet
            dy = {"a": 6, "b": -5}.get(opt, 0)  # A and B end close together
            ax.annotate(opt.upper(), (x[-1], y[-1]), xytext=(4, dy), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
    x, y = data[("cpu", "f64")]
    ax.plot(x, y, color=MUTED, lw=2, ls="--", label="CPU's own error (h against h/10)")
    ax.plot([1000], [1e-5], marker="o", ms=8, color=INK, ls="none",
            label="PLAN tolerance: 1e-5 rad at 1000 orbits")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 14000)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel("orbits", color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
axes[0].set_ylabel("RMS phase difference, GPU - CPU (rad)", color=INK)
axes[1].legend(loc="lower right", fontsize=8, frameon=False)
fig.suptitle("Core orbit (e = 0.0024), 32 particles, 150 steps per orbit", x=0.06, ha="left",
             fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(HERE / "g1_phase.png", dpi=130, facecolor=SURFACE)
print("wrote", HERE / "g1_phase.png")
