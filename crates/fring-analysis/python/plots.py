"""Figures for task A1-ANALYSIS (Altair / Vega-Lite).

Reads the small CSV files that `fring-analyse` writes to docs/results/data/ and
writes, for every figure, a PNG (for the note) and a Vega-Lite JSON spec with
its data (for the web explainer) to docs/results/figures/.

    cd crates/fring-analysis/python && uv sync && uv run python plots.py [name ...]
"""

import json
import pathlib
import sys

import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.disable_max_rows()  # Altair stops at 5,000 rows by default

ROOT = pathlib.Path(__file__).resolve().parents[3] / "docs" / "results"
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CORE_KM = 140222.43  # params/ring.toml
# Okabe-Ito colours: bounded, regular, uncertain, chaotic.
CLASSES = ["bounded", "regular", "uncertain", "chaotic"]
CLASS_COLOURS = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
CLASS_RANGE = {
    "bounded": "mean Y < 1.5",
    "regular": "1.5 to 2.5",
    "uncertain": "2.5 to 5",
    "chaotic": ">= 5",
}
CLASS_LABELS = [f"{c} ({CLASS_RANGE[c]})" for c in CLASSES]


def save(chart, name):
    chart.save(OUT / f"{name}.png", scale_factor=2)
    text = chart.to_json()
    (OUT / f"{name}.vl.json").write_text(text)
    print(f"{name}: {len(text) / 1e3:.0f} kB spec")


def moon_lines():
    g = pd.read_csv(DATA / "moon_geometry.csv").set_index("name")
    return pd.DataFrame(
        {
            "r": [CORE_KM, g.loc["prometheus", "r_max_km"], g.loc["pandora", "r_min_km"]],
            "label": [
                "core, 140,222 km",
                "Prometheus apoapsis",
                "Pandora periapsis",
            ],
        }
    )


def vlines_h(lines):
    """Horizontal rules (y = radius) with a label."""
    rule = (
        alt.Chart(lines)
        .mark_rule(strokeDash=[5, 3], color="black")
        .encode(y="r:Q", tooltip=["label:N", "r:Q"])
    )
    return rule


def class_of(y):
    return np.select([y < 1.5, y <= 2.5, y < 5.0], [0, 1, 2], 3)


def map_chart(tag, years, r_lo, r_hi, title_extra=""):
    m = pd.read_csv(DATA / f"{tag}_map.csv")
    r_bin = float((m["r_hi_km"] - m["r_lo_km"]).iloc[0])
    lon_bin = float((m["lon_hi_deg"] - m["lon_lo_deg"]).iloc[0])
    d = pd.DataFrame(
        {
            "r": m["r_lo_km"].round(1),
            "lon": m["lon_lo_deg"].round(1),
            "y": m["median_megno"].round(1),
            "n": m["n"],
        }
    )
    cls_expr = (
        f"datum.y < 1.5 ? '{CLASS_LABELS[0]}' : datum.y <= 2.5 ? '{CLASS_LABELS[1]}' : "
        f"datum.y < 5 ? '{CLASS_LABELS[2]}' : '{CLASS_LABELS[3]}'"
    )
    scale = alt.Scale(domain=CLASS_LABELS, range=CLASS_COLOURS)
    heat = (
        alt.Chart(d)
        .transform_calculate(cls=cls_expr, r2=f"datum.r + {r_bin}", lon2=f"datum.lon + {lon_bin}")
        .mark_rect()
        .encode(
            x=alt.X(
                "lon:Q",
                title="Initial longitude minus longitude of Prometheus (deg)",
                scale=alt.Scale(domain=[-180, 180], nice=False),
            ),
            x2="lon2:Q",
            y=alt.Y(
                "r:Q",
                title="Initial orbital radius (km)",
                scale=alt.Scale(domain=[r_lo, r_hi], nice=False),
            ),
            y2="r2:Q",
            color=alt.Color(
                "cls:N",
                scale=scale,
                title="Class of the bin median",
                legend=alt.Legend(orient="bottom", columns=2),
            ),
            tooltip=[
                alt.Tooltip("r:Q", title="radius (km)"),
                alt.Tooltip("lon:Q", title="longitude (deg)"),
                alt.Tooltip("y:Q", title="median mean MEGNO"),
                alt.Tooltip("n:Q", title="particles"),
            ],
        )
    )
    lines = moon_lines()
    lines = lines[(lines.r > r_lo) & (lines.r < r_hi)]
    rules = vlines_h(lines)
    text = (
        alt.Chart(lines)
        .mark_text(align="left", dx=4, dy=-5, fontSize=11)
        .encode(x=alt.value(4), y="r:Q", text="label:N")
    )
    return (
        (heat + rules + text)
        .properties(
            width=620,
            height=420,
            title=f"Chaos map, {years} years of integration (bins {r_bin:g} km x {lon_bin:g} deg){title_extra}",
        )
        .configure_view(strokeWidth=0)
    )


def profile_data(tag, label):
    p = pd.read_csv(DATA / f"{tag}_profile.csv")
    p["r_mid"] = 0.5 * (p["r_lo_km"] + p["r_hi_km"])
    p["run"] = label
    long = p.melt(
        id_vars=["r_lo_km", "r_hi_km", "r_mid", "run", "n"],
        value_vars=[f"frac_{c}" for c in CLASSES],
        var_name="cls",
        value_name="fraction",
    )
    long["cls"] = long["cls"].str.replace("frac_", "")
    long["fraction"] = long["fraction"].round(3)
    return long


def profile_chart(tags, r_lo, r_hi, title):
    lines = moon_lines()
    lines = lines[(lines.r > r_lo) & (lines.r < r_hi)]
    panels = []
    for t, lab in tags:
        d = profile_data(t, lab)[["r_lo_km", "cls", "fraction"]]
        d["r_lo_km"] = d["r_lo_km"].round(0).astype(int)
        bars = (
            alt.Chart(d)
            .mark_line(point=alt.OverlayMarkDef(size=14, filled=True), strokeWidth=1.2)
            .encode(
                x=alt.X("r_lo_km:Q", title="Initial orbital radius (km)", scale=alt.Scale(domain=[r_lo, r_hi], nice=False)),
                y=alt.Y("fraction:Q", title="Fraction of particles", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "cls:N",
                    scale=alt.Scale(domain=CLASSES, range=CLASS_COLOURS),
                    title="Class of the mean MEGNO: bounded < 1.5 <= regular <= 2.5 < uncertain < 5 <= chaotic",
                    legend=alt.Legend(orient="bottom", columns=4, titleLimit=0),
                ),
                tooltip=["r_lo_km", "cls", "fraction"],
            )
        )
        rules = (
            alt.Chart(lines)
            .mark_rule(strokeDash=[5, 3], color="black")
            .encode(x="r:Q", tooltip=["label:N", "r:Q"])
        )
        panels.append((bars + rules).properties(width=620, height=110, title=f"{lab} of integration"))
    return alt.vconcat(*panels).properties(title=title).resolve_scale(color="shared").configure_view(strokeWidth=0)


def fig_chaos_map():
    save(map_chart("main10", 10, 139700, 141100), "chaos_map_10yr")
    save(map_chart("main5", 5, 139700, 141100), "chaos_map_5yr")
    if (DATA / "main30_map.csv").exists():
        save(map_chart("main30", 30, 139950, 141050), "chaos_map_30yr")


def fig_chaos_profile():
    tags = [("main5", "5 years"), ("main10", "10 years")]
    if (DATA / "main30_profile.csv").exists():
        tags.append(("main30", "30 years"))
    save(
        profile_chart(
            tags,
            139700,
            141100,
            "Class fractions against initial radius (2 x 10^5 circular particles, 2 km bins; 30 years: 1.1 x 10^5 particles)",
        ),
        "chaos_profile",
    )


def fig_chaos_zoom():
    if not (DATA / "zoom10_map.csv").exists():
        return
    save(map_chart("zoom10", 10, 140000, 140450), "chaos_zoom_10yr")
    save(
        profile_chart(
            [("zoom10", "10 years")],
            140000,
            140450,
            "Zoom around the core: class fractions against initial radius (5 x 10^4 particles, 2 km bins)",
        ),
        "chaos_zoom_profile",
    )


def fig_megno_hist():
    frames = []
    for tag, lab in [("main5", "5 years"), ("main10", "10 years"), ("main30", "30 years")]:
        f = DATA / f"{tag}_hist.csv"
        if f.exists():
            h = pd.read_csv(f)
            h["run"] = lab
            frames.append(h)
    h = pd.concat(frames)
    h["count"] = h["count"].astype(int)
    rules = pd.DataFrame({"x": [1.5, 2.5, 5.0]})
    bars = (
        alt.Chart(h)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("megno_lo:Q", title="Mean MEGNO (last bin collects all larger values)", scale=alt.Scale(domain=[-2, 20], nice=False)),
            x2="megno_hi:Q",
            y=alt.Y("count:Q", title="Particles", scale=alt.Scale(type="symlog")),
            color=alt.Color("run:N", title="Run length", scale=alt.Scale(range=["#56B4E9", "#E69F00", "#CC79A7"])),
            tooltip=["run", "megno_lo", "count"],
        )
        .properties(width=620, height=260)
    )
    r = alt.Chart(rules).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
    save(
        (bars + r).properties(title="Distribution of the mean MEGNO (dashed: class limits 1.5, 2.5 and 5)").configure_view(strokeWidth=0),
        "megno_hist",
    )


def fig_closest():
    f = DATA / "closest_hist.csv"
    if not f.exists():
        return
    h = pd.read_csv(f)
    bars = (
        alt.Chart(h)
        .mark_bar(color="#0072B2")
        .encode(
            x=alt.X("hill_lo:Q", title="Closest approach to Prometheus (Hill radii; 1 R_H = 63.4 km)", scale=alt.Scale(type="log", domain=[0.5, 100])),
            x2="hill_hi:Q",
            y=alt.Y("count:Q", title="Particles"),
            tooltip=["hill_lo", "hill_hi", "count"],
        )
    )
    rules = alt.Chart(pd.DataFrame({"x": [1.0, 3.0, 7.3]})).mark_rule(strokeDash=[4, 3], color="#D55E00").encode(x="x:Q")
    save(
        (bars + rules)
        .properties(width=620, height=240, title="Closest approach to Prometheus in 10 years, 20,000 particles (dashed: 1, 3 and 7.3 Hill radii)")
        .configure_view(strokeWidth=0),
        "closest_approach",
    )
    r = pd.read_csv(DATA / "closest_rbins.csv")
    r["r"] = (0.5 * (r.r_lo_km + r.r_hi_km)).round(0)
    long = r.melt(id_vars="r", value_vars=["frac_below_1_hill", "frac_below_3_hill", "frac_below_7p3_hill"], var_name="limit", value_name="fraction")
    long["limit"] = long["limit"].map({"frac_below_1_hill": "closer than 1 R_H", "frac_below_3_hill": "closer than 3 R_H", "frac_below_7p3_hill": "closer than 7.3 R_H (459 km)"})
    long["fraction"] = long["fraction"].round(4)
    c = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("r:Q", title="Initial orbital radius (km)", scale=alt.Scale(domain=[139700, 141100], nice=False)),
            y=alt.Y("fraction:Q", title="Fraction of particles"),
            color=alt.Color("limit:N", title=None, legend=alt.Legend(orient="bottom", columns=3)),
            tooltip=["r", "limit", "fraction"],
        )
        .properties(width=620, height=240, title="Fraction of particles that come close to Prometheus in 10 years (50 km bins)")
        .configure_view(strokeWidth=0)
    )
    save(c, "closest_by_radius")


def fig_apse():
    f = DATA / "apse_fits.csv"
    if not f.exists():
        return
    fits = pd.read_csv(f)
    names = {
        "apse-none": "Saturn only (J2 to J6)",
        "apse-prom": "+ Prometheus",
        "apse-pand": "+ Pandora",
        "apse-both": "+ both moons",
        "apse-both-phase": "+ both, other apse phase and seed",
        "apse-both-half": "+ both, half time step",
    }
    fits["label"] = fits["run"].map(names)
    lap = pd.read_csv(DATA / "laplace_secular.csv").set_index("name")["rate_deg_per_day"]
    fits = pd.concat(
        [
            fits,
            pd.DataFrame(
                {
                    "run": ["analytic"],
                    "rate_deg_per_day": [lap["saturn_only"] + lap["sum_of_moons"]],
                    "se_jackknife": [0.0],
                }
            ),
        ],
        ignore_index=True,
    )
    names["analytic"] = "Saturn + secular formula (Laplace)"
    fits["label"] = fits["run"].map(names)
    fits["lo"] = fits["rate_deg_per_day"] - fits["se_jackknife"]
    fits["hi"] = fits["rate_deg_per_day"] + fits["se_jackknife"]
    obs = pd.DataFrame({"lo": [2.70025 - 0.00029], "hi": [2.70025 + 0.00029], "v": [2.70025]})
    xs = alt.Scale(domain=[2.6975, 2.7035], nice=False)
    order = list(names.values())
    pts = (
        alt.Chart(fits)
        .mark_point(filled=True, size=70, color="#0072B2")
        .encode(
            y=alt.Y("label:N", sort=order, title=None),
            x=alt.X("rate_deg_per_day:Q", title="Apse precession rate at the core (deg/day)", scale=xs, axis=alt.Axis(format=".4f", tickCount=6, labelOverlap=True)),
            tooltip=["label", "rate_deg_per_day", "se_jackknife"],
        )
    )
    err = (
        alt.Chart(fits)
        .mark_rule(color="#0072B2")
        .encode(y=alt.Y("label:N", sort=order), x=alt.X("lo:Q", scale=xs), x2="hi:Q")
    )
    band = alt.Chart(obs).mark_rect(opacity=0.25, color="#D55E00").encode(x=alt.X("lo:Q", scale=xs), x2="hi:Q")
    line = alt.Chart(obs).mark_rule(color="#D55E00").encode(x=alt.X("v:Q", scale=xs))
    chart = (band + line + err + pts).properties(
        width=560,
        height=230,
        title="Apse precession of core particles, 50-year runs (orange band: observed 2.70025 +- 0.00029)",
    )
    save(chart.configure_view(strokeWidth=0), "apse_rate")

    # Residual of the apse longitude about the straight line, both moons.
    s = DATA / "apse_series_apse-both.csv"
    if s.exists():
        d = pd.read_csv(s)
        d["residual_deg"] = d["residual_deg"].round(4)
        d = d[["t_years", "residual_deg"]]
        c = (
            alt.Chart(d)
            .mark_line(color="#0072B2")
            .encode(
                x=alt.X("t_years:Q", title="Time (years)"),
                y=alt.Y("residual_deg:Q", title="Apse longitude minus linear fit (deg)"),
            )
            .properties(width=620, height=220, title="Apse longitude of the core pattern, both moons: wobble about the mean rate")
            .configure_view(strokeWidth=0)
        )
        save(c, "apse_wobble")


def fig_lyapunov():
    f = DATA / "lyapunov_curve.csv"
    if not f.exists():
        return
    c = pd.read_csv(f)
    c = c[c.t_years >= 1.0].round(4)
    long = []
    for col, sd, lab in [
        ("ftle_mean", "ftle_std", "plain L(T) / T"),
        ("slope_mean", "slope_std", "slope of L from year 30 to T"),
        ("megno_rate_mean", "megno_rate_std", "2 d<Y>/dt from year 30 to T"),
    ]:
        d = c[["t_years", col, sd]].rename(columns={col: "mean", sd: "std"})
        d["method"] = lab
        d["lo"] = d["mean"] - d["std"]
        d["hi"] = d["mean"] + d["std"]
        long.append(d.dropna())
    d = pd.concat(long)
    band = (
        alt.Chart(d)
        .mark_area(opacity=0.18)
        .encode(x=alt.X("t_years:Q", scale=alt.Scale(domain=[1, 100], type="log")), y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 1.2], clamp=True)), y2="hi:Q", color=alt.Color("method:N", title=None, legend=alt.Legend(orient="bottom", columns=1)))
    )
    line = (
        alt.Chart(d)
        .mark_line()
        .encode(
            x=alt.X("t_years:Q", title="Time T (years)", scale=alt.Scale(domain=[1, 100], type="log")),
            y=alt.Y("mean:Q", title="Lyapunov exponent (1/year)", scale=alt.Scale(domain=[0, 1.2])),
            color="method:N",
            tooltip=["method", "t_years", "mean", "std"],
        )
    )
    lit = pd.DataFrame({"lo": [0.30], "hi": [0.35]})
    lit_band = alt.Chart(lit).mark_rect(opacity=0.18, color="gray").encode(y="lo:Q", y2="hi:Q")
    save(
        (lit_band + band + line)
        .properties(width=620, height=300, title="Lyapunov exponent of Prometheus and Pandora (mean and spread of 100 independent runs; grey: literature 0.30 to 0.35)")
        .configure_view(strokeWidth=0),
        "moon_lyapunov",
    )


def fig_lyapunov_pairs():
    f = DATA / "lyapunov_pairs.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)[["pair", "slope_30_end_per_yr"]].rename(columns={"slope_30_end_per_yr": "slope"})
    d["slope"] = d["slope"].round(4)
    hist = (
        alt.Chart(d)
        .mark_bar(color="#0072B2")
        .encode(
            x=alt.X("slope:Q", bin=alt.Bin(step=0.02), title="Growth rate of ln(separation), fitted over years 30 to 100 (1/year)"),
            y=alt.Y("count():Q", title="Independent runs"),
            tooltip=["count()"],
        )
    )
    rules = (
        alt.Chart(pd.DataFrame({"x": [0.1, 0.30, 0.35], "label": ["split used in the note", "literature low", "literature high"]}))
        .mark_rule(strokeDash=[4, 3], color="#D55E00")
        .encode(x="x:Q", tooltip=["label:N"])
    )
    save(
        (hist + rules)
        .properties(width=620, height=240, title="Lyapunov exponent of Prometheus and Pandora: 100 independent runs (two populations)")
        .configure_view(strokeWidth=0),
        "moon_lyapunov_pairs",
    )


def window_summary():
    """Class fractions (weighted by particle count) in a few radius windows, for the note.
    Also lambda = 2 <Y> / t from the median of the mean MEGNO (valid for chaotic orbits)."""
    wins = [
        ("inner edge, 139,700 to 139,950 km", 139700, 139950),
        ("inner flank, 139,950 to 140,150 km", 139950, 140150),
        ("core window, 140,197 to 140,247 km", 140197, 140247),
        ("quiet zone, 140,150 to 140,700 km", 140150, 140700),
        ("outer flank, 140,700 to 140,900 km", 140700, 140900),
        ("outer edge, 140,900 to 141,100 km", 140900, 141100),
    ]
    rows = []
    for tag, yrs in [("main5", 5), ("main10", 10), ("main30", 30)]:
        p = pd.read_csv(DATA / f"{tag}_profile.csv")
        for name, lo, hi in wins:
            q = p[(p.r_lo_km >= lo) & (p.r_hi_km <= hi)]
            if q.empty or q.n.sum() < 500:
                continue
            w = q.n / q.n.sum()
            row = {"run_years": yrs, "window": name, "n": int(q.n.sum())}
            for c in CLASSES:
                row[c] = round(float((q[f"frac_{c}"] * w).sum()), 3)
            row["median_megno"] = round(float(q.median_megno.median()), 1)
            row["lambda_per_yr"] = round(2 * row["median_megno"] / yrs, 2)
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(DATA / "window_summary.csv", index=False)
    print(t.to_string(index=False))


FIGS = {
    "map": fig_chaos_map,
    "profile": fig_chaos_profile,
    "zoom": fig_chaos_zoom,
    "hist": fig_megno_hist,
    "closest": fig_closest,
    "apse": fig_apse,
    "lyapunov": fig_lyapunov,
    "lyapunov_pairs": fig_lyapunov_pairs,
    "windows": window_summary,
}



def comb_period(tag="zoom10", r_lo=140000.0, r_hi=140450.0):
    """Dominant period (km) of the 'bounded' fraction against radius (FFT), after
    removing the mean and a linear trend. Compared in the note with the spacing of
    first-order resonances with Prometheus, (2/3) a / m^2 with m = 110."""
    p = pd.read_csv(DATA / f"{tag}_profile.csv")
    p = p[(p.r_lo_km >= r_lo) & (p.r_hi_km <= r_hi)]
    x = p["frac_bounded"].to_numpy()
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    bin_km = float(p["r_hi_km"].iloc[0] - p["r_lo_km"].iloc[0])
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), d=bin_km)
    ok = np.where(freq >= 1.0 / 60.0)[0]  # periods below 60 km
    k = int(ok[np.argmax(spec[ok])])
    period = 1.0 / freq[k]
    spacing = (2.0 / 3.0) * 140222.43 / 110.0**2
    print(f"{tag}: dominant period of the bounded fraction {period:.1f} km (bin {bin_km:g} km); "
          f"first-order resonance spacing near m=110: {spacing:.1f} km")


FIGS["comb"] = comb_period


if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for k in which:
        FIGS[k]()
