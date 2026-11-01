"""V2: Prometheus and Pandora over 20 Julian years with REBOUND (serves C3).

Usage (from validation/):
  uv run python scripts/v2_moons.py            # full run (about 15 min, 5 runs in parallel; reuses finished runs in reference/large): data, analysis, plots
  uv run python scripts/v2_moons.py --analyse  # redo the analysis and plots from saved runs
  uv run python scripts/v2_moons.py --check    # recompute the first day, compare with the committed file
Model: ADR 0004 (Saturn a free massive body; J2, J4, J6 force relative to Saturn; Saturn's reaction to
the J force neglected, relative order 3e-10).  Outputs are Saturn-centred, Saturn-equator frame, km, s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import rebound

from fring_ref.moons import COLUMNS, YEAR_S, run
from fring_ref import resonance as rs
from fring_ref.params import load_moons, load_saturn

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"
LARGE = REF / "large"
REPORTS = ROOT / "reports"
DURATION = 20 * YEAR_S
CADENCE = 3600.0
EPS = 1e-9
def _unit_velocity():
    v = np.array(load_moons()[0].vel)
    return v / np.linalg.norm(v)


# name: Prometheus offset in km.  "along": exactly along its velocity (a pure time shift on the orbit,
# no change of semi-major axis, so no Kepler shear).  "x" and "y" are the plain axes (y is nearly radial at
# the epoch, so it changes a and gives a linear Kepler shear that hides the chaotic growth).
SHADOWS = {"along1e-3": tuple(1e-3 * _unit_velocity()), "along1e-6": tuple(1e-6 * _unit_velocity()),
           "tang1e-3": (1e-3, 0, 0), "rad1e-3": (0, 1e-3, 0)}


def _job(args):
    name, off, duration = args
    t0 = time.time()
    arr = run(load_saturn(), duration, CADENCE, offset_km=off, epsilon=EPS)
    return name, arr, time.time() - t0


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def full_run():
    LARGE.mkdir(parents=True, exist_ok=True)
    jobs = [("main", (0.0, 0.0, 0.0), DURATION)] + [(k, v, DURATION) for k, v in SHADOWS.items()]
    jobs = [j for j in jobs if not (LARGE / f"v2_{j[0]}_1h.npy").exists()]  # reuse finished runs
    t0 = time.time()
    res = []
    if jobs:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            res = pool.map(_job, jobs)
    wall = time.time() - t0
    walls = {}
    for name, arr, w in res:
        np.save(LARGE / f"v2_{name}_1h.npy", arr)
        walls[name] = w
    main = np.load(LARGE / "v2_main_1h.npy")
    # committed products
    np.save(REF / "v2_moons_6h.npy", main[::6])
    daily = main[::24]
    np.savetxt(REF / "v2_moons_1d.csv", daily, delimiter=",", fmt="%.17g",
               header="V2 REBOUND reference, Saturn-centred, Saturn-equator frame, km and km/s; columns: " + ",".join(COLUMNS),
               comments="# ")
    np.save(REF / "v2_shadow_along1e-3_6h.npy", np.load(LARGE / "v2_along1e-3_1h.npy")[::6])
    return walls, wall


def analyse(walls=None, wall=None):
    sat = load_saturn()
    main = np.load(LARGE / "v2_main_1h.npy")
    t = main[:, 0]
    tout = t[(t > 2 * 86400) & (t < t[-1] - 2 * 86400)][::6]  # 6 h grid inside the run
    el = rs.mean_elements(sat, main, tout)
    pr, pd = el
    half = 24.84 * rs.DAY  # one synodic period either side, so two synodic periods in all
    n_pr = rs.mean_motion(tout, pr["lam"], half)
    n_pd = rs.mean_motion(tout, pd["lam"], half)
    to_deg_yr = 180 / np.pi * rs.YEAR
    phis = rs.resonant_angles(pr["lam"], pd["lam"], pr["varpi"], pd["varpi"])
    _, st = rs.split(main)
    tc, sep = rs.conjunctions(t, st[0][0], st[1][0])
    from scipy.signal import find_peaks
    from scipy.ndimage import median_filter
    from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d
    # the conjunction longitude walks around the orbit, so separations alternate between the lower and upper
    # envelopes; take the rolling minimum over 15 conjunctions (about 1 yr), then smooth over 15 more
    sep_s = uniform_filter1d(minimum_filter1d(sep, size=15, mode="nearest"), size=15, mode="nearest")
    sep_up = uniform_filter1d(maximum_filter1d(sep, size=15, mode="nearest"), size=15, mode="nearest")
    mins, _ = find_peaks(-sep_s, distance=30, prominence=30.0)
    dvarpi = np.unwrap(pr["varpi"] - pd["varpi"])
    rate = np.polyfit(tout, dvarpi, 1)[0]
    anti_period_yr = 2 * np.pi / abs(rate) / rs.YEAR
    # anti-alignment epochs: dvarpi = pi mod 2 pi
    k = np.arange(np.floor((dvarpi.min() - np.pi) / (2 * np.pi)), np.ceil((dvarpi.max() - np.pi) / (2 * np.pi)) + 1)
    anti_epochs = []
    for kk in k:
        lvl = np.pi + 2 * np.pi * kk
        idx = np.nonzero(np.sign(dvarpi[:-1] - lvl) != np.sign(dvarpi[1:] - lvl))[0]
        if len(idx):
            i = idx[0]
            anti_epochs.append(tout[i] / rs.YEAR)
    summary = dict(
        n_prometheus_deg_yr=dict(mean=float(np.nanmean(n_pr) * to_deg_yr), min=float(np.nanmin(n_pr) * to_deg_yr),
                                 max=float(np.nanmax(n_pr) * to_deg_yr), range=float((np.nanmax(n_pr) - np.nanmin(n_pr)) * to_deg_yr),
                                 std=float(np.nanstd(n_pr) * to_deg_yr)),
        n_pandora_deg_yr=dict(mean=float(np.nanmean(n_pd) * to_deg_yr), min=float(np.nanmin(n_pd) * to_deg_yr),
                              max=float(np.nanmax(n_pd) * to_deg_yr), range=float((np.nanmax(n_pd) - np.nanmin(n_pd)) * to_deg_yr),
                              std=float(np.nanstd(n_pd) * to_deg_yr)),
        n_prometheus_deg_day_mean=float(np.nanmean(n_pr) * 180 / np.pi * rs.DAY),
        n_pandora_deg_day_mean=float(np.nanmean(n_pd) * 180 / np.pi * rs.DAY),
        mean_motion_note="orbit-averaged longitude, central difference over +-24.84 d (two synodic periods); range = max - min of the 20 yr series",
        apse_antialignment_period_yr_from_linear_fit=float(anti_period_yr),
        apse_antialignment_epochs_yr=[float(x) for x in anti_epochs],
        conjunction_count=int(len(tc)),
        conjunction_min_separation_epochs_yr=[float(tc[i] / rs.YEAR) for i in mins],
        conjunction_min_separation_spacing_yr=[float(x) for x in np.diff(tc[mins] / rs.YEAR)],
        conjunction_separation_km=dict(min=float(sep.min()), max=float(sep.max())),
        synodic_period_days_mean=float(np.mean(np.diff(tc)) / rs.DAY),
        resonant_angles_note="phi_k = 121 lam_Pd - 118 lam_Pr - k varpi_Pr - (3-k) varpi_Pd, orbit-averaged (mean) elements, not geometric",
        ecc_mean=dict(prometheus=float(pr["ecc"].mean()), pandora=float(pd["ecc"].mean())),
    )
    # chaos
    chaos = {}
    shadows = {}
    for name in SHADOWS:
        f = LARGE / f"v2_{name}_1h.npy"
        if not f.exists():
            continue
        sh = np.load(f)
        shadows[name] = sh
        dpr, dpd = rs.divergence(main, sh)
        dmax = np.maximum(dpr, dpd)
        dd = np.sqrt(dpr**2 + dpd**2)
        lyap = {f"{a}-{b} yr": rs.growth_rate(t, dd, a, b) for a, b in ((0.2, 2), (2, 6), (6, 12), (12, 20), (0.2, 20))}
        chaos[name] = dict(
            offset_km=list(SHADOWS[name]),
            T_1km_yr=float(rs.first_crossing(t, dmax, 1.0) / rs.YEAR),
            T_1km_prometheus_yr=float(rs.first_crossing(t, dpr, 1.0) / rs.YEAR),
            T_1km_pandora_yr=float(rs.first_crossing(t, dpd, 1.0) / rs.YEAR),
            T_1m_yr=float(rs.first_crossing(t, dmax, 1e-3 * 10) / rs.YEAR),
            final_dr_km=dict(prometheus=float(dpr[-1]), pandora=float(dpd[-1])),
            growth_rate_per_yr_of_ln_dr=lyap,
            d_over_offset_at_yr={str(y): float(dd[int(round(y * rs.YEAR / 3600))] / np.linalg.norm(SHADOWS[name])) for y in (1, 3, 6, 12, 20)},
        )
    summary["chaos"] = chaos
    meta = dict(
        task="V2", frame="Saturn-centred (Saturn position and velocity subtracted), Saturn equator frame of params/moons.toml",
        units="km, km/s, s", epoch="2008-01-01 00:00 TDB = t_s 0", duration_years_julian=20,
        model="ADR 0004: Saturn a free massive body; J2,J4,J6 force on the moons relative to Saturn; Saturn's reaction to the J force neglected (relative order 3e-10)",
        integrator=dict(name="ias15", epsilon=EPS, cadence_s_primary=CADENCE),
        versions=dict(rebound=rebound.__version__, numpy=np.__version__, python=sys.version.split()[0]),
        columns=COLUMNS,
        files={"v2_moons_6h.npy": "float64 (29221, 13); rows every 21600 s from t = 0",
               "v2_moons_1d.csv": "preview, rows every 86400 s",
               "v2_shadow_along1e-3_6h.npy": "same layout; Prometheus displaced by 1e-3 km along its velocity at t = 0 (shadow run)"},
        wall_s=walls, wall_total_s=wall,
        sha256={"v2_moons_6h.npy": sha256(REF / "v2_moons_6h.npy"), "v2_moons_1d.csv": sha256(REF / "v2_moons_1d.csv"),
                "v2_shadow_along1e-3_6h.npy": sha256(REF / "v2_shadow_along1e-3_6h.npy")},
        large_files_git_ignored={f.name: dict(sha256=sha256(f), bytes=f.stat().st_size) for f in sorted(LARGE.glob("*.npy"))},
        analysis=summary,
    )
    if (REF / "v2_moons.json").exists():  # keep run times of earlier runs of the other jobs
        old = json.loads((REF / "v2_moons.json").read_text())
        merged = dict(old.get("wall_s") or {})
        merged.update(walls or {})
        meta["wall_s"] = merged
        meta["wall_total_s"] = (wall if walls else old.get("wall_total_s"))
    (REF / "v2_moons.json").write_text(json.dumps(meta, indent=2) + "\n")
    plots(sat, main, shadows, tout, el, n_pr, n_pd, phis, tc, sep, sep_s, mins, dvarpi, anti_epochs, summary, sep_up)
    print(json.dumps(summary, indent=2))


def plots(sat, main, shadows, tout, el, n_pr, n_pd, phis, tc, sep, sep_s, mins, dvarpi, anti_epochs, summary, sep_up=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ty = tout / rs.YEAR
    c = ["#1f77b4", "#d95f02"]
    deg_yr = 180 / np.pi * rs.YEAR
    # 1: mean motions and angles
    fig, ax = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    a = ax[0, 0]
    a.plot(ty, (n_pr - np.nanmean(n_pr)) * deg_yr, c=c[0], label="Prometheus")
    a.plot(ty, (n_pd - np.nanmean(n_pd)) * deg_yr, c=c[1], label="Pandora")
    a.set_ylabel("n - mean (deg/yr)"); a.set_title("Mean motion (orbit-averaged, +-24.84 d difference)"); a.legend()
    a = ax[0, 1]
    a.plot(ty, np.degrees(np.unwrap(pr_lon := (el[1]["lam"] - el[0]["lam"]))) % 360 - 180, ".", ms=1, c="k")
    a.set_ylabel("lam_Pd - lam_Pr mod 360 (deg, centred)"); a.set_title("Longitude difference (sawtooth: one conjunction per cycle)")
    for k in range(4):
        a = ax[1 + k // 2, k % 2]
        a.plot(ty, np.degrees(phis[k]), ".", ms=1, c="k")
        a.set_title(f"phi_{k} = 121 lam_Pd - 118 lam_Pr - {k} varpi_Pr - {3 - k} varpi_Pd")
        a.set_ylabel("deg"); a.set_ylim(-180, 180)
    for a in ax[:, :].ravel():
        a.set_xlabel("time since 2008-01-01 (Julian years)")
    fig.suptitle("V2: Prometheus and Pandora, REBOUND IAS15, 20 yr (mean elements = orbit-averaged, not geometric)")
    fig.savefig(REPORTS / "v2_resonance.png", dpi=120)
    # 2: conjunction separation and apse alignment
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    ax[0].plot(tc / rs.YEAR, sep, ".", ms=3, c="gray", label="at each conjunction")
    ax[0].plot(tc / rs.YEAR, sep_s, c="k", label="lower envelope (rolling min of 15 conjunctions, smoothed)")
    if sep_up is None:
        sep_up = sep
    ax[0].plot(tc / rs.YEAR, sep_up, c="b", lw=1, label="upper envelope")
    ax[0].plot(tc[mins] / rs.YEAR, sep_s[mins], "rv", label="minimum-separation epochs")
    for e in anti_epochs:
        ax[0].axvline(e, c="g", ls=":", lw=1)
    ax[0].set_ylabel("r_Pd - r_Pr at conjunction (km)"); ax[0].legend(); ax[0].set_xlabel("time (yr)")
    ax[0].set_title(f"Closest approaches; green dotted: apses anti-aligned; fit period {summary['apse_antialignment_period_yr_from_linear_fit']:.2f} yr")
    ax[1].plot(ty, np.degrees(dvarpi), c="k")
    ax[1].axhline(180, c="g", ls=":"); ax[1].set_ylabel("varpi_Pr - varpi_Pd (deg, unwrapped)"); ax[1].set_xlabel("time (yr)")
    fig.savefig(REPORTS / "v2_closest_approach.png", dpi=120)
    # 3: chaos
    if shadows:
        t = main[:, 0] / rs.YEAR
        fig, ax = plt.subplots(1, 1, figsize=(10, 5.5), constrained_layout=True)
        for name, sh in shadows.items():
            dpr, dpd = rs.divergence(main, sh)
            d = np.maximum(np.sqrt(dpr**2 + dpd**2), 1e-12)
            ax.semilogy(t, d, lw=0.7, label=f"offset {name}")
        ax.axhline(1.0, c="r", ls="--", label="1 km")
        ax.set_xlabel("time (yr)"); ax.set_ylabel("|dr| of both moons, REBOUND vs shifted copy (km)")
        ch = summary["chaos"]
        ax.set_title("Divergence of two REBOUND runs; " + ", ".join(f"T1km({k})={v['T_1km_yr']:.2f} yr" for k, v in ch.items()), fontsize=9)
        ax.legend()
        fig.savefig(REPORTS / "v2_chaos.png", dpi=120)


def check():
    """Recompute the first day (hourly) and compare with the committed 6 h file."""
    ref = np.load(REF / "v2_moons_6h.npy")
    new = run(load_saturn(), 86400.0, CADENCE, epsilon=EPS)[::6]
    m = len(new)
    dpos = np.abs(new[:, [1, 2, 3, 7, 8, 9]] - ref[:m, [1, 2, 3, 7, 8, 9]]).max()
    dvel = np.abs(new[:, [4, 5, 6, 10, 11, 12]] - ref[:m, [4, 5, 6, 10, 11, 12]]).max()
    print(f"first day: max |dr| = {dpos:.3e} km, max |dv| = {dvel:.3e} km/s")
    return dpos, dvel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        dp, dv = check()
        sys.exit(0 if dp < 1e-6 and dv < 1e-12 else 1)
    if a.analyse:
        analyse()
    else:
        walls, wall = full_run()
        analyse(walls, wall)
