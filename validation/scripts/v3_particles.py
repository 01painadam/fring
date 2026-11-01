"""V3: 100 test particles near the F ring core with Prometheus and Pandora, REBOUND reference (serves C4).

Usage (from validation/):
  uv run python scripts/v3_particles.py            # full run (about 2 minutes): data, analysis, plots, report
  uv run python scripts/v3_particles.py --analyse  # redo plots and report from saved files
  uv run python scripts/v3_particles.py --check    # recompute the first orbit of 3 particles, compare with the committed file
Model: ADR 0004 as in V2 (Saturn a free massive body, J2+J4+J6 force relative to Saturn), IAS15 epsilon 1e-9,
massless particles.  Units: km, km/s, s; Saturn-centred, Saturn-equator frame, t = 0 is the epoch of params/moons.toml.
One "orbit" is T0 = 2 pi / n(a0) with n from the J2+J4+J6 frequencies at a0 = 140222.43 km.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

import numpy as np
import rebound

from fring_ref.moons import COLUMNS as MOON_COLUMNS
from fring_ref.particles import A0_KM, epicycle_states, hermite_min_distance, hill_radii_km, orbit_period, simulate
from fring_ref.params import load_saturn

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"
LARGE = REF / "large"
REPORTS = ROOT / "reports"

SEED = 20261101          # numpy PCG64
BATCH = 150              # candidates per round
N_KEEP = 100             # particles in the reference files
BAND_KM = 150.0          # guiding radius a ~ U[a0 - 150, a0 + 150]
E_MAX = 1e-3             # epicyclic eccentricity ~ U[0, E_MAX]
I_MAX = 2e-4             # epicyclic inclination, rad, ~ U[0, I_MAX]
N_ORBITS = 100
FINE = 100               # internal samples per orbit (screening); saved files use every 10th
SAVE_EVERY = 10          # saved cadence: T0 / 10
SHADOW_KM = 1e-3         # shadow offset along the velocity
RHILL_MIN = 5.0          # reject if closer than this many Hill radii to either moon
GROWTH_MAX = 10.0        # reject if max shadow separation / SHADOW_KM exceeds this
EPS = 1e-9
EPS_FLOOR = 1e-11
N_RADIAL = 10            # particles in the radial-shadow (Kepler shear) check
PCOLS = ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def times_fine(T0):
    return np.arange(N_ORBITS * FINE + 1) * (T0 / FINE)


def draw(rng, n):
    """Draw n candidates in a fixed order (a, e, i, M, phase_z, lambda), each an array of n."""
    a = A0_KM + rng.uniform(-BAND_KM, BAND_KM, n)
    e = rng.uniform(0.0, E_MAX, n)
    inc = rng.uniform(0.0, I_MAX, n)
    m, pz, lam = (rng.uniform(0.0, 2 * np.pi, n) for _ in range(3))
    return np.stack([a, e, inc, m, pz, lam], axis=1)


def screen(sat, elems, T0):
    """Run candidates and shadows once; return ICs, subsampled trajectories and the screening numbers."""
    states = epicycle_states(sat, *elems.T)
    vhat = states[:, 3:6] / np.linalg.norm(states[:, 3:6], axis=1, keepdims=True)
    shadow = states.copy()
    shadow[:, 0:3] += SHADOW_KM * vhat
    n = len(states)
    t = times_fine(T0)
    moons, parts = simulate(sat, np.vstack([states, shadow]), t, EPS)
    main, shad = parts[:, :n], parts[:, n:]
    sep = np.linalg.norm(shad[..., 0:3] - main[..., 0:3], axis=-1)       # (nt, n)
    growth = sep.max(axis=0) / SHADOW_KM
    rh = hill_radii_km(sat)
    dmin = np.empty((n, 2))
    for k in range(2):
        rel = main[..., 0:3] - moons[:, None, k, 0:3]
        vel = main[..., 3:6] - moons[:, None, k, 3:6]
        dmin[:, k] = hermite_min_distance(rel, vel, T0 / FINE) / rh[k]
    return dict(states=states, traj=main[::SAVE_EVERY], moons=moons[::SAVE_EVERY], growth=growth, dmin_rh=dmin,
                sep=sep[::SAVE_EVERY])


def _eps_job(args):
    sat, states, T0 = args
    t0 = time.time()
    _, parts = simulate(sat, states, times_fine(T0), EPS_FLOOR)
    return parts[::SAVE_EVERY], time.time() - t0


def _radial_job(args):
    sat, states, T0 = args
    r = states[:, 0:3].copy()
    rhat = r / np.linalg.norm(r, axis=1, keepdims=True)
    sh = states.copy()
    sh[:, 0:3] += SHADOW_KM * rhat
    n = len(states)
    _, parts = simulate(sat, np.vstack([states, sh]), times_fine(T0)[::SAVE_EVERY], EPS)
    return np.linalg.norm(parts[:, n:, 0:3] - parts[:, :n, 0:3], axis=-1)


def full_run():
    sat = load_saturn()
    T0 = orbit_period(sat)
    LARGE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(np.random.PCG64(SEED))
    wall0 = time.time()
    batches, n_acc = [], 0
    while n_acc < N_KEEP:
        elems = draw(rng, BATCH)
        res = screen(sat, elems, T0)
        res["elems"] = elems
        res["bad_close"] = (res["dmin_rh"] < RHILL_MIN).any(axis=1)
        res["bad_growth"] = res["growth"] > GROWTH_MAX
        res["ok"] = ~(res["bad_close"] | res["bad_growth"])
        batches.append(res)
        n_acc += int(res["ok"].sum())
    t_screen = time.time() - wall0
    cat = lambda k: np.concatenate([b[k] for b in batches])
    elems, states, growth, dmin = cat("elems"), cat("states"), cat("growth"), cat("dmin_rh")
    bad_close, bad_growth, ok = cat("bad_close"), cat("bad_growth"), cat("ok")
    traj_all = np.concatenate([b["traj"] for b in batches], axis=1)
    sep_all = np.concatenate([b["sep"] for b in batches], axis=1)
    moons = batches[0]["moons"]
    accepted = np.flatnonzero(ok)[:N_KEEP]
    ids = -np.ones(len(ok), dtype=int)
    ids[accepted] = np.arange(N_KEEP)
    drawn_used = int(accepted[-1]) + 1   # candidates examined up to the last one kept
    traj = traj_all[:, accepted]
    ic = states[accepted]

    # follow-up runs in parallel: error floor (epsilon 1e-11) and the radial-shadow (Kepler shear) check
    t1 = time.time()
    with mp.get_context("spawn").Pool(2) as pool:
        r_eps = pool.apply_async(_eps_job, ((sat, ic, T0),))
        r_rad = pool.apply_async(_radial_job, ((sat, ic[:N_RADIAL], T0),))
        traj_eps, t_eps = r_eps.get()
        radial_sep = r_rad.get()
    t_follow = time.time() - t1
    wall = time.time() - wall0

    # ---- committed files
    nt = traj.shape[0]
    tk = np.arange(nt) * (T0 / SAVE_EVERY)
    hdr = f"V3 initial states at t = 0, Saturn-centred, Saturn-equator frame, km and km/s; columns: id," + ",".join(PCOLS)
    with open(REF / "v3_ic.csv", "w") as f:
        f.write("# " + hdr + "\n")
        for i, s in enumerate(ic):
            f.write(f"{i}," + ",".join(f"{v:.17g}" for v in s) + "\n")
    names = ["draw_index", "id", "status", "a_km", "e_epicyclic", "i_epicyclic_rad", "M0_rad", "phase_z0_rad", "lambda0_rad",
             "min_dist_prometheus_RH", "min_dist_pandora_RH", "growth_factor"] + PCOLS
    with open(REF / "v3_particles_meta.csv", "w", newline="") as f:
        f.write("# V3 all drawn candidates; status: accepted, not_needed (passed but beyond the first 100), "
                "close_approach, shadow_growth, close_approach+shadow_growth; states are the initial Cartesian states\n")
        w = csv.writer(f)
        w.writerow(names)
        for j in range(drawn_used):
            if ok[j]:
                status = "accepted" if ids[j] >= 0 else "not_needed"
            else:
                status = "+".join(s for s, b in (("close_approach", bad_close[j]), ("shadow_growth", bad_growth[j])) if b)
            w.writerow([j, ids[j], status] + [f"{v:.17g}" for v in elems[j]] + [f"{dmin[j, 0]:.6g}", f"{dmin[j, 1]:.6g}",
                                                                                f"{growth[j]:.6g}"] + [f"{v:.17g}" for v in states[j]])
    np.save(REF / "v3_traj.npy", traj)
    mcols = ["k", "t_s"] + MOON_COLUMNS[1:]
    with open(REF / "v3_moons.csv", "w") as f:
        f.write("# V3 moons (same run as the particles), Saturn-centred, km and km/s; k = index on the T0/10 grid; columns: " + ",".join(mcols) + "\n")
        for k in range(nt):
            f.write(f"{k},{tk[k]:.17g}," + ",".join(f"{v:.17g}" for v in moons[k].ravel()) + "\n")
    longhdr = "k,t_s,id," + ",".join(PCOLS)
    def write_long(path, ks):
        with open(path, "w") as f:
            f.write("# V3 particle trajectories, Saturn-centred, km and km/s; k = index on the T0/10 grid, t_s = k T0/10; columns: " + longhdr + "\n")
            for k in ks:
                for i in range(N_KEEP):
                    f.write(f"{k},{tk[k]:.17g},{i}," + ",".join(f"{v:.17g}" for v in traj[k, i]) + "\n")
    write_long(REF / "v3_traj_orbits.csv", range(0, nt, SAVE_EVERY))
    write_long(REF / "v3_final.csv", [nt - 1])
    write_long(LARGE / "v3_traj_full.csv", range(nt))
    np.save(LARGE / "v3_traj_eps1e-11.npy", traj_eps)
    np.save(LARGE / "v3_shadow_sep.npy", sep_all)
    np.save(LARGE / "v3_radial_sep.npy", radial_sep)

    d_eps = np.linalg.norm(traj_eps[..., 0:3] - traj[..., 0:3], axis=-1)       # (nt, 100)
    dv_eps = np.linalg.norm(traj_eps[..., 3:6] - traj[..., 3:6], axis=-1)
    info = dict(
        seed=SEED, rng="numpy PCG64", batch=BATCH, n_keep=N_KEEP, rebound=rebound.__version__, python=platform.python_version(),
        a0_km=A0_KM, T0_s=T0, n_orbits=N_ORBITS, save_cadence_s=T0 / SAVE_EVERY, n_times=nt,
        band_km=[A0_KM - BAND_KM, A0_KM + BAND_KM], e_max=E_MAX, i_max_rad=I_MAX, epsilon=EPS,
        thresholds=dict(min_dist_hill_radii=RHILL_MIN, shadow_offset_km=SHADOW_KM, growth_factor_max=GROWTH_MAX),
        hill_radius_km=dict(zip(("prometheus", "pandora"), hill_radii_km(sat).tolist())),
        counts=dict(drawn=int(len(ok)), examined=drawn_used, accepted=N_KEEP,
                    rejected=int((~ok[:drawn_used]).sum()), close_approach=int(bad_close[:drawn_used].sum()),
                    shadow_growth=int(bad_growth[:drawn_used].sum()), both=int((bad_close & bad_growth)[:drawn_used].sum()),
                    passed_not_needed=int(ok[:drawn_used].sum() - N_KEEP)),
        growth_factor=dict(accepted_min=float(growth[accepted].min()), accepted_median=float(np.median(growth[accepted])),
                           accepted_max=float(growth[accepted].max()), all_max=float(growth[:drawn_used].max())),
        min_dist_hill_radii=dict(accepted_min=float(dmin[accepted].min()), rejected=[float(dmin[j].min()) for j in range(drawn_used) if not ok[j]]),
        error_floor=dict(
            description="REBOUND epsilon 1e-9 (the reference file) against epsilon 1e-11, same particles, same cadence, 100 accepted particles",
            final_position_km=d_eps[-1].tolist(), max_position_km=d_eps.max(axis=0).tolist(),
            final_velocity_km_s=dv_eps[-1].tolist(),
            worst_final_position_km=float(d_eps[-1].max()), worst_any_time_position_km=float(d_eps.max())),
        radial_shadow=dict(offset_km=SHADOW_KM, n_particles=N_RADIAL, final_separation_km=radial_sep[-1].tolist()),
        wall_s=dict(screening=t_screen, follow_up=t_follow, eps11_run=t_eps, total=wall),
        sha256_large={p.name: sha256(p) for p in sorted(LARGE.glob("v3_*"))},
        sha256_committed={p: sha256(REF / p) for p in ("v3_ic.csv", "v3_particles_meta.csv", "v3_traj.npy", "v3_moons.csv",
                                                       "v3_traj_orbits.csv", "v3_final.csv")},
    )
    (REF / "v3.json").write_text(json.dumps(info, indent=1) + "\n")
    analyse()


def analyse():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = json.loads((REF / "v3.json").read_text())
    T0 = info["T0_s"]
    traj = np.load(REF / "v3_traj.npy")
    eps = np.load(LARGE / "v3_traj_eps1e-11.npy")
    sep = np.load(LARGE / "v3_shadow_sep.npy")
    rsep = np.load(LARGE / "v3_radial_sep.npy")
    rows = [r for r in csv.reader(l for l in open(REF / "v3_particles_meta.csv") if not l.startswith("#"))]
    head, rows = rows[0], rows[1:]
    col = {h: np.array([r[i] for r in rows]) for i, h in enumerate(head)}
    num = lambda k: col[k].astype(float)
    status = col["status"]
    acc = status == "accepted"
    rej = ~np.isin(status, ["accepted", "not_needed"])
    thr = info["thresholds"]
    orb = np.arange(traj.shape[0]) / SAVE_EVERY

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    ax[0, 0].hist(num("a_km")[acc] - A0_KM, bins=15, color="C0")
    ax[0, 0].set(xlabel="a - a0 (km)", ylabel="particles", title="Guiding radius (accepted)")
    ax[0, 1].hist(num("e_epicyclic")[acc], bins=15, color="C0")
    ax[0, 1].set(xlabel="e (epicyclic)", title="Eccentricity")
    ax[0, 2].hist(num("i_epicyclic_rad")[acc], bins=15, color="C0")
    ax[0, 2].set(xlabel="i (rad)", title="Inclination")
    dm = np.minimum(num("min_dist_prometheus_RH"), num("min_dist_pandora_RH"))
    ax[1, 0].plot(num("a_km")[acc] - A0_KM, dm[acc], ".", label="accepted")
    ax[1, 0].plot(num("a_km")[rej] - A0_KM, dm[rej], "rx", label="rejected")
    ax[1, 0].axhline(thr["min_dist_hill_radii"], color="k", ls="--", lw=1)
    ax[1, 0].set(xlabel="a - a0 (km)", ylabel="closest approach to a moon (Hill radii)", title="Encounter test")
    ax[1, 0].legend()
    g = num("growth_factor")
    ax[1, 1].semilogy(np.arange(len(g))[acc], g[acc], ".", label="accepted")
    ax[1, 1].semilogy(np.arange(len(g))[rej], g[rej], "rx", label="rejected")
    ax[1, 1].axhline(thr["growth_factor_max"], color="k", ls="--", lw=1)
    ax[1, 1].set(xlabel="draw index", ylabel="max shadow separation / offset", title="Shadow (along velocity) growth")
    ax[1, 1].legend()
    ax[1, 2].semilogy(orb, sep.max(axis=1) / SHADOW_KM, label="worst candidate")
    ax[1, 2].semilogy(orb, np.median(sep, axis=1) / SHADOW_KM, label="median candidate")
    ax[1, 2].set(xlabel="orbits", ylabel="separation / offset", title="Shadow growth in time")
    ax[1, 2].legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "v3_ensemble.png", dpi=120)
    plt.close(fig)

    d = np.linalg.norm(eps[..., 0:3] - traj[..., 0:3], axis=-1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].semilogy(orb[1:], d.max(axis=1)[1:], label="worst particle")
    ax[0].semilogy(orb[1:], np.median(d, axis=1)[1:], label="median")
    ax[0].axhline(1.0, color="k", ls="--", lw=1, label="1 km tolerance")
    ax[0].set(ylim=(1e-9, 3), xlabel="orbits", ylabel="|r(1e-9) - r(1e-11)| (km)", title="REBOUND error floor")
    ax[0].legend()
    ax[1].semilogy(orb, rsep.max(axis=1), label="radial 1e-3 km offset (10 particles)")
    ax[1].semilogy(orb, sep.max(axis=1), label="along-velocity 1e-3 km offset (worst candidate)")
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].set(xlabel="orbits", ylabel="separation (km)", title="Kepler shear against chaos test")
    ax[1].legend()
    r = np.linalg.norm(traj[..., 0:3], axis=-1)
    for i in range(0, 100, 20):
        ax[2].plot(orb, r[:, i] - A0_KM, lw=0.8)
    ax[2].set(xlabel="orbits", ylabel="r - a0 (km)", title="Five sample particles (every 1/10 orbit)")
    fig.tight_layout()
    fig.savefig(REPORTS / "v3_floor.png", dpi=120)
    plt.close(fig)
    write_report(info)


def write_report(info):
    c, gf, ef, rs, w = info["counts"], info["growth_factor"], info["error_floor"], info["radial_shadow"], info["wall_s"]
    kep = 1.5 * 2 * rs['offset_km'] * 2 * np.pi * N_ORBITS
    sizes = {p.name: p.stat().st_size for p in sorted(REF.glob("v3_*"))}
    lsizes = {p.name: p.stat().st_size for p in sorted(LARGE.glob("v3_*"))}
    kb = lambda n: f"{n / 1024:.0f} kB" if n < 1e6 else f"{n / 1e6:.1f} MB"
    rej = ", ".join(f"{x:.2f}" for x in info["min_dist_hill_radii"]["rejected"]) or "none"
    txt = f"""# V3: REBOUND reference for test particles near the F ring core

**Read this first (bit-exact initial states).** The agreement target is 1 km after 100 orbits. A radial position offset of
only {rs['offset_km'] * 1e3:.0f} m changes the semi-major axis by 2 m and grows to about {np.median(rs['final_separation_km']):.2f} km by Kepler shear
over 100 orbits (median of {rs['n_particles']} particles; estimate 1.5 x 2 m x 2 pi x 100 = {kep:.2f} km). So a difference of about 1 m in a, or 1 part in 10^8 in
the mean motion, uses up the whole tolerance. The Rust test must read the initial states from `validation/reference/v3_ic.csv`
(17 significant digits, round-trip exact) and must not rebuild them from elements, and its moons must start from `params/moons.toml`
exactly as REBOUND's do.

Task V3 (PLAN section 9), serves C4. Role A6. All numbers below come from `validation/reference/v3.json`.

## Model
ADR 0004 as in V2: REBOUND 5.2.2, IAS15 with epsilon {info['epsilon']:g}, Saturn a free massive body, Prometheus and Pandora with
GM and states from `params/moons.toml` at t = 0 (the 2008-01-01 epoch), J2+J4+J6 acceleration from `fring_ref.force` acting on the moons and the
particles relative to Saturn, massless test particles, Saturn-centred outputs in Saturn's equator frame. The particles do not
act on the moons or on each other, and IAS15's step is set by the moons only (a particle run alone is bit-identical to the same particle
in the full run).

One orbit is T0 = 2 pi / n(a0) = {info['T0_s']:.3f} s ({info['T0_s'] / 3600:.3f} h), with n(a0) the J2+J4+J6 circular-orbit
frequency at a0 = {info['a0_km']} km (PLAN 7.2). 100 orbits are {100 * info['T0_s'] / 86400:.2f} days. Output every T0/10, 1001 times.

## Particles
Seed {info['seed']} (numpy PCG64). Per candidate, in this order: guiding radius a ~ U[a0 - {BAND_KM:g}, a0 + {BAND_KM:g}] km (a {2 * BAND_KM:g} km band),
epicyclic e ~ U[0, {E_MAX:g}], epicyclic i ~ U[0, {I_MAX:g}] rad, then three phases (radial epicycle, vertical, longitude) ~ U[0, 2 pi).
Initial states are first-order epicycles about the circular orbit at a with the exact n, kappa, nu (`fring_ref.particles.epicycle_states`).
e and i are epicyclic amplitudes, not osculating elements (the J2 osculating e of a circular orbit is about 4.5e-3, see V1).

## Selection (non-chaotic orbits)
- Close approach: the minimum distance to either moon over the run (cubic Hermite between samples every T0/100) must be at least
  {info['thresholds']['min_dist_hill_radii']:g} Hill radii (Hill radius {info['hill_radius_km']['prometheus']:.1f} km Prometheus, {info['hill_radius_km']['pandora']:.1f} km Pandora).
- Shadow test: a second particle offset by {info['thresholds']['shadow_offset_km']:g} km along the velocity (no first-order change of a, so no Kepler shear) must stay within
  {info['thresholds']['growth_factor_max']:g} times the offset for the whole run. A quiet orbit gives a growth factor of about 3: a position
  offset with unchanged velocity excites an epicycle of the same size, which is not chaos. A larger factor means that a 1 m change grows,
  so differences between codes would grow too; this happens for passes at 6 to 10 Hill radii.
- Drawn {c['drawn']}, examined in draw order until 100 passed: {c['examined']}. Rejected {c['rejected']}: {c['close_approach']} close approach, {c['shadow_growth']} shadow growth ({c['both']} both).
  Closest approaches of the rejected, in Hill radii: {rej}. Passed but not needed: {c['passed_not_needed']}.
- Growth factor of the accepted: min {gf['accepted_min']:.2f}, median {gf['accepted_median']:.2f}, max {gf['accepted_max']:.2f}
  (largest of all candidates {gf['all_max']:.2f}). Over 100 orbits (shorter than the synodic period of about 111 orbits) true chaos is not
  expected; the rejections are sensitive encounters, not chaotic orbits.
- Closest approach among the accepted: {info['min_dist_hill_radii']['accepted_min']:.2f} Hill radii.

## REBOUND's own error (floor of the reference)
Epsilon {info['epsilon']:g} against {EPS_FLOOR:g} for the same 100 particles: worst final position difference {ef['worst_final_position_km']:.2e} km, worst at any
output time {ef['worst_any_time_position_km']:.2e} km (per particle in `v3.json`, `error_floor`). Any Rust/REBOUND difference above this is not REBOUND's noise.

## Files (`validation/reference/`)
| File | Size | Content |
|---|---|---|
""" + "\n".join(f"| `{n}` | {kb(s)} | {d} |" for (n, s), d in zip(
        sizes.items(), [{
            "v3.json": "seed, T0, thresholds, counts, error floor, hashes, timings",
            "v3_final.csv": "final states (k = 1000), long CSV format",
            "v3_ic.csv": "initial states, `id,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s`",
            "v3_moons.csv": "moons at all 1001 times: `k,t_s,` then `prometheus_{x,y,z}_km,prometheus_v{x,y,z}_km_s`, then Pandora",
            "v3_particles_meta.csv": "all candidates: elements, screening numbers, status",
            "v3_traj.npy": "float64 (1001, 100, 6), [time k, id, (x,y,z,vx,vy,vz)], 1/10 orbit cadence",
            "v3_traj_orbits.csv": "every full orbit (k = 0, 10, ... 1000), long CSV: `k,t_s,id,x_km,...,vz_km_s`",
        }[n] for n in sizes])) + f"""

Not committed (`reference/large/`, SHA-256 in `v3.json`): {", ".join(f"`{n}` ({kb(s)})" for n, s in lsizes.items())}.

## Plots (`validation/reports/`)
- `v3_ensemble.png`: distributions of a, e, i; encounter test; shadow growth.
- `v3_floor.png`: REBOUND error floor, Kepler shear against the along-velocity shadow, sample trajectories.
- `v3_rust_vs_rebound.png`: written by `scripts/v3_compare.py --rust <file>` from a Rust trajectory in the same long CSV format.

## Run time
Screening ({c['examined']} candidates + shadows in one run): {w['screening']:.0f} s. Follow-up runs in parallel: {w['follow_up']:.0f} s.
Total {w['total']:.0f} s on this Mac (Python {platform.python_version()}, {platform.machine()}).

## Commands
```sh
cd validation && uv run python scripts/v3_particles.py --check   # first orbit of 3 particles
cd validation && uv run python scripts/v3_particles.py           # full run
cd validation && uv run python scripts/v3_compare.py --rust <file.csv>   # compare a Rust trajectory
```
"""
    (REPORTS / "v3.md").write_text(txt)


def check():
    """Recompute the first orbit of particles 0 to 2 alone and compare with the committed file."""
    sat = load_saturn()
    T0 = orbit_period(sat)
    ic = np.loadtxt(REF / "v3_ic.csv", delimiter=",", comments="#")[:3, 1:]
    ref = np.load(REF / "v3_traj.npy")[:SAVE_EVERY + 1, :3]
    t = np.arange(FINE + 1) * (T0 / FINE)
    _, parts = simulate(sat, ic, t, EPS)
    new = parts[::SAVE_EVERY]
    dpos = np.abs(new[..., 0:3] - ref[..., 0:3]).max()
    dvel = np.abs(new[..., 3:6] - ref[..., 3:6]).max()
    print(f"first orbit, 3 particles: max |dr| = {dpos:.3e} km, max |dv| = {dvel:.3e} km/s")
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
        full_run()
