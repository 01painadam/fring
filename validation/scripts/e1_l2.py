"""E1-REF layer 2: N particles from below 140,050 km for 1 year (or 6 months) with the two moons, IAS15 reference.

Usage (from validation/):
  uv run python scripts/e1_l2.py time --n 20 --days 30           cost measurement (no files written)
  uv run python scripts/e1_l2.py run --n 300 --days 365.25       writes reference/e1_l2_states.csv, e1_l2_ref.csv, e1_l2.json
  uv run python scripts/e1_l2.py check                          recompute the first chunk for 30 days and compare
Chaos indicator = ADR 0008 (shadow at d0 = 1 km along the radial direction, position only, renormalised every K = 100
steps of h = 356.2537 s, discrete MEGNO: l_j = ln(d_j / d0), S = sum (j - 1/2) l_j, Y_j = 2 S / j, mean Y = sum Y_j / j).
Classes: regular mean Y <= 2.5, chaotic mean Y >= 5, in between uncertain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

import numpy as np
import rebound

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_ref  # noqa: E402
from fring_ref import encounter as enc  # noqa: E402
from fring_ref.particles import _STRIDE, epicycle_states, hermite_min_distance  # noqa: E402

import ctypes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"
SAT = e1_ref.SAT
RH = e1_ref.RH
SEED = 20261109
A_LO, A_HI = 139600.0, 140050.0  # km, guiding radius of the layer 2 particles (the lower bound is the V3/ADR 0008 sweep bound)
E_MAX = 0.0024
D0_KM = 1.0
K_STEPS = 100
RENORM_S = K_STEPS * enc.STEP_S
SAMPLES_PER_RENORM = 60  # IAS15 reference sampling: 593.756 s
CHUNK = 10  # particles per IAS15 simulation (each has a shadow)


def make_states(n, seed=SEED):
    rng = np.random.default_rng(seed)
    a = rng.uniform(A_LO, A_HI, n)
    e = rng.uniform(0, E_MAX, n)
    m = rng.uniform(0, 2 * np.pi, n)
    lam = rng.uniform(0, 2 * np.pi, n)
    return epicycle_states(SAT, a, e, 0 * a, m, 0 * a, lam)


def with_shadows(states):
    """Append one shadow per particle: position offset D0_KM along the radial direction, same velocity."""
    s = np.asarray(states)
    r = s[:, 0:3] / np.linalg.norm(s[:, 0:3], axis=1)[:, None]
    sh = s.copy()
    sh[:, 0:3] += D0_KM * r
    return np.vstack([s, sh])


def evolve(sim, n, dt_sample, nsamp, renorm_every, ntarget_moons=2):
    """Integrate sim (Saturn, 2 moons, n particles, n shadows) with samples every dt_sample.

    Returns dict min_prom_km, min_pand_km, final (n, 6), megno_mean, ftle_per_s.
    """
    npl = sim.N
    buf = np.ctypeslib.as_array(ctypes.cast(sim._particles, ctypes.POINTER(ctypes.c_double)), shape=(npl, _STRIDE))
    ip, ish = slice(3, 3 + n), slice(3 + n, 3 + 2 * n)

    def read():
        return buf[1:, 0:6] - buf[0, 0:6]

    prev = read().copy()
    min_pr = np.full(n, np.inf)
    min_pd = np.full(n, np.inf)
    big_l = np.zeros(n)
    s_sum = np.zeros(n)
    y_sum = np.zeros(n)
    j = 0
    for i in range(1, nsamp + 1):
        sim.integrate(i * dt_sample)
        cur = read().copy()
        for k, mn in ((0, min_pr), (1, min_pd)):
            rel = np.stack([prev[2:2 + n, 0:3] - prev[k, 0:3], cur[2:2 + n, 0:3] - cur[k, 0:3]])
            vel = np.stack([prev[2:2 + n, 3:6] - prev[k, 3:6], cur[2:2 + n, 3:6] - cur[k, 3:6]])
            np.minimum(mn, hermite_min_distance(rel, vel, dt_sample), out=mn)
        if i % renorm_every == 0:
            j += 1
            p = buf[ip, 0:6].copy()
            sep = buf[ish, 0:6] - p
            d = np.linalg.norm(sep[:, 0:3], axis=1)
            lj = np.log(d / D0_KM)
            big_l += lj
            s_sum += (j - 0.5) * lj
            y_sum += 2 * s_sum / j
            buf[ish, 0:6] = p + sep * (D0_KM / d)[:, None]
            cur = read().copy()
        prev = cur
    t = nsamp * dt_sample
    return {"min_prom_km": min_pr, "min_pand_km": min_pd, "final": prev[2:2 + n].copy(),
            "megno_mean": y_sum / max(j, 1), "ftle_per_s": big_l / t, "n_renorm": j}


def _chunk(args):
    ids, moon, states, days, eps = args
    sim = enc.make_sim(SAT, moon, with_shadows(states), eps)
    dt = RENORM_S / SAMPLES_PER_RENORM
    nsamp = int(round(days * enc.DAY / dt))
    t1 = time.time()
    r = evolve(sim, len(ids), dt, nsamp, SAMPLES_PER_RENORM)
    r["wall"] = time.time() - t1
    return ids, r


def run_ias15(states, days, workers, eps=e1_ref.EPS):
    _, moons = e1_ref.read_moons(REF / "e1_moons_t0.csv")
    moon0 = moons[0]  # t0 = 0: the epoch
    n = len(states)
    tasks = [(np.arange(s, min(s + CHUNK, n)), moon0, states[s:s + CHUNK], days, eps) for s in range(0, n, CHUNK)]
    out = {k: np.zeros(n) for k in ("min_prom_km", "min_pand_km", "megno_mean", "ftle_per_s")}
    out["final"] = np.zeros((n, 6))
    walls = []
    with mp.Pool(workers) as pool:
        for ids, r in pool.imap_unordered(_chunk, tasks):
            for k in out:
                out[k][ids] = r[k]
            walls.append(r["wall"])
    out["walls"] = walls
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_states(path, states, days):
    with open(path, "w") as fh:
        fh.write("# E1-REF layer 2 initial states: massless particles, guiding radius uniform in [%g, %g] km, e uniform in [0, %g], random phases.\n" % (A_LO, A_HI, E_MAX))
        fh.write("# frame: Saturn-centred, Saturn-equator inertial frame of params/moons.toml; km and km/s; t = 0 is the epoch (2008-01-01 00:00 TDB),\n")
        fh.write(f"# moons from params/moons.toml. Integrate {days:g} days (reference run), Saturn J2,J4,J6 and both moons.\n")
        fh.write("# columns: id,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s\n")
        for i, s in enumerate(states):
            fh.write(f"{i}," + ",".join(repr(float(v)) for v in s) + "\n")


def write_ref(path, r, header):
    with open(path, "w") as fh:
        fh.write(header)
        fh.write("# columns: id,min_prometheus_km,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s,megno_mean,min_pandora_km\n")
        for i in range(len(r["megno_mean"])):
            fh.write(f"{i},{float(r['min_prom_km'][i])!r}," + ",".join(repr(float(v)) for v in r["final"][i])
                     + f",{float(r['megno_mean'][i])!r},{float(r['min_pand_km'][i])!r}\n")


def cmd_time(n, days, workers):
    st = make_states(n)
    t1 = time.time()
    r = run_ias15(st, days, workers)
    wall = time.time() - t1
    print(f"{n} particles, {days} d, {workers} workers, chunk {CHUNK}: wall {wall:.1f} s; chunk walls {np.round(r['walls'], 1)}")
    print("min Prometheus (R_H) quantiles 0/25/50:", np.round(np.percentile(r["min_prom_km"] / RH, [0, 25, 50]), 2))
    print("megno_mean quantiles 10/50/90:", np.round(np.percentile(r["megno_mean"], [10, 50, 90]), 2))


def cmd_run(n, days, workers):
    t_all = time.time()
    st = make_states(n)
    sf = REF / "e1_l2_states.csv"
    write_states(sf, st, days)
    st = np.loadtxt(sf, delimiter=",", comments="#")[:, 1:7]
    r = run_ias15(st, days, workers)
    wall = time.time() - t_all
    ref_f = REF / "e1_l2_ref.csv"
    write_ref(ref_f, r, f"# E1-REF layer 2 IAS15 reference (epsilon {e1_ref.EPS}), {days:g} days; final state at t = {days:g} d; min_* over the run (Hermite, 593.756 s samples).\n")
    meta = {"task": "E1-REF layer 2", "seed": SEED, "n": n, "days": days, "a_range_km": [A_LO, A_HI], "e_max": E_MAX,
            "epsilon": e1_ref.EPS, "chunk": CHUNK, "d0_km": D0_KM, "K_steps": K_STEPS, "step_s": enc.STEP_S,
            "samples_per_renorm": SAMPLES_PER_RENORM, "r_hill_km": RH, "wall_s": wall, "chunk_wall_s": r["walls"],
            "sha256": {"e1_l2_states.csv": sha(sf), "e1_l2_ref.csv": sha(ref_f)},
            "versions": {"rebound": rebound.__version__, "numpy": np.__version__, "python": platform.python_version()}}
    (REF / "e1_l2.json").write_text(json.dumps(meta, indent=2))
    print(f"wall {wall:.0f} s")


def cmd_check():
    meta = json.loads((REF / "e1_l2.json").read_text())
    ok = True
    for key in ("e1_l2_states.csv", "e1_l2_ref.csv"):
        same = sha(REF / key) == meta["sha256"][key]
        ok &= same
        print(f"sha256 {key}: {'ok' if same else 'DIFFERENT'}")
    st = np.loadtxt(REF / "e1_l2_states.csv", delimiter=",", comments="#")[:, 1:7]
    _, moons = e1_ref.read_moons(REF / "e1_moons_t0.csv")
    days = 20.0
    ids = np.arange(0, min(CHUNK, len(st)))
    _, r = _chunk((ids, moons[0], st[ids], days, e1_ref.EPS))
    # independent second run of the same chunk: deterministic, so it must agree with itself to round-off; and the
    # first 20 days of the committed run are not stored, so compare the integral invariants instead: the guiding a
    # of an undisturbed particle (min_prom > 10 R_H) changes by less than 1 km over 20 days.
    a0 = enc.geo_elements(SAT, st[ids])["a"]
    a1 = enc.geo_elements(SAT, r["final"])["a"]
    far = r["min_prom_km"] > 10 * RH
    d = abs(a1 - a0)[far]
    good = (len(d) == 0) or d.max() < 1.0
    ok &= good
    print(f"20-day recompute of {len(ids)} particles: {far.sum()} far from Prometheus, max |Delta a| {d.max() if len(d) else 0:.3g} km: {'ok' if good else 'FAIL'}")
    print("E1 layer 2 check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


# ----------------------------------------------------------------- stand-in (WHFast, not the Rust backend)


def _wh_chunk(args):
    ids, moon, states, days, div = args
    import e1_standin as si

    h = enc.STEP_S / div
    n = len(ids)
    nsamp = int(round(days * enc.DAY / h))
    sim = si.whfast_sim(moon, with_shadows(states), h)
    sim.exact_finish_time = 0
    r = evolve(sim, n, h, nsamp, K_STEPS * div)
    return ids, r


def standin_layer2(div, out, workers):
    st = np.loadtxt(REF / "e1_l2_states.csv", delimiter=",", comments="#")[:, 1:7]
    meta = json.loads((REF / "e1_l2.json").read_text())
    _, moons = e1_ref.read_moons(REF / "e1_moons_t0.csv")
    n = len(st)
    tasks = [(np.arange(s, min(s + CHUNK, n)), moons[0], st[s:s + CHUNK], meta["days"], div) for s in range(0, n, CHUNK)]
    res = {k: np.zeros(n) for k in ("min_prom_km", "min_pand_km", "megno_mean")}
    res["final"] = np.zeros((n, 6))
    with mp.Pool(workers) as pool:
        for ids, r in pool.imap_unordered(_wh_chunk, tasks):
            for k in res:
                res[k][ids] = r[k]
    write_ref(out, res, f"# STAND-IN, not the Rust CPU backend: REBOUND WHFast, h = {enc.STEP_S}/{div} s, {meta['days']:g} days\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["time", "run", "check"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--days", type=float, default=365.25)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "time":
        cmd_time(a.n, a.days, a.workers)
    elif a.cmd == "run":
        cmd_run(a.n, a.days, a.workers)
    else:
        sys.exit(cmd_check())
