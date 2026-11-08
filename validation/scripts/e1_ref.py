"""E1-REF layer 1: IAS15 reference for 2-day passes of a massless particle near Prometheus (ICR-004 section 6).

Usage (from validation/):
  uv run python scripts/e1_ref.py layer1 [--workers 5]   build the states, run IAS15 (eps 1e-9 and 1e-11), write reference/e1_l1_*
  uv run python scripts/e1_ref.py analyse1               distributions against docs/results/data, report + figure
  uv run python scripts/e1_ref.py check                  recompute two committed chunks and compare (under a minute)
Files (reference/): e1_moons_t0.csv, e1_l1_states.csv, e1_l1_ref.csv, e1_l1.json.  See docs/handoffs/E1-REF.md.
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

from fring_ref import encounter as enc
from fring_ref.moons import run as moons_run
from fring_ref.params import load_saturn

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"
SEED = 20261108
PER_STRATUM = 500
EPS = 1e-9
EPS_CHECK = 1e-11
DT_OUT = 600.0  # s, sampling of the pass (Hermite minimum)
CHUNK = 25  # particles per IAS15 simulation in the reference run (fixed, so the run is reproducible)
EDGE_MARGIN = 1e-3  # R_H: candidates this close to a stratum edge are dropped
TIMES = np.arange(0.0, enc.WINDOW_S + 1.0, DT_OUT)

SAT = load_saturn()
RH = enc.r_hill_prometheus(SAT)

STATE_COLS = ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ moons at the epochs t0


def moons_at_t0():
    """Prometheus and Pandora (IAS15 from the params/moons.toml epoch, as V2) at t0 = 0, 10, ..., 350 d."""
    f = REF / "e1_moons_t0.csv"
    if f.exists():
        return read_moons(f)
    out = moons_run(SAT, 350 * enc.DAY, 10 * enc.DAY, epsilon=EPS)  # (36, 13): t, 2 x 6
    with open(f, "w") as fh:
        fh.write("# E1-REF moon states at the encounter epochs t0 = j x 10 d after the epoch of params/moons.toml (2008-01-01 TDB).\n")
        fh.write("# IAS15 epsilon 1e-9 from the epoch (as V2), Saturn-centred, Saturn-equator frame, km and km/s.\n")
        fh.write("# columns: j,t0_s,prometheus_x_km,prometheus_y_km,prometheus_z_km,prometheus_vx_km_s,prometheus_vy_km_s,"
                 "prometheus_vz_km_s,pandora_x_km,pandora_y_km,pandora_z_km,pandora_vx_km_s,pandora_vy_km_s,pandora_vz_km_s\n")
        for j, row in enumerate(out):
            fh.write(f"{j}," + ",".join(repr(float(v)) for v in row) + "\n")
    return read_moons(f)


def read_moons(f):
    d = np.loadtxt(f, delimiter=",", comments="#")
    return d[:, 1].copy(), d[:, 2:].reshape(-1, 2, 6)  # t0_s (36,), moons (36, 2, 6)


# ------------------------------------------------------------------ screening and reference runs


def _screen(args):
    """One IAS15 simulation of a batch of candidates at one t0.  Returns accepted rows."""
    j, moon, x_rh, e, m = args
    sign = np.where(x_rh >= 0, 1.0, -1.0)
    st = enc.hill_frame_states(SAT, moon[0], x_rh * RH, sign * 10 * RH, e, m)
    mo, pa = enc.run_pass(SAT, moon, st, TIMES, EPS)
    r = enc.pass_measures(mo, pa, DT_OUT, RH)
    ok = r["interior"] & (r["d_end_rh"] > 3.0) & (r["b_rh"] > enc.STRATA[0][0] + EDGE_MARGIN) \
        & (r["b_rh"] < enc.STRATA[-1][1] - EDGE_MARGIN)
    for lo, hi in enc.STRATA[1:]:
        ok &= ~((abs(r["b_rh"] - lo) < EDGE_MARGIN))
    ok &= ~(abs(r["b_rh"] - 1.0) < EDGE_MARGIN)
    idx = np.nonzero(ok)[0]
    return [(j, float(x_rh[i]), float(e[i]), float(m[i]), st[i], float(r["b_rh"][i])) for i in idx]


def stratum_of(b):
    for k, (lo, hi) in enumerate(enc.STRATA):
        if lo <= b < hi or (k == len(enc.STRATA) - 1 and b == hi):
            return k
    return -1


def build_states(workers, log=print):
    t0_s, moons = moons_at_t0()
    rng = np.random.default_rng(SEED)
    accepted = [[] for _ in enc.STRATA]
    rnd = 0
    t_start = time.time()
    with mp.Pool(workers) as pool:
        while min(len(a) for a in accepted) < PER_STRATUM:
            rnd += 1
            tasks = []
            for k, (lo, hi) in enumerate(enc.STRATA):
                need = PER_STRATUM - len(accepted[k])
                if need <= 0:
                    continue
                ncand = int(min(max(need / 0.10, 400), 6000))
                nchunks = int(np.ceil(ncand / 200))
                for _ in range(nchunks):
                    j = int(rng.integers(0, len(t0_s)))
                    x = rng.uniform(0.6 * lo, 1.5 * hi + 1.0, 200) * rng.choice([-1.0, 1.0], 200)
                    e = rng.uniform(0.0, 0.0024, 200)
                    m = rng.uniform(0.0, 2 * np.pi, 200)
                    tasks.append((j, moons[j], x, e, m))
            for res in pool.imap_unordered(_screen, tasks):
                for row in res:
                    k = stratum_of(row[5])
                    if k >= 0 and len(accepted[k]) < PER_STRATUM:
                        accepted[k].append(row)
            log(f"round {rnd}: counts {[len(a) for a in accepted]}  ({time.time() - t_start:.0f} s)")
    return t0_s, moons, accepted


def write_states(path, accepted, t0_s):
    ids = []
    with open(path, "w") as fh:
        fh.write("# E1-REF layer 1 initial states: massless particle near Prometheus, start about 10 R_H upstream.\n")
        fh.write("# frame: Saturn-centred, Saturn-equator inertial frame of params/moons.toml; km and km/s.\n")
        fh.write("# time: each state is at t0_s seconds after the epoch (2008-01-01 00:00 TDB); moons at t0_s are in e1_moons_t0.csv (row j = t0_idx);\n")
        fh.write(f"# integrate {enc.WINDOW_S:.0f} s (2 days) from t0_s with Saturn J2,J4,J6 (params/saturn.toml) and both moons.\n")
        fh.write("# stratum: 0..3 = measured closest distance b in [0.68,1), [1,2), [2,4.1), [4.1,6.3] R_H (R_H = %.6f km).\n" % RH)
        fh.write("# columns: id,stratum,t0_idx,t0_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s\n")
        i = 0
        for k, rows in enumerate(accepted):
            for (j, x_rh, e, m, st, b) in rows:
                fh.write(f"{i},{k},{j},{float(t0_s[j])!r}," + ",".join(repr(float(v)) for v in st) + "\n")
                ids.append(i)
                i += 1
    return ids


def read_states(path=None):
    path = path or REF / "e1_l1_states.csv"
    d = np.loadtxt(path, delimiter=",", comments="#")
    return {"id": d[:, 0].astype(int), "stratum": d[:, 1].astype(int), "t0_idx": d[:, 2].astype(int),
            "t0_s": d[:, 3], "state": d[:, 4:10]}


def chunks(t0_idx):
    """Fixed chunks: ids with the same t0_idx, in id order, CHUNK at a time.  Returns list of id arrays."""
    out = []
    for j in np.unique(t0_idx):
        ids = np.nonzero(t0_idx == j)[0]
        for s in range(0, len(ids), CHUNK):
            out.append(ids[s:s + CHUNK])
    return out


def _ref_chunk(args):
    ids, j, moon, states, eps = args
    mo, pa = enc.run_pass(SAT, moon, states, TIMES, eps)
    r = enc.pass_measures(mo, pa, DT_OUT, RH)
    da, de, dv = enc.delta_elements(SAT, pa[0], pa[-1])
    return ids, {"b_rh": r["b_rh"], "u_kms": r["u_kms"], "da_km": da, "de": de, "dvarpi_rad": dv, "end": pa[-1]}


def run_reference(st, moons, eps, workers):
    tasks = [(ids, int(st["t0_idx"][ids[0]]), moons[st["t0_idx"][ids[0]]], st["state"][ids], eps)
             for ids in chunks(st["t0_idx"])]
    n = len(st["id"])
    out = {k: np.zeros(n) for k in ("b_rh", "u_kms", "da_km", "de", "dvarpi_rad")}
    out["end"] = np.zeros((n, 6))
    with mp.Pool(workers) as pool:
        for ids, r in pool.imap_unordered(_ref_chunk, tasks):
            for k, v in r.items():
                out[k][ids] = v
    return out


def layer1(workers):
    REF.mkdir(exist_ok=True)
    t_all = time.time()
    t0_s, moons, accepted = build_states(workers)
    t_screen = time.time() - t_all
    states_f = REF / "e1_l1_states.csv"
    write_states(states_f, accepted, t0_s)
    st = read_states(states_f)
    t1 = time.time()
    main = run_reference(st, moons, EPS, workers)
    t_main = time.time() - t1
    t1 = time.time()
    fine = run_reference(st, moons, EPS_CHECK, workers)
    t_fine = time.time() - t1
    strata_ok = [int(((main["b_rh"][st["stratum"] == k] >= lo) & (main["b_rh"][st["stratum"] == k] <= hi)).sum())
                 for k, (lo, hi) in enumerate(enc.STRATA)]
    with open(REF / "e1_l1_ref.csv", "w") as fh:
        fh.write("# E1-REF layer 1 IAS15 reference (epsilon 1e-9) and its check run (epsilon 1e-11). One row per id of e1_l1_states.csv.\n")
        fh.write("# da_km, de, dvarpi_rad: geometric elements (src/fring_ref/encounter.py) at t0+2 d minus at t0; dvarpi wrapped to (-pi, pi]\n")
        fh.write("# and includes the secular apse precession of about 0.09 rad in 2 d, which is the same in every method.\n")
        fh.write("# columns: id,stratum,b_rh,u_kms,da_km,de,dvarpi_rad,da_eps11_km,de_eps11,dvarpi_eps11_rad\n")
        for i in st["id"]:
            vals = [main["b_rh"][i], main["u_kms"][i], main["da_km"][i], main["de"][i], main["dvarpi_rad"][i],
                    fine["da_km"][i], fine["de"][i], fine["dvarpi_rad"][i]]
            fh.write(f"{i},{st['stratum'][i]}," + ",".join(repr(float(v)) for v in vals) + "\n")
    meta = {
        "task": "E1-REF layer 1", "seed": SEED, "numpy_generator": "numpy.random.default_rng(seed)",
        "window_s": enc.WINDOW_S, "r_hill_km": RH, "strata_rh": enc.STRATA, "per_stratum": PER_STRATUM,
        "strata_counts_in_range": strata_ok, "epsilon": EPS, "epsilon_check": EPS_CHECK, "dt_out_s": DT_OUT,
        "chunk": CHUNK, "t0_grid_days": [0, 350, 10], "start": "about 10 R_H upstream (x radial offset |x| in 0..10 R_H, e in 0..0.0024)",
        "sha256": {"e1_l1_states.csv": sha256(states_f), "e1_moons_t0.csv": sha256(REF / "e1_moons_t0.csv"),
                   "e1_l1_ref.csv": sha256(REF / "e1_l1_ref.csv")},
        "wall_s": {"screen": t_screen, "ref_eps1e-9": t_main, "ref_eps1e-11": t_fine, "workers": workers},
        "versions": {"rebound": rebound.__version__, "numpy": np.__version__, "python": platform.python_version()},
    }
    (REF / "e1_l1.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["strata_counts_in_range"]), meta["wall_s"])


# ------------------------------------------------------------------ analysis and check


def analyse1():
    from scipy import stats

    st = read_states()
    ref = np.loadtxt(REF / "e1_l1_ref.csv", delimiter=",", comments="#")
    meta = json.loads((REF / "e1_l1.json").read_text())
    hist = np.loadtxt(ROOT.parent / "docs" / "results" / "data" / "closest_hist.csv", delimiter=",", skiprows=1)
    lines = []
    lines.append(f"R_H = {RH:.4f} km; seed {SEED}; IAS15 epsilon {EPS} (check {EPS_CHECK}); window 2 d")
    lines.append("")
    lines.append("Reference error, |Delta a(1e-9) - Delta a(1e-11)| / |Delta a(1e-11)| (the limits are 5 % and 25 %):")
    lines.append(f"{'stratum':>10} {'n':>4} {'median':>10} {'p95':>10} {'max':>10} {'median |da| km':>15} {'n |da|<10 floor':>16}")
    floor = 10 * 1e-9  # 10 x noise floor of the geometric a (km); see test_geometric_a_is_flat
    for k, (lo, hi) in enumerate(enc.STRATA):
        s = ref[:, 1] == k
        da, da2 = ref[s, 4], ref[s, 7]
        rel = abs(da - da2) / abs(da2)
        lines.append(f"[{lo},{hi}] {s.sum():>4} {np.median(rel):10.2e} {np.percentile(rel, 95):10.2e} {rel.max():10.2e} "
                     f"{np.median(abs(da2)):15.3e} {(abs(da2) < floor).sum():>16}")
    lines.append("")
    lines.append("Delta e and apse angle, eps 1e-9 against 1e-11: max |d(Delta e)| = %.2e, max |d(Delta varpi)| = %.2e rad" % (
        abs(ref[:, 5] - ref[:, 8]).max(), abs(ref[:, 6] - ref[:, 9]).max()))
    lines.append("")
    lines.append("Measured b within each stratum against the closest-approach density of docs/results/data/closest_hist.csv")
    lines.append("(counts per bin in b; the run has a density that rises with ln b; the synthetic set is built per stratum):")
    edges = np.concatenate([hist[:, 0], hist[-1:, 1]])
    cnt = hist[:, 2]
    for k, (lo, hi) in enumerate(enc.STRATA):
        b = ref[ref[:, 1] == k, 2]
        # piecewise-constant density per R_H from the histogram (counts / width), clipped to the stratum
        w = np.diff(edges)
        dens = cnt / w
        grid = np.linspace(lo, hi, 2001)
        d = dens[np.clip(np.searchsorted(edges, grid, side="right") - 1, 0, len(dens) - 1)]
        cdf_tab = np.concatenate([[0], np.cumsum((d[1:] + d[:-1]) / 2 * np.diff(grid))])
        cdf_tab /= cdf_tab[-1]
        ks = stats.kstest(b, lambda x: np.interp(x, grid, cdf_tab))
        lines.append(f"  stratum [{lo},{hi}]: ours median {np.median(b):.3f}, run-density median {np.interp(0.5, cdf_tab, grid):.3f} R_H; "
                     f"KS statistic {ks.statistic:.3f} (p = {ks.pvalue:.2g})")
    lines.append("")
    lines.append("Relative speed at the closest sample u (km/s), 5/50/95 %%: %s" % np.round(np.percentile(ref[:, 3], [5, 50, 95]), 4))
    lines.append("u not checked against the run: no per-encounter data in docs/results/data.")
    text = "\n".join(lines)
    print(text)
    (ROOT / "reports" / "e1_l1_distribution.txt").write_text(text + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].hist(ref[:, 2], bins=np.linspace(0.68, 6.3, 40), color="C0")
        ax[0].set_xlabel("measured closest distance b (R_H)")
        ax[0].set_ylabel("encounters")
        ax[0].set_title("synthetic set (500 per stratum)")
        ax[1].stairs(cnt / np.diff(edges), edges, color="C1")
        ax[1].set_xlim(0.5, 9)
        ax[1].set_xlabel("closest approach (R_H)")
        ax[1].set_ylabel("run count per R_H")
        ax[1].set_title("closest_hist.csv")
        for k in range(4):
            s = ref[:, 1] == k
            rel = abs(ref[s, 4] - ref[s, 7]) / abs(ref[s, 7])
            ax[2].loglog(np.sort(rel), 1 - np.arange(s.sum()) / s.sum(), label=f"stratum {k}")
        ax[2].axvline(0.05, color="k", ls=":")
        ax[2].set_xlabel("|da(1e-9) - da(1e-11)| / |da|")
        ax[2].set_ylabel("fraction above")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(ROOT / "reports" / "e1_l1_distribution.png", dpi=120)
    except Exception as exc:  # plotting must not stop the numbers
        print("plot failed:", exc, file=sys.stderr)


def check():
    st = read_states()
    meta = json.loads((REF / "e1_l1.json").read_text())
    ref = np.loadtxt(REF / "e1_l1_ref.csv", delimiter=",", comments="#")
    ok = True
    for key in ("e1_l1_states.csv", "e1_moons_t0.csv", "e1_l1_ref.csv"):
        same = sha256(REF / key) == meta["sha256"][key]
        ok &= same
        print(f"sha256 {key}: {'ok' if same else 'DIFFERENT'}")
    _, moons = read_moons(REF / "e1_moons_t0.csv")
    ch = chunks(st["t0_idx"])
    for ids in (ch[0], ch[len(ch) // 2]):
        j = int(st["t0_idx"][ids[0]])
        _, r = _ref_chunk((ids, j, moons[j], st["state"][ids], EPS))
        for key, col in (("da_km", 4), ("de", 5), ("dvarpi_rad", 6)):
            d = abs(r[key] - ref[ids, col]).max()
            tol = 1e-6 * abs(ref[ids, col]).max() + 1e-9
            good = d <= tol
            ok &= good
            print(f"chunk j={j} ids {ids[0]}..{ids[-1]} {key}: max diff {d:.2e} (tol {tol:.1e}) {'ok' if good else 'FAIL'}")
    print("E1 layer 1 check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["layer1", "analyse1", "check"])
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "layer1":
        layer1(a.workers)
    elif a.cmd == "analyse1":
        analyse1()
    else:
        sys.exit(check())
