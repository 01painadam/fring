"""K4: deep-collision fraction of the hard-sphere runs (SPEC section 8) and the missed-pair estimate.

Usage: uv run python scripts/k4_deep.py <set> [--runs-dir ../../../runs/k4] [--w0 80] [--w1 200]
set is the run-name prefix, for example t01 (runs <set>-hs-s1 ...). Reads the raw stats.csv and the first
frame header of each HS run, and data/<set>-hs-s<k>.csv (from k4_reduce.py) for the velocity dispersions.

Definitions
- Orbit period P = 2 pi / n0, with n0 read from the frame header. Orbit k means t in ((k-1) P, k P].
- A stats row is assigned to an orbit by its END time (it holds the collisions of the steps since the
  previous row). Rows are 89 steps (0.1 orbit) wide and an orbit is 891 steps, so an orbit boundary is
  resolved to one row (at most 0.1 orbit). The groups are disjoint and cover all rows, so they add up exactly.
- The run is a little longer than 200 orbits (duration 0.3389 yr), so the last rows have k = 201; they form
  the group 'after orbit 200' so that the groups tile the whole run.
- Deep collision: overlap depth > 0.2 s (SPEC section 8). Limit: more than 1 % deep in a RUN is too large a step.
  The verdict line is the whole-run fraction. The other groups are diagnostics.
- Missed-pair estimate (SPEC section 8): 2.09 (<g> h / 4 s)^2, with the proxy
  <g> = sqrt(8/pi) * sqrt(2) * sigma_eff, sigma_eff^2 = (c_rho^2 + c_phi^2 + c_z^2) / 3 (isotropic Maxwell
  pair speed with the window-mean dispersions of the run). It ignores the shear speed of a close pair.
Writes data/<set>_deep.csv.
"""
import sys, os, csv, struct, argparse, tomllib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("set"); ap.add_argument("--runs-dir", default="../../../runs/k4")
ap.add_argument("--w0", type=float, default=80); ap.add_argument("--w1", type=float, default=200)
a = ap.parse_args()
GROUPS = [("whole run", 0, None)] + [(f"orbit {k}", k - 1, k) for k in range(1, 6)] + \
         [(f"orbits 6-{a.w0 - 1:g}", 5, a.w0 - 1), (f"window {a.w0:g}-{a.w1:g}", a.w0 - 1, a.w1), (f"after orbit {a.w1:g}", a.w1, None)]
# group bounds (lo, hi] on the orbit index k = ceil(t / P); 'orbits 80-200' means k in [80, 200] = (79, 200]
out = []
print(f"{'run':12s} {'group':14s} {'deep':>8s} {'collisions':>12s} {'deep %':>8s}")
for s in (1, 2, 3, 4):
    name = f"{a.set}-hs-s{s}"
    rd = os.path.join(a.runs_dir, name)
    with open(os.path.join(rd, "frames", "000000.bin"), "rb") as f:
        b = f.read(96)
    n0 = struct.unpack_from("<dddd", b, 40)[2]
    P = 2 * np.pi / n0
    cfg = tomllib.load(open(os.path.join(rd, "config.toml"), "rb"))
    rows = list(csv.DictReader(open(os.path.join(rd, "stats.csv"))))
    t = np.array([float(r["time_s"]) for r in rows])
    deep = np.array([int(r["deep_collisions"]) for r in rows])
    coll = np.array([int(r["collisions_since_last"]) for r in rows])
    k = np.ceil(t / P - 1e-9)       # orbit index of each row, 0 for the t = 0 row
    tot = {}
    for g, lo, hi in GROUPS:
        m = (k > lo) if hi is None else (k > lo) & (k <= hi)
        if g == "whole run": m = k > 0
        d, c = int(deep[m].sum()), int(coll[m].sum())
        tot[g] = (d, c)
        out.append(dict(run=name, group=g, deep=d, collisions=c, deep_fraction=d / c if c else float("nan")))
        print(f"{name:12s} {g:14s} {d:8d} {c:12d} {100 * d / c if c else float('nan'):8.4f}")
    parts = [tot[g] for g in ("orbit 1", "orbit 2", "orbit 3", "orbit 4", "orbit 5", f"orbits 6-{a.w0 - 1:g}", f"window {a.w0:g}-{a.w1:g}", f"after orbit {a.w1:g}")]
    sd = sum(p[0] for p in parts); sc = sum(p[1] for p in parts)
    print(f"{name:12s} sum of parts   {sd:8d} {sc:12d}   whole run {tot['whole run'][0]} {tot['whole run'][1]}  "
          f"{'ADDS UP' if (sd, sc) == tot['whole run'] else 'DOES NOT ADD UP'}")
    # missed-pair estimate
    d = {kk: [] for kk in ("orbit", "c_rho", "c_phi", "c_z")}
    for r in csv.DictReader(open(f"data/{name}.csv")):
        for kk in d: d[kk].append(float(r[kk]))
    d = {kk: np.array(v) for kk, v in d.items()}
    m = (d["orbit"] >= a.w0) & (d["orbit"] <= a.w1)
    sig2 = (np.mean(d["c_rho"][m] ** 2) + np.mean(d["c_phi"][m] ** 2) + np.mean(d["c_z"][m] ** 2)) / 3
    gbar = np.sqrt(8 / np.pi) * np.sqrt(2 * sig2)
    h, sr = cfg["time"]["step_s"], cfg["collisions"]["particle_radius_km"]
    miss = 2.09 * (gbar * h / (4 * sr)) ** 2
    print(f"{name:12s} missed-pair estimate window: <g> = {gbar * 1e6:.2f} mm/s, <g> h / 4 s = {gbar * h / (4 * sr):.4f}, "
          f"miss = {100 * miss:.3f} %")
    out.append(dict(run=name, group="miss_estimate_window", deep="", collisions="", deep_fraction=miss))
with open(f"data/{a.set}_deep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
