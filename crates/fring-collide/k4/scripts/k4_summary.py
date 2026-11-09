"""K4: summary of the reduced time series data/<set>-{st,hs}-s<k>.csv (from k4_reduce.py).

Usage: uv run python scripts/k4_summary.py <set> [--w0 80] [--w1 200] [--tol 0.1]
Writes data/<set>_perseed.csv, data/<set>_summary.csv, data/<set>_relax.csv and figures/<set>_overview.png.

Window. Frames with w0 - tol <= orbit <= w1 + tol. Frames are 1782 steps = 2.00058 orbits apart, so
orbit 100 of the run is at 200.058 and the frame at 80.023 is the first one: 61 frames, tol = 0.1 orbit.
The final frame (200.11 orbits) is outside. Collisions are summed over the intervals AFTER the first window
frame (about 80.02 to 200.06, 120.03 orbits).
Per run (one seed, one model):
- c_rho, c_phi, c_z: mean over the window frames, mm/s.
- coll/particle/orbit = 2 * (accepted pairs) / N / (orbits): each accepted pair is one collision of two particles
  (SPEC section 9: "collisions (accepted pairs)"), so every particle takes part in 2 * pairs / N collisions.
- nu = slope of sigma2 (variance of rho minus c_rho^2 / kappa^2) against t over the window frames / 6, m2/s.
  nu_se_ols is the OLS standard error of that slope. Frames are correlated, so it is a cross-check only.
- nu_E = (4/9) * (sum of energy_loss_km2_s2 per particle per s) / Omega^2, m2/s, same collision window.
Across seeds (n = 4): mean, SE = std(ddof=1) / sqrt(n), 95 % interval = t(0.975, n-1) * SE (t = 3.182 for 3 dof).
HS/ST ratio:
- paired (headline): r_k = HS_k / ST_k for each seed k (same seed = identical initial state, see k4_initcheck.py);
  mean of r_k, SE = std(r_k, ddof=1) / sqrt(n), 95 % interval = t * SE.
- unpaired (comparison): mean_HS / mean_ST with SE = ratio * hypot(se_HS / mean_HS, se_ST / mean_ST).
Relaxation (per run, window frames, no pass/fail): slope of c_rho against orbit with OLS SE, relative drift in % per
100 orbits, means of the first and second half of the window, and the window means of c_phi/c_rho and c_z/c_rho.
"""
import glob, csv, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("set"); ap.add_argument("--w0", type=float, default=80); ap.add_argument("--w1", type=float, default=200)
ap.add_argument("--tol", type=float, default=0.1)
a = ap.parse_args()
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

def load(p):
    r = list(csv.DictReader(open(p)))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}

def ols(x, y):
    """slope and its standard error"""
    xm = x - x.mean(); sxx = np.dot(xm, xm)
    slope = np.dot(xm, y - y.mean()) / sxx
    res = (y - y.mean()) - slope * xm
    return slope, np.sqrt(np.sum(res ** 2) / (len(x) - 2) / sxx)

def one(path):
    d = load(path); o = d["orbit"]; t = d["t_s"]; n = d["n"][0]; om = d["n0_rad_s"][0]
    m = (o >= a.w0 - a.tol) & (o <= a.w1 + a.tol)
    i = np.flatnonzero(m); i0, i1 = i[0], i[-1]
    mc = np.zeros_like(m); mc[i0 + 1:i1 + 1] = True       # collision intervals after the first window frame
    orbits = o[i1] - o[i0]
    cr, cp, cz = (d[k][m] for k in ("c_rho", "c_phi", "c_z"))
    nu_s, nu_se = ols(t[m], d["sigma2_km2"][m])
    half = m.sum() // 2
    cr_slope, cr_se = ols(o[m], cr)
    P = 2 * np.pi / om
    r = dict(name=path, frames=int(m.sum()), orbits=orbits,
             c_rho=cr.mean() * 1e6, c_phi=cp.mean() * 1e6, c_z=cz.mean() * 1e6,
             coll_per_part_orbit=2 * d["coll"][mc].sum() / n / orbits,
             nu=nu_s / 6 * 1e6, nu_se_ols=nu_se / 6 * 1e6,
             nu_E=(4 / 9) * d["loss"][mc].sum() / n / (orbits * P) / om ** 2 * 1e6,
             deep_window=int(d["deep"][mc].sum()), gmax_exceeded_window=int(d["gmax_exceeded"][mc].sum()),
             crho_slope_mm_s_per100orb=cr_slope * 1e6 * 100, crho_slope_se=cr_se * 1e6 * 100,
             crho_drift_pct_per100orb=100 * cr_slope * 100 / cr.mean(),
             crho_first_half=cr[:half].mean() * 1e6, crho_second_half=cr[-half:].mean() * 1e6,
             cphi_over_crho=(cp / cr).mean(), cz_over_crho=(cz / cr).mean(),
             cphi_over_crho_first=(cp / cr)[:half].mean(), cphi_over_crho_second=(cp / cr)[-half:].mean(),
             cz_over_crho_first=(cz / cr)[:half].mean(), cz_over_crho_second=(cz / cr)[-half:].mean())
    return r, d

R = {}; D = {}
for mod in ("st", "hs"):
    R[mod] = []; D[mod] = []
    for p in sorted(glob.glob(f"data/{a.set}-{mod}-s[0-9].csv")):
        r, d = one(p); R[mod].append(r); D[mod].append(d)
ns = len(R["st"]); assert ns == len(R["hs"]) and ns > 1
t = T975[ns - 1]
print(f"{a.set}: window orbits {a.w0:g}-{a.w1:g} (tol {a.tol}), {R['st'][0]['frames']} frames, {R['st'][0]['orbits']:.2f} orbits, seeds={ns}, t(0.975,{ns - 1} dof)={t}")

pers = ["c_rho", "c_phi", "c_z", "coll_per_part_orbit", "nu", "nu_se_ols", "nu_E", "deep_window", "gmax_exceeded_window"]
print("\nPer seed (c in mm/s, nu in m2/s; coll = 2 * accepted pairs / N / orbit)")
print(f"{'run':14s}" + "".join(f"{k:>14s}" for k in pers))
rows = []
for mod in R:
    for r in R[mod]:
        nm = r["name"].split("/")[-1][:-4]
        print(f"{nm:14s}" + "".join(f"{r[k]:14.5g}" for k in pers))
        rows.append(dict(run=nm, **{k: r[k] for k in pers}))
with open(f"data/{a.set}_perseed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

keys = [("c_rho", "c_rho (mm/s)"), ("c_phi", "c_phi (mm/s)"), ("c_z", "c_z (mm/s)"),
        ("coll_per_part_orbit", "coll/particle/orbit"), ("nu", "nu (m2/s)"), ("nu_E", "nu_E (m2/s)"),
        ("cphi_over_crho", "c_phi/c_rho"), ("cz_over_crho", "c_z/c_rho")]
def ms(v): return v.mean(), v.std(ddof=1) / np.sqrt(len(v))
print(f"\nMeans over seeds: mean +- SE [95 % interval half width = t*SE]")
print(f"{'stat':22s} {'ST':>26s} {'HS':>26s} {'paired HS/ST':>28s} {'unpaired HS/ST':>20s}")
summ = []
for k, lab in keys:
    st = np.array([r[k] for r in R["st"]]); hs = np.array([r[k] for r in R["hs"]])
    (ms_, se_s), (mh, se_h) = ms(st), ms(hs)
    rk = hs / st; mr, se_r = ms(rk)
    ur = mh / ms_; use = ur * np.hypot(se_h / mh, se_s / ms_)
    print(f"{lab:22s} {ms_:10.5g}+-{se_s:8.2g}[{t * se_s:7.2g}] {mh:10.5g}+-{se_h:8.2g}[{t * se_h:7.2g}] "
          f"{mr:10.5f}+-{se_r:8.5f}[{t * se_r:8.5f}] {ur:9.5f}+-{use:8.5f}")
    summ.append(dict(stat=k, st=ms_, st_se=se_s, st_ci95=t * se_s, hs=mh, hs_se=se_h, hs_ci95=t * se_h,
                     paired_ratio=mr, paired_ratio_se=se_r, paired_ratio_ci95=t * se_r,
                     unpaired_ratio=ur, unpaired_ratio_se=use, **{f"ratio_seed{i + 1}": x for i, x in enumerate(rk)}))
with open(f"data/{a.set}_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

rk_ = ["frames", "crho_slope_mm_s_per100orb", "crho_slope_se", "crho_drift_pct_per100orb", "crho_first_half", "crho_second_half",
       "cphi_over_crho", "cphi_over_crho_first", "cphi_over_crho_second", "cz_over_crho", "cz_over_crho_first", "cz_over_crho_second"]
print("\nRelaxation (window frames; slope of c_rho in mm/s per 100 orbits +- OLS SE; halves = first / second half of the window)")
print(f"{'run':14s}" + "".join(f"{k[:13]:>14s}" for k in rk_))
rel = []
for mod in R:
    for r in R[mod]:
        nm = r["name"].split("/")[-1][:-4]
        print(f"{nm:14s}" + "".join(f"{r[k]:14.5g}" for k in rk_))
        rel.append(dict(run=nm, **{k: r[k] for k in rk_}))
with open(f"data/{a.set}_relax.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rel[0].keys())); w.writeheader(); w.writerows(rel)

col = {"st": "tab:blue", "hs": "tab:red"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for mod in D:
    for i, d in enumerate(D[mod]):
        for k, ls in (("c_rho", "-"), ("c_phi", ":"), ("c_z", "--")):
            ax[0].plot(d["orbit"], d[k] * 1e6, ls, color=col[mod], alpha=.6, label=f"{mod} {k}" if i == 0 else None)
        ax[1].plot(d["orbit"][1:], d["c_phi"][1:] / d["c_rho"][1:], ":", color=col[mod], alpha=.6, label=f"{mod} c_phi/c_rho" if i == 0 else None)
        ax[1].plot(d["orbit"][1:], d["c_z"][1:] / d["c_rho"][1:], "--", color=col[mod], alpha=.6, label=f"{mod} c_z/c_rho" if i == 0 else None)
for x in ax: x.axvspan(a.w0, a.w1, color="grey", alpha=.1); x.set_xlabel("orbits")
ax[0].set_ylabel("c (mm/s)"); ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"figures/{a.set}_overview.png", dpi=120)
