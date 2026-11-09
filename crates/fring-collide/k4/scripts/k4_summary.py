"""K4: summary of reduced time series (data/<set>-{st,hs}-s<k>.csv).

Usage: uv run python scripts/k4_summary.py <set> [--tau-peak 0.105] [--width-deg 0.0163] [--w0 80] [--w1 200]
set is the file prefix, for example t01. Prints a table and writes figures/<set>_*.png and data/<set>_summary.csv.
Definitions: see REPORT.md section 3.
"""
import sys, glob, csv, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("set"); ap.add_argument("--width-deg", type=float, default=0.0163)
ap.add_argument("--w0", type=float, default=80); ap.add_argument("--w1", type=float, default=200)
ap.add_argument("--win", type=float, default=40)   # sliding window for nu_eff, orbits
a = ap.parse_args()
OM = 1.1756e-4; A0 = 140220.0; S = 0.01; N = 1e5; H = 60.0
D = A0 * np.radians(a.width_deg)

def load(p):
    r = list(csv.DictReader(open(p)))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}

def stats(model):
    out = []
    for p in sorted(glob.glob(f"data/{a.set}-{model}-s*.csv")):
        d = load(p); o = d["orbit"]
        m = (o >= a.w0) & (o <= a.w1)
        t = d["t_s"]
        slope = np.polyfit(t[m], d["sigma2_km2"][m], 1)[0]             # km2/s
        nu = slope / 6 * 1e6                                           # m2/s
        # equilibrium averages over the window
        cr, cp, cz = (d[k][m].mean() * 1e6 for k in ("c_rho", "c_phi", "c_z"))   # mm/s
        mm = (o > a.w0) & (o <= a.w1)
        orb = (t[mm][-1] - t[mm][0] + (t[1] - t[0])) / (2 * np.pi / OM)
        coll = d["coll"][mm].sum(); loss = d["loss"][mm].sum()
        cpo = 2 * coll / N / orb
        nuE = (4 / 9) * loss / N / (orb * 2 * np.pi / OM) / OM ** 2 * 1e6   # m2/s
        s2 = d["sigma2_km2"]; tau_eff = N * np.pi * S ** 2 / (D * np.sqrt(4 * np.pi) * np.sqrt(s2))
        # sliding-window nu_eff
        te, ne = [], []
        for i in range(len(o)):
            w = (o >= o[i] - a.win / 2) & (o <= o[i] + a.win / 2)
            if w.sum() >= 8 and o[i] - a.win / 2 >= 0 and o[i] + a.win / 2 <= o[-1]:
                te.append(tau_eff[i]); ne.append(np.polyfit(t[w], s2[w], 1)[0] / 6 * 1e6)
        out.append(dict(name=p, nu=nu, cr=cr, cp=cp, cz=cz, cpo=cpo, nuE=nuE, deep=d["deep"].sum(),
                        gx=d["gmax_exceeded"].sum(), tau_w=tau_eff[mm].mean(), te=np.array(te), ne=np.array(ne), d=d))
    return out

R = {m: stats(m) for m in ("st", "hs")}
keys = [("nu", "nu (m2/s)"), ("cr", "c_rho (mm/s)"), ("cp", "c_phi (mm/s)"), ("cz", "c_z (mm/s)"),
        ("cpo", "coll/particle/orbit"), ("nuE", "nu_E (m2/s)")]
rows = []
print(f"{a.set}: window orbits {a.w0}-{a.w1}, seeds st={len(R['st'])} hs={len(R['hs'])}")
print(f"{'stat':22s} {'stoch':>18s} {'hard':>18s} {'ratio HS/ST':>18s}")
for k, lab in keys:
    v = {m: np.array([r[k] for r in R[m]]) for m in R}
    mean = {m: v[m].mean() for m in v}
    se = {m: v[m].std(ddof=1) / np.sqrt(len(v[m])) if len(v[m]) > 1 else float("nan") for m in v}
    ratio = mean["hs"] / mean["st"]
    rse = ratio * np.hypot(se["hs"] / mean["hs"], se["st"] / mean["st"])
    print(f"{lab:22s} {mean['st']:10.4g}+-{se['st']:7.2g} {mean['hs']:10.4g}+-{se['hs']:7.2g} {ratio:9.4f}+-{rse:7.4f}")
    rows.append(dict(stat=k, st=mean["st"], st_se=se["st"], hs=mean["hs"], hs_se=se["hs"], ratio=ratio, ratio_se=rse))
for m in R:
    print(m, "deep", sum(r["deep"] for r in R[m]), "gmax_exceeded", sum(r["gx"] for r in R[m]),
          "mean tau_eff in window", np.mean([r["tau_w"] for r in R[m]]))
with open(f"data/{a.set}_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

col = {"st": "tab:blue", "hs": "tab:red"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for m in R:
    for i, r in enumerate(R[m]):
        d = r["d"]
        for k, ls in (("c_rho", "-"), ("c_phi", ":"), ("c_z", "--")):
            ax[0].plot(d["orbit"], d[k] * 1e6, ls, color=col[m], alpha=.6, label=f"{m} {k}" if i == 0 else None)
        ax[1].plot(r["te"], r["ne"], color=col[m], alpha=.6, label=m if i == 0 else None)
ax[0].set_xlabel("orbits"); ax[0].set_ylabel("c (mm/s)"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("tau_eff (surface-density weighted)"); ax[1].set_ylabel("nu_eff (m2/s)"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"figures/{a.set}_overview.png", dpi=120)
