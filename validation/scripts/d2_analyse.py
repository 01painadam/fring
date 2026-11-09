"""D2 (role A6): analysis of the arc error test.

    uv run python scripts/d2_analyse.py t0 --runs ../runs/d2_t0        # t = 0 statistics a to d
    uv run python scripts/d2_analyse.py reduce --runs ../runs/d2       # frames -> reference/d2/stats_<core>.csv
    uv run python scripts/d2_analyse.py report                          # CSV -> reports/d2.md and reports/d2_arc_error.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fring_ref import arc_stats as A  # noqa: E402
from fring_ref.params import load_saturn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "d2"
A0 = 140222.43
E_ECC = 0.00235
SIGMA_A = 10.0
STAT_KEYS = ["sigma_rg", "pw_rg", "e_mean", "e_pw", "loc_rms", "loc_pw", "e_coh", "vdisp"]


def parse_name(name: str):
    """d2-<core>-ring-s<k> or d2-<core>-arc<WW>-c<CCC>"""
    _, core, dom, *rest = name.split("-")
    if dom == "ring":
        return core, A.Domain(360.0, 0.0), None
    return core, A.Domain(float(dom[3:]), float(rest[0][1:])), float(dom[3:])


def t0_table(runs: Path):
    sat = load_saturn()
    pred = A.predicted_slopes(sat, A0)
    rows = []
    for d in sorted(runs.iterdir()):
        core, dom, w = parse_name(d.name)
        files = A.frame_files(d)
        fr0 = A.read_frame(files[0])
        o0 = A.orbit_elements(sat, fr0)
        s = A.frame_stats(sat, fr0, dom, A0, o0)
        fr1 = A.read_frame(files[-1])
        o1 = A.orbit_elements(sat, fr1)
        s["steps"] = fr1.step
        if core == "ecc" and fr1.step > 0:
            s.update(A.apse_slope(sat, o0, o1, fr1, A0))
        rows.append((d.name, core, dom, s))
    return rows, pred


def cmd_t0(a):
    rows, pred = t0_table(Path(a.runs))
    ae = A0 * E_ECC
    print("t = 0 statistics (frame 0). km, km/s; coherence = |<e_vec>| / <e>")
    print(f"{'run':22s} {'sigma_rg':>9s} {'pw_rg':>8s} {'e_mean':>9s} {'e_pw':>9s} {'loc_rms':>9s} {'loc_pw':>9s} {'e_coh':>9s} {'coh':>6s} {'vdisp m/s':>10s}")
    for name, core, dom, s in rows:
        print(f"{name:22s} {s['sigma_rg']:9.3f} {s['pw_rg']:8.3f} {s['e_mean']:9.3e} {s['e_pw']:9.2e} "
              f"{s['loc_rms']:9.3f} {s['loc_pw']:9.3f} {s['e_coh']:9.3e} {s['e_coh']/s['e_mean']:6.3f} {1e3*s['vdisp']:10.3f}")
    print("\npredictions (ecc core): ae = %.1f km" % ae)
    print("  ring: loc_rms ~ sigma_a (+ 1 deg slope term %.1f km), coherence ~1" % (ae * np.deg2rad(1) / np.sqrt(12)))
    print("  arc : loc_rms ~ sqrt(sigma_a^2 + (ae)^2/2) = %.1f km, coherence ~ 0 (1/sqrt(N) = %.4f), vdisp ~ 0.79 A kappa = %.1f m/s" % (
        np.hypot(SIGMA_A, ae / np.sqrt(2)), 1 / np.sqrt(1e5), 0.79 * ae * float(A.kappa(load_saturn(), A0)) * 1e3))
    print("  apse slopes (deg/day/km): arc wrap %.3e, real differential precession %.3e, ratio %.1f" % (
        pred["wrap_slope"], pred["precession_slope"], pred["wrap_slope"] / pred["precession_slope"]))
    for name, core, dom, s in rows:
        if "apse_slope_deg_day_km" in s:
            print(f"  step {s['steps']} {name}: apse slope {s['apse_slope_deg_day_km']:.3e}, common {s['apse_common_deg_day']:.4f} deg/day")


def cmd_reduce(a):
    sat = load_saturn()
    runs = Path(a.runs)
    out = {"ecc": [], "circ": []}
    for d in sorted(runs.iterdir()):
        core, dom, w = parse_name(d.name)
        files = A.frame_files(d)
        fr0 = A.read_frame(files[0])
        o0 = A.orbit_elements(sat, fr0)
        for f in files:
            fr = A.read_frame(f)
            o = A.orbit_elements(sat, fr)
            s = A.frame_stats(sat, fr, dom, A0, o)
            if core == "ecc" and fr.step > 0:
                s.update(A.apse_slope(sat, o0, o, fr, A0))
            row = {"run": d.name, "width_deg": dom.width_deg, "centre_deg": dom.centre_deg,
                   "t_yr": fr.time_s / A.YEAR_S, "step": fr.step}
            row.update(s)
            out[core].append(row)
        print("reduced", d.name, flush=True)
    for core, rows in out.items():
        if not rows:
            continue
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(REF / f"stats_{core}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("t0", "reduce"):
        p = sp.add_parser(c)
        p.add_argument("--runs", required=True)
    a = ap.parse_args()
    {"t0": cmd_t0, "reduce": cmd_reduce}[a.cmd](a)
