"""E1-REF comparison: Rust Wisdom-Holman results against the IAS15 reference (ICR-004 section 6).

Usage (from validation/):
  uv run python scripts/e1_compare.py layer1 --on wh_on.csv [--off wh_off.csv]
  uv run python scripts/e1_compare.py layer2 --on wh_on.csv [--off wh_off.csv]

File contract (the WH side).  Both files use `#` comment lines and a numeric CSV body.
Layer 1: the states of every id of reference/e1_l1_states.csv at t0 + 2 days (t0_s + 172800 s), same frame and units:
  columns: id,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s
  The script computes the geometric elements (src/fring_ref/encounter.py) before (from the states file) and after,
  so IAS15 and WH go through the same function.
Layer 2: the file written for reference/e1_l2_states.csv; per id, the state at the end of the run and the closest approach:
  columns: id,min_prometheus_km,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s,megno_mean
  min_prometheus_km is the minimum distance to Prometheus over the run; megno_mean is the mean MEGNO Y of the run's chaos
  indicator (ADR 0008: d0 = 1 km, K = 100 steps; chaotic if >= 5, regular if <= 2.5).
`--on` is a run with sub-stepping on (limits apply), `--off` a run with sub-stepping off (power check).
Exit code 0 if every applicable limit is met.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fring_ref import encounter as enc  # noqa: E402
from fring_ref.params import load_saturn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"
SAT = load_saturn()
MEDIAN_LIMIT = 0.05  # AJ decision 3 (ICR-004)
P95_LIMIT = 0.25
NOISE_FLOOR_KM = 1e-9  # noise of the geometric a (tests/test_encounter.py: 1.5e-10 km over 2 d); 10 x this is reported


def load_ref1():
    ref = np.loadtxt(REF / "e1_l1_ref.csv", delimiter=",", comments="#")
    st = np.loadtxt(REF / "e1_l1_states.csv", delimiter=",", comments="#")
    return {"id": ref[:, 0].astype(int), "stratum": ref[:, 1].astype(int), "b_rh": ref[:, 2],
            "da": ref[:, 4], "de": ref[:, 5], "dv": ref[:, 6], "start": st[:, 4:10]}


def wh_elements1(ref, wh_file):
    d = np.loadtxt(wh_file, delimiter=",", comments="#")
    order = {int(i): k for k, i in enumerate(d[:, 0])}
    if set(order) != set(ref["id"].tolist()):
        raise SystemExit(f"{wh_file}: ids differ from reference/e1_l1_states.csv")
    after = np.array([d[order[int(i)], 1:7] for i in ref["id"]])
    return enc.delta_elements(SAT, ref["start"], after)


def layer1_stats(ref, da, de, dv):
    """Per stratum: median and p95 of |da_WH - da_ref| / |da_ref|, plus Delta e and apse differences."""
    rows = []
    for k, (lo, hi) in enumerate(enc.STRATA):
        s = ref["stratum"] == k
        rel = abs(da[s] - ref["da"][s]) / abs(ref["da"][s])
        rows.append({"stratum": k, "range_rh": [lo, hi], "n": int(s.sum()), "median": float(np.median(rel)),
                     "p95": float(np.percentile(rel, 95)), "max": float(rel.max()),
                     "n_ref_da_below_10_floor": int((abs(ref["da"][s]) < 10 * NOISE_FLOOR_KM).sum()),
                     "median_abs_de_diff": float(np.median(abs(de[s] - ref["de"][s]))),
                     "median_abs_dvarpi_diff_rad": float(np.median(abs(enc.wrap_pi(dv[s] - ref["dv"][s]))))})
    return rows


def print_rows(title, rows):
    print(title)
    print(f"  {'stratum':>12} {'n':>4} {'median':>9} {'p95':>9} {'max':>9} {'|dDe|med':>9} {'|dvarpi|med':>11} {'ref|da|<10floor':>15}")
    for r in rows:
        lo, hi = r["range_rh"]
        print(f"  [{lo},{hi}] {r['n']:>4} {r['median']:9.2e} {r['p95']:9.2e} {r['max']:9.2e} {r['median_abs_de_diff']:9.2e} "
              f"{r['median_abs_dvarpi_diff_rad']:11.2e} {r['n_ref_da_below_10_floor']:>15}")


def verdict_on(rows):
    """Limits with sub-stepping on: median < 5 % and p95 < 25 % for b >= 1 R_H (strata 1..3). Stratum 0 reported only."""
    ok = True
    for r in rows[1:]:
        good = r["median"] < MEDIAN_LIMIT and r["p95"] < P95_LIMIT
        ok &= good
        print(f"    stratum {r['stratum']}: median {r['median']:.3g} < {MEDIAN_LIMIT} and p95 {r['p95']:.3g} < {P95_LIMIT}: "
              f"{'pass' if good else 'FAIL'}")
    print("    stratum 0 [0.68,1) is reported, not a pass criterion")
    return ok


def power_off(rows):
    """Without sub-stepping the p95 must exceed 25 % in at least one stratum with b >= 1 R_H."""
    hit = [r["stratum"] for r in rows[1:] if r["p95"] > P95_LIMIT]
    if hit:
        print(f"    power check: p95 > {P95_LIMIT} in stratum(s) {hit}: the test has power")
    else:
        print("    power check: NO POWER: p95 <= 25 % in every stratum with b >= 1 R_H; sub-stepping is not needed for these strata")
    return bool(hit)


def run_layer1(on, off):
    ref = load_ref1()
    ok = True
    out = {}
    if on:
        rows = layer1_stats(ref, *wh_elements1(ref, on))
        print_rows(f"layer 1, sub-stepping ON ({on})", rows)
        ok &= verdict_on(rows)
        out["on"] = rows
    if off:
        rows = layer1_stats(ref, *wh_elements1(ref, off))
        print_rows(f"layer 1, sub-stepping OFF ({off})", rows)
        ok &= power_off(rows)
        out["off"] = rows
    print("layer 1:", "OK" if ok else "FAILED")
    return ok, out


# ------------------------------------------------------------------ layer 2 (statistics)

CLOSE_RH = (1.0, 3.0, 6.3)
CHAOTIC_Y = 5.0
REGULAR_Y = 2.5


def load_l2(f):
    d = np.loadtxt(f, delimiter=",", comments="#")
    d = d[np.argsort(d[:, 0])]
    return {"id": d[:, 0].astype(int), "min_km": d[:, 1], "final": d[:, 2:8], "megno": d[:, 8]}


def layer2_stats(ref, wh, start, r_hill, p_min=0.01):
    from scipy import stats

    n = len(ref["id"])
    if len(wh["id"]) != n:
        raise SystemExit("layer 2: the WH file and the reference have different numbers of particles")
    da_ref = enc.geo_elements(SAT, ref["final"])["a"] - enc.geo_elements(SAT, start)["a"]
    da_wh = enc.geo_elements(SAT, wh["final"])["a"] - enc.geo_elements(SAT, start)["a"]
    out = {"n": n, "counts": [], "limits": {}}
    ok = True
    for c in CLOSE_RH:
        n_ref = int((ref["min_km"] < c * r_hill).sum())
        n_wh = int((wh["min_km"] < c * r_hill).sum())
        lim = max(1.0, float(np.sqrt(n_ref)))
        good = abs(n_wh - n_ref) <= lim
        ok &= good
        out["counts"].append({"inside_rh": c, "n_ias15": n_ref, "n_wh": n_wh, "limit": lim, "pass": bool(good)})
    ks = stats.ks_2samp(da_wh, da_ref)
    out["ks_da"] = {"statistic": float(ks.statistic), "p": float(ks.pvalue), "pass": bool(ks.pvalue > p_min)}
    ok &= ks.pvalue > p_min
    f_ref = float((ref["megno"] >= CHAOTIC_Y).mean())
    f_wh = float((wh["megno"] >= CHAOTIC_Y).mean())
    sig = float(np.sqrt(max(f_ref * (1 - f_ref), 1.0 / n) / n))
    out["chaotic_fraction"] = {"ias15": f_ref, "wh": f_wh, "sigma": sig, "pass": bool(abs(f_wh - f_ref) <= 3 * sig)}
    ok &= abs(f_wh - f_ref) <= 3 * sig
    out["median_abs_da_km"] = {"ias15": float(np.median(abs(da_ref))), "wh": float(np.median(abs(da_wh)))}
    return ok, out


def print_l2(title, o):
    print(title)
    for c in o["counts"]:
        print(f"  closest approach < {c['inside_rh']} R_H: IAS15 {c['n_ias15']}, WH {c['n_wh']} (limit +-{c['limit']:.1f}) {'pass' if c['pass'] else 'FAIL'}")
    k = o["ks_da"]
    print(f"  KS of Delta a: D = {k['statistic']:.3f}, p = {k['p']:.3g} (> 0.01) {'pass' if k['pass'] else 'FAIL'}")
    c = o["chaotic_fraction"]
    print(f"  chaotic fraction (Y >= 5): IAS15 {c['ias15']:.3f}, WH {c['wh']:.3f}, 3 sigma = {3 * c['sigma']:.3f} {'pass' if c['pass'] else 'FAIL'}")
    print(f"  median |Delta a|: IAS15 {o['median_abs_da_km']['ias15']:.3g} km, WH {o['median_abs_da_km']['wh']:.3g} km")


def run_layer2(on, off):
    import json as _json

    ref = load_l2(REF / "e1_l2_ref.csv")
    start = np.loadtxt(REF / "e1_l2_states.csv", delimiter=",", comments="#")[:, 1:7]
    r_hill = _json.loads((REF / "e1_l2.json").read_text())["r_hill_km"]
    ok = True
    res = {}
    if on:
        g, o = layer2_stats(ref, load_l2(on), start, r_hill)
        print_l2(f"layer 2, sub-stepping ON ({on})", o)
        ok &= g
        res["on"] = o
    if off:
        _, o = layer2_stats(ref, load_l2(off), start, r_hill)
        print_l2(f"layer 2, sub-stepping OFF ({off}), reported only", o)
        res["off"] = o
    print("layer 2:", "OK" if ok else "FAILED")
    return ok, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", choices=["layer1", "layer2"])
    ap.add_argument("--on")
    ap.add_argument("--off")
    ap.add_argument("--json", help="write the numbers to this file")
    a = ap.parse_args()
    if a.layer == "layer1":
        ok, res = run_layer1(a.on, a.off)
    else:
        ok, res = run_layer2(a.on, a.off)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
    sys.exit(0 if ok else 1)
