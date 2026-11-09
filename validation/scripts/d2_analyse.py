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
A_WIDTHS = (5.0, 10.0, 30.0)
A_CENTRES = (0.0, 90.0, 180.0, 270.0)
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
                   "t_yr": fr.time_s / A.YEAR_S, "step": fr.step, "win_w": "", "win_c": ""}
            row.update(s)
            out[core].append(row)
            if dom.ring:  # the ring inside each arc window (same longitudes, same moon phase)
                for w in A_WIDTHS:
                    for c in A_CENTRES:
                        wd = A.Domain(w, c)
                        sel = A.window_mask(wd, o.theta)
                        sw = A.frame_stats(sat, fr, wd, A0, o, sel)
                        if core == "ecc" and fr.step > 0:
                            sw.update(A.apse_slope(sat, o0, o, fr, A0, sel))
                        r2 = {"run": d.name, "width_deg": 360.0, "centre_deg": 0.0,
                              "t_yr": fr.time_s / A.YEAR_S, "step": fr.step, "win_w": w, "win_c": c}
                        r2.update(sw)
                        out[core].append(r2)
        print("reduced", d.name, flush=True)
    for core, rows in out.items():
        if not rows:
            continue
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(REF / f"stats_{core}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows)


LABELS = {
    "sigma_rg": "core width, RMS of r_g",
    "pw_rg": "core width, (P84-P16)/2 of r_g",
    "e_mean": "mean e per particle",
    "e_pw": "spread of e, (P84-P16)/2",
    "loc_rms": "(a) local radial RMS, 1 deg bins",
    "loc_pw": "(a) local radial (P84-P16)/2",
    "e_coh": "(b) coherent |<e_vec>|",
    "vdisp": "(c) local velocity dispersion",
    "apse_slope_deg_day_km": "(d) apse drift slope vs r_g",
}
TIMES = [1.0, 2.0, 3.0, 4.0, 5.0]
SLOPE_TIMES = [0.25, 0.5, 1.0]
LIMIT = 0.05


def load_csv(core):
    path = REF / f"stats_{core}.csv"
    if not path.exists():
        return []
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({k: (float(v) if v not in ("", None) and k != "run" else v) for k, v in r.items()})
    seen, uniq = set(), []
    for r in rows:  # the last frame is written twice (a frame at t = 5 yr and a final one)
        key = (r["run"], round(r["t_yr"] * 4), r["win_w"], r["win_c"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def at(rows, t, **sel):
    out = []
    for r in rows:
        if abs(r["t_yr"] - t) < 0.02 and all(r.get(k) == v for k, v in sel.items()):
            out.append(r)
    return out


def ring_global(rows, t):
    return [r for r in at(rows, t, width_deg=360.0) if r["win_w"] == ""]


def ring_window(rows, t, w, c):
    return [r for r in at(rows, t, win_w=w, win_c=c)]


def arc_rows(rows, t, w):
    return sorted(at(rows, t, width_deg=w), key=lambda r: r["centre_deg"])


def evaluate(core, mode="win"):
    """-> {width: {stat: {t: compare-dict}}}, the t = 0 table and pass flags."""
    rows = load_csv(core)
    res, flags = {}, {}
    widths = sorted({r["width_deg"] for r in rows if r["width_deg"] < 360})
    for w in widths:
        res[w], flags[w] = {}, {}
        for key in STAT_KEYS + ["apse_slope_deg_day_km"]:
            if core == "circ" and key == "apse_slope_deg_day_km":
                continue
            times = SLOPE_TIMES if key.startswith("apse") else TIMES
            res[w][key] = {}
            ok = True
            for t in times:
                arcr = [r for r in arc_rows(rows, t, w) if key in r]
                arcs = [r[key] for r in arcr]
                if mode == "win":
                    pairs = [[x[key] for x in ring_window(rows, t, w, r["centre_deg"]) if key in x] for r in arcr]
                    if len(arcs) < 2 or any(len(p) < 2 for p in pairs):
                        continue
                    c = A.compare_paired(pairs, arcs)
                else:
                    ring = [r[key] for r in ring_global(rows, t) if key in r]
                    if len(ring) < 2 or len(arcs) < 2:
                        continue
                    c = A.compare(ring, arcs)
                res[w][key][t] = c
            # the rule: coherent e of the e = 0 control only counts when the ring value is
            # above 3 times the random-phase floor <e>/sqrt(N)
            applies = True
            if key == "e_coh" and res[w][key]:
                t1 = min(res[w][key])
                em = np.mean([r["e_mean"] for r in ring_global(rows, t1)])
                applies = res[w][key][t1]["ring"] > 3 * em / np.sqrt(1e5)
            bounds = [c["bound"] for c in res[w][key].values()]
            flags[w][key] = (A.passes(bounds, LIMIT) if bounds else False) if applies else True
            res[w][key]["applies"] = applies
    return res, flags, rows


MODE_TEXT = {
    "win": "Primary comparison: the arc against the ring in the SAME longitude window and the same moon phase "
           "(ring particles whose rotating-frame longitude is inside the arc at that time). This isolates the "
           "error of the wrap and of the fold; d_c = arc_c / ring_window_c - 1, paired by centre.",
    "glob": "Practical comparison: the arc against the whole ring (all longitudes, mean of 2 seeds). It adds the "
            "effect that one arc samples one longitude, and so one moon phase, at a time.",
}


def report_mode(mode, out, recs, axes):
    for ci, core in enumerate(("circ", "ecc")):
        res, flags, rows = evaluate(core, mode)
        if not res:
            continue
        recs[(mode, core)] = A.recommend(flags)
        title = "e = 0 core" if core == "circ" else "eccentric core (e = 0.00235)"
        out += [f"## {title}, {'ring window' if mode == 'win' else 'whole ring'}", ""]
        ws = sorted(res)
        out.append("Bound (percent) at the worst sample time, per statistic and width (|d| = worst mean difference).\n")
        out.append("| statistic | " + " | ".join(f"{w:g} deg" for w in ws) + " |")
        out.append("|---|" + "---|" * len(ws))
        for key in STAT_KEYS + ["apse_slope_deg_day_km"]:
            if core == "circ" and key.startswith("apse"):
                continue
            cells = []
            for w in ws:
                cs = {t: c for t, c in res[w][key].items() if t != "applies"}
                if not cs:
                    cells.append("n/a")
                    continue
                b = max(c["bound"] for c in cs.values())
                dd = max(abs(c["d"]) for c in cs.values())
                tag = "PASS" if flags[w][key] else "FAIL"
                if not res[w][key]["applies"]:
                    tag = "n/a (ring value at noise floor)"
                cells.append(f"{100*b:.3g} % (|d| {100*dd:.3g} %) {tag}")
            out.append(f"| {LABELS[key]} | " + " | ".join(cells) + " |")
        out.append("")
        out.append(f"Recommended smallest width: **{recs[(mode, core)] if recs[(mode, core)] else 'none'}**.\n")
        out.append("d (percent) per time, mean over centres:\n")
        out.append("| statistic | width | " + " | ".join(f"{t:g} yr" for t in TIMES) + " |")
        out.append("|---|---|" + "---|" * len(TIMES))
        for key in STAT_KEYS:
            for w in ws:
                cs = res[w][key]
                out.append(f"| {key} | {w:g} | " + " | ".join(
                    f"{100*cs[t]['d']:.3g}" if t in cs else "" for t in TIMES) + " |")
        out.append("")
        if mode == "win":
            out.append("t = 0 (frame 0), mean over centres:\n")
            out.append("| domain | loc_rms km | coherence | vdisp m/s | sigma_rg km | e_mean |")
            out.append("|---|---|---|---|---|---|")
            for w in [360.0] + ws:
                rr = ring_global(rows, 0.0) if w == 360.0 else arc_rows(rows, 0.0, w)
                if not rr:
                    continue
                m = lambda k: np.mean([r[k] for r in rr])
                out.append(f"| {'ring' if w == 360 else f'arc {w:g}'} | {m('loc_rms'):.3f} | {m('e_coh')/m('e_mean'):.4f} | "
                           f"{1e3*m('vdisp'):.3f} | {m('sigma_rg'):.3f} | {m('e_mean'):.4e} |")
            out.append("")
            out.append("Absolute values at 1, 3, 5 yr (ring = whole ring, mean of seeds; arc = mean of centres):\n")
            out.append("| statistic | t | ring | arc 5 | arc 10 | arc 30 |")
            out.append("|---|---|---|---|---|---|")
            for key in STAT_KEYS:
                for t in (1.0, 3.0, 5.0):
                    rg = [r[key] for r in ring_global(rows, t)]
                    vals = [np.mean([r[key] for r in arc_rows(rows, t, w)]) for w in ws]
                    out.append(f"| {key} | {t:g} | {np.mean(rg):.4g} | " + " | ".join(f"{v:.4g}" for v in vals) + " |")
            out.append("")
        if axes is not None:
            for ri, key in enumerate(STAT_KEYS + ["apse_slope_deg_day_km"]):
                ax = axes[ri][ci]
                for w in ws:
                    cs = {t: c for t, c in res[w].get(key, {}).items() if t != "applies"}
                    if cs:
                        ts = sorted(cs)
                        ax.errorbar(ts, [100 * cs[t]["d"] for t in ts],
                                    yerr=[200 * np.hypot(cs[t]["sem_arc"], cs[t]["noise_ring"]) for t in ts],
                                    marker="o", label=f"{w:g} deg")
                ax.axhspan(-5, 5, color="0.9")
                ax.set_ylabel("d [%]")
                ax.set_title(f"{title}: {LABELS[key]}", fontsize=8)
                if key in ("e_coh", "apse_slope_deg_day_km", "vdisp", "loc_rms", "loc_pw"):
                    ax.set_yscale("symlog", linthresh=10)
                ax.legend(fontsize=6)


def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ["# D2 tables: arc against full ring", ""]
    out.append("Auto-generated by `validation/scripts/d2_analyse.py report`. Rule: a statistic passes at a width when "
               "`|mean d| + 2 sqrt(sem^2 + noise^2) < 5 %` at every sample time "
               "(t = 1 to 5 yr; t = 0.25, 0.5, 1 yr for the apse slope). Mean over 4 arc centres (0, 90, 180, 270 deg); "
               "ring: 2 seeds. N = 1e5 per run.\n")
    recs = {}
    fig, axes = plt.subplots(len(STAT_KEYS) + 1, 2, figsize=(11, 22), sharex="col")
    for mode in ("win", "glob"):
        out.append(f"# {'A. Ring window (primary)' if mode == 'win' else 'B. Whole ring'}\n")
        out.append(MODE_TEXT[mode] + "\n")
        report_mode(mode, out, recs, axes if mode == "win" else None)
    axes[-1][0].set_xlabel("t [yr]")
    axes[-1][1].set_xlabel("t [yr]")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "d2_arc_error.png", dpi=110)
    (ROOT / "reports" / "d2_tables.md").write_text("\n".join(out) + "\n")
    print("recommendations:", recs)


def cmd_cpu(a):
    """CPU cross-check (N = 2e4, ring and 10 deg arc, centre 0): CPU statistics against the GPU runs."""
    sat = load_saturn()
    runs = Path(a.runs)
    lines = ["# D2 CPU cross-check", "",
             "CPU backend (N = 2e4, f64) against the GPU backend (N = 1e5, D+) for the same configs "
             "(ring: seed differs; arc: 10 deg, centre 0). Relative difference CPU / GPU - 1 in percent. "
             "Expected noise: 1/sqrt(2N) = 0.5 % for a width at N = 2e4 (local statistics in a 10 deg window: larger).", ""]
    for core in ("circ", "ecc"):
        gpu = load_csv(core)
        lines += [f"## {core}", "", "| domain | statistic | t [yr] | CPU | GPU | CPU/GPU - 1 [%] |", "|---|---|---|---|---|---|"]
        for tag, dom in (("ring", A.Domain(360.0)), ("arc10", A.Domain(10.0, 0.0))):
            d = runs / (f"d2cpu-{core}-ring-s0" if tag == "ring" else f"d2cpu-{core}-arc10-c000")
            if not d.exists() or not (d / "meta.json").exists():
                continue
            files = A.frame_files(d)
            for t_want in (0.0, 1.0, 3.0, 5.0):
                f = min(files, key=lambda q: abs(A.read_frame(q).time_s / A.YEAR_S - t_want))
                fr = A.read_frame(f)
                st = A.frame_stats(sat, fr, dom, A0)
                if tag == "ring":
                    g = [r for r in ring_global(gpu, t_want)]
                else:
                    g = [r for r in arc_rows(gpu, t_want, 10.0) if r["centre_deg"] == 0.0]
                for key in STAT_KEYS:
                    gv = float(np.mean([r[key] for r in g]))
                    lines.append(f"| {tag} | {key} | {t_want:g} | {st[key]:.5g} | {gv:.5g} | {100*(st[key]/gv-1):.3g} |")
    (ROOT / "reports" / "d2_cpu.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("t0", "reduce"):
        p = sp.add_parser(c)
        p.add_argument("--runs", required=True)
    sp.add_parser("report")
    pc = sp.add_parser("cpu")
    pc.add_argument("--runs", required=True)
    a = ap.parse_args()
    {"t0": cmd_t0, "reduce": cmd_reduce, "report": cmd_report, "cpu": cmd_cpu}[a.cmd](a)
