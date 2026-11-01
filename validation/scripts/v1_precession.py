"""V1: REBOUND reference for single-particle precession (serves C2).

Run:  uv run python scripts/v1_precession.py
Writes validation/reference/v1_precession.json, v1_series_<r>.csv and
validation/reports/v1_precession.png.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rebound

from fring_ref.force import frequencies, harmonic_acceleration, make_rebound_force
from fring_ref.measure import event_rates, frequency_fit
from fring_ref.params import REPO, load_saturn

ROOT = Path(__file__).resolve().parents[1]
RADII = [100000.0, 140222.43, 180000.0]
E = I = 1e-3
N_ORBITS = 200
SAMPLES_PER_ORBIT = 100
EPS = 1e-9


def run(sat, r, theta0=0.0, E=E, I=I, n_orbits=N_ORBITS, spo=SAMPLES_PER_ORBIT):
    n, k, nu = frequencies(sat, r)
    c, s = np.cos(theta0), np.sin(theta0)
    er, et = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    pos = r * er
    vel = E * r * k * er + r * n * et + np.array([0, 0, I * r * nu])
    sim = rebound.Simulation()
    sim.G = 1.0  # masses are GM in km^3 s^-2
    sim.add(m=sat.gm)
    sim.add(m=0.0, x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2])
    sim.integrator = "ias15"
    sim.integrator.epsilon = EPS
    sim.force_is_velocity_dependent = 0
    sim.additional_forces = make_rebound_force(sat)
    period = 2 * np.pi / n
    ts = np.arange(n_orbits * spo + 1) * period / spo
    P = np.empty((len(ts), 3)); V = np.empty((len(ts), 3))
    t0 = time.time()
    for j, t in enumerate(ts):
        sim.integrate(t)
        p = sim.particles[1]
        P[j] = (p.x, p.y, p.z); V[j] = (p.vx, p.vy, p.vz)
    wall = time.time() - t0
    return ts, P, V, wall


def invariants(sat, P, V):
    """Lz and the Jacobi energy, both conserved in the static axisymmetric field."""
    from fring_ref.force import harmonic_potential
    r = np.linalg.norm(P, axis=1)
    lz = P[:, 0] * V[:, 1] - P[:, 1] * V[:, 0]
    e = 0.5 * np.einsum("ij,ij->i", V, V) - sat.gm / r + harmonic_potential(sat, P)
    return lz, e


def main():
    sat = load_saturn()
    results = []
    series = {}
    for r in RADII:
        n, k, nu = frequencies(sat, r)
        ts, P, V, wall = run(sat, r)
        ev = event_rates(ts, P, V)
        nf, kf, vf = frequency_fit(ts, P, V, n, k, nu)
        lz, en = invariants(sat, P, V)
        exp_varpi, exp_node = n - k, -(nu - n)
        res = dict(
            r_km=r, wall_s=wall, steps_samples=len(ts),
            n_rad_s=n, kappa_rad_s=k, nu_rad_s=nu,
            expected_varpi_dot_rad_s=exp_varpi, expected_node_dot_rad_s=exp_node,
            event=dict(
                varpi_dot_rad_s=ev["peri"]["rate"], varpi_sigma=ev["peri"]["sigma"],
                node_dot_rad_s=ev["node"]["rate"], node_sigma=ev["node"]["sigma"],
                varpi_rel_err=ev["peri"]["rate"] / exp_varpi - 1,
                node_rel_err=ev["node"]["rate"] / exp_node - 1,
                n_peri=ev["peri"]["n_events"], n_node=ev["node"]["n_events"]),
            freqfit=dict(
                n=nf, kappa=kf, nu=vf,
                varpi_dot_rad_s=nf - kf, node_dot_rad_s=-(vf - nf),
                varpi_rel_err=(nf - kf) / exp_varpi - 1,
                node_rel_err=-(vf - nf) / exp_node - 1,
                n_rel_err=nf / n - 1, kappa_rel_err=kf / k - 1, nu_rel_err=vf / nu - 1),
            lz_rel_drift=float(np.max(np.abs(lz / lz[0] - 1))),
            energy_rel_drift=float(np.max(np.abs(en / en[0] - 1))),
        )
        results.append(res)
        series[r] = (ts, P, V, ev)
        print(f"r={r:.2f} km  wall {wall:.1f}s  event: varpi {res['event']['varpi_rel_err']:+.2e} "
              f"node {res['event']['node_rel_err']:+.2e} | freq fit: varpi {res['freqfit']['varpi_rel_err']:+.2e} "
              f"node {res['freqfit']['node_rel_err']:+.2e} | dLz {res['lz_rel_drift']:.1e} dE {res['energy_rel_drift']:.1e}")
        # series file
        path = ROOT / "reference" / f"v1_series_r{int(round(r))}.csv"
        hdr = (f"V1 REBOUND series; r_guiding={r} km, e=i=1e-3, theta0=0; Saturn-centred inertial; "
               f"columns t_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s; sampled once per orbit/{SAMPLES_PER_ORBIT}")
        sel = slice(None, None, 10)  # every tenth sample (10 per orbit) to keep the file small
        np.savetxt(path, np.column_stack([ts, P, V])[sel], delimiter=",", fmt="%.17g",
                   header=hdr, comments="# ")
    # sensitivity checks at the core radius: amplitude scaling and sampling density
    r = RADII[1]
    n, k, nu = frequencies(sat, r)
    checks = []
    for label, kw in (("e=i=1e-4", dict(E=1e-4, I=1e-4)), ("e=i=1e-3, 50 samples/orbit", dict(spo=50)),
                      ("e=i=1e-3, theta0=1.0 rad", dict(theta0=1.0))):
        kw = dict(n_orbits=100, **kw)
        ts, P, V, wall = run(sat, r, **kw)
        ev = event_rates(ts, P, V)
        c = dict(label=label, n_orbits=100,
                 varpi_rel_err=ev["peri"]["rate"] / (n - k) - 1, node_rel_err=ev["node"]["rate"] / (-(nu - n)) - 1)
        checks.append(c)
        print("check", c)
    meta = dict(
        task="V1", frame="Saturn-centred, Saturn equator inertial; Saturn at origin",
        units="km, s, rad",
        versions=dict(rebound=rebound.__version__, numpy=np.__version__, python=sys.version.split()[0]),
        integrator=dict(name="ias15", epsilon=EPS, note="Saturn is a free body but a massless particle does not move it"),
        field=dict(gm_km3_s2=sat.gm, radius_km=sat.radius, j2=sat.j2, j4=sat.j4, j6=sat.j6),
        start="guiding centre at r, z=0, theta0=0; v_r=e r kappa, v_t=r n, v_z=i r nu; e=i=1e-3",
        n_orbits=N_ORBITS, samples_per_orbit=SAMPLES_PER_ORBIT,
        methods=dict(
            event="slope of azimuth of pericentre passages (rdot -,+ zero) and ascending nodes (z -,+ zero) vs time; polynomial root finding on 8 samples",
            freqfit="least-squares fit of n (azimuth), kappa (r), nu (z); varpi_dot=n-kappa, node_dot=-(nu-n)"),
        sensitivity_checks=checks,
        results=results,
    )
    (ROOT / "reference" / "v1_precession.json").write_text(json.dumps(meta, indent=2) + "\n")
    make_plot(sat, series, results)


def make_plot(sat, series, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    x = np.arange(len(results))
    w = 0.2
    labs = [f"{r['r_km']:.0f}" for r in results]
    for j, (key, lab) in enumerate([("varpi_rel_err", "event varpi"), ("node_rel_err", "event node")]):
        ax[0].bar(x + (j - 1.5) * w, [abs(r["event"][key]) + 1e-16 for r in results], w, label=lab)
    for j, (key, lab) in enumerate([("varpi_rel_err", "fit n-kappa"), ("node_rel_err", "fit -(nu-n)")]):
        ax[0].bar(x + (j + 0.5) * w, [abs(r["freqfit"][key]) + 1e-16 for r in results], w, label=lab)
    ax[0].axhline(1e-3, color="r", ls="--", label="PLAN tolerance 0.1 %")
    ax[0].axhline(1e-4, color="gray", ls=":", label="A6 target 0.01 %")
    ax[0].set_yscale("log"); ax[0].set_xticks(x, labs); ax[0].set_xlabel("guiding radius (km)")
    ax[0].set_ylabel("|measured / PLAN 7.2 - 1|"); ax[0].legend(fontsize=7); ax[0].set_title("Rate error vs PLAN 7.2")
    for r in series:
        ts, P, V, ev = series[r]
        d = ev["peri"]
        ax[1].plot((d["t"] - d["t"][0]) / 86400, d["lon"] - d["lon"][0], label=f"{r:.0f} km")
        d = ev["node"]
        ax[2].plot((d["t"] - d["t"][0]) / 86400, d["lon"] - d["lon"][0], label=f"{r:.0f} km")
    ax[1].set_title("Pericentre longitude (REBOUND, IAS15)"); ax[2].set_title("Ascending node longitude")
    for a in ax[1:]:
        a.set_xlabel("time (days)"); a.set_ylabel("rad, unwrapped"); a.legend(fontsize=8)
    fig.suptitle("V1: single-particle precession, J2+J4+J6, e = i = 1e-3, 200 orbits")
    out = ROOT / "reports" / "v1_precession.png"
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
