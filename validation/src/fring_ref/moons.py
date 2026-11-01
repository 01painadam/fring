"""Prometheus and Pandora with Saturn as a free massive body (ADR 0004).

REBOUND units: G = 1, masses are GM in km^3 s^-2, lengths km, time s.
Saturn starts at the origin at rest and the moons start at their Saturn-centred states;
because the equations are Galilean invariant, the Saturn-centred motion is the same for
any Saturn velocity.  The J2, J4, J6 force acts on the moons relative to Saturn.  The
reaction of Saturn to that force (relative order GM_moon/GM, about 3e-10) is neglected.
Outputs are Saturn-centred: Saturn's position and velocity are subtracted.
"""
from __future__ import annotations

import numpy as np
import rebound

from .force import make_rebound_force
from .params import Saturn, load_moons

YEAR_S = 365.25 * 86400.0  # Julian year
COLUMNS = ["t_s"] + [f"{m}_{c}" for m in ("prometheus", "pandora") for c in ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")]


def run(sat: Saturn, duration_s: float, cadence_s: float, offset_km=(0.0, 0.0, 0.0), epsilon=1e-9):
    """Integrate and return array (N, 13): t, then Saturn-centred states of the two moons."""
    moons = load_moons()
    assert [m.name for m in moons] == ["prometheus", "pandora"]
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=sat.gm)
    for k, m in enumerate(moons):
        pos = list(m.pos)
        if k == 0:  # offset applies to Prometheus only
            pos = [pos[i] + offset_km[i] for i in range(3)]
        sim.add(m=m.gm, x=pos[0], y=pos[1], z=pos[2], vx=m.vel[0], vy=m.vel[1], vz=m.vel[2])
    sim.integrator = "ias15"
    sim.integrator.epsilon = epsilon
    sim.force_is_velocity_dependent = 0
    sim.additional_forces = make_rebound_force(sat)
    n = int(round(duration_s / cadence_s))
    out = np.empty((n + 1, 13))
    for j in range(n + 1):
        t = j * cadence_s
        sim.integrate(t)
        ps = sim.particles
        s = ps[0]
        out[j, 0] = t
        for k in (1, 2):
            p = ps[k]
            out[j, 1 + 6 * (k - 1):7 + 6 * (k - 1)] = (p.x - s.x, p.y - s.y, p.z - s.z, p.vx - s.vx, p.vy - s.vy, p.vz - s.vz)
    return out
