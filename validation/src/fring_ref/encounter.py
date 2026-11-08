"""E1-REF: geometric elements and IAS15 encounter runs (A6, from the specification only).

Geometric elements (see docs/handoffs/E1-REF.md, "Departure from Renner and Sicardy 2006"):
  a   guiding radius: the circular-orbit radius with the same z angular momentum, a^2 n(a) = L_z.
      L_z is exactly conserved in the axisymmetric field (J2, J4, J6), so a changes only by the moons.
  e   epicyclic amplitude from the energy excess: E_xy - E_circ(a) = 0.5 kappa(a)^2 a^2 e^2.
  M   epicycle phase, from e sin M = v_r / (a kappa) and the sign of (a - r); 0 at pericentre.
  varpi = theta - 2 e (n/kappa) sin M - M   (longitude of the apse, the inverse of the map used in
      `particles.epicycle_states`).
Units: km, s, rad.  The same functions are used for IAS15 and for WH results.
"""
from __future__ import annotations

import ctypes

import numpy as np
import rebound

from .force import frequencies, harmonic_potential
from .params import Saturn, load_moons
from .particles import _STRIDE, hermite_min_distance, hill_radii_km, make_fast_force  # noqa: F401

DAY = 86400.0
WINDOW_S = 2.0 * DAY  # layer 1 window (ICR-004 section 6)
STEP_S = 356.2537  # step of the run (apps/fring-run/configs/chaos-map.toml)
STRATA = [(0.68, 1.0), (1.0, 2.0), (2.0, 4.1), (4.1, 6.3)]  # measured closest distance, R_H
T0_GRID_S = np.arange(0, 36) * 10 * DAY  # moon epochs of the encounters: first year after the epoch


def r_hill_prometheus(sat: Saturn) -> float:
    """Hill radius of Prometheus, km: |r| at the epoch (params/moons.toml) times (GM_m / 3 GM)^(1/3)."""
    return float(hill_radii_km(sat)[0])


def _phi_total(sat: Saturn, r):
    """Equatorial potential -GM/r + Phi_J(r, z = 0), km^2 s^-2."""
    r = np.asarray(r, dtype=float)
    pos = np.stack([r, 0 * r, 0 * r], axis=-1)
    return -sat.gm / r + harmonic_potential(sat, pos)


def guiding_radius(sat: Saturn, lz):
    """a with a^2 n(a) = lz (fixed point: lz^2 = GM a F(a), F = n^2 a^3 / GM)."""
    lz = np.asarray(lz, dtype=float)
    a = lz**2 / sat.gm
    for _ in range(40):
        q = sat.radius / a
        f = 1 + 1.5 * sat.j2 * q**2 - 15 / 8 * sat.j4 * q**4 + 35 / 16 * sat.j6 * q**6
        a = lz**2 / (sat.gm * f)
    return a


def geo_elements(sat: Saturn, states):
    """Geometric elements of states (N, 6) (Saturn-centred, equatorial).  Returns dict a, e, M, varpi, lam."""
    s = np.atleast_2d(np.asarray(states, dtype=float))
    x, y, vx, vy = s[:, 0], s[:, 1], s[:, 3], s[:, 4]
    rp = np.hypot(x, y)
    lz = x * vy - y * vx
    a = guiding_radius(sat, lz)
    n, kap, _ = frequencies(sat, a)
    e_c = 0.5 * (a * n) ** 2 + _phi_total(sat, a)
    e_xy = 0.5 * (vx**2 + vy**2) + _phi_total(sat, rp)
    e2 = 2 * (e_xy - e_c) / (kap * a) ** 2
    e = np.sqrt(np.maximum(e2, 0.0))
    vr = (x * vx + y * vy) / rp
    esin = vr / (a * kap)
    ecos_mag = np.sqrt(np.maximum(e2 - esin**2, 0.0))
    ecos = np.where(a - rp >= 0, ecos_mag, -ecos_mag)
    m = np.arctan2(esin, ecos)
    th = np.arctan2(y, x)
    lam = th - 2 * e * (n / kap) * np.sin(m)
    varpi = np.mod(lam - m, 2 * np.pi)
    return {"a": a, "e": e, "M": m, "varpi": varpi, "lam": np.mod(lam, 2 * np.pi)}


def osculating_a(sat: Saturn, states):
    """Osculating semi-major axis about Saturn's GM alone (point mass), km.  Only for the noise test."""
    s = np.atleast_2d(np.asarray(states, dtype=float))
    r = np.linalg.norm(s[:, :3], axis=1)
    v2 = np.sum(s[:, 3:6] ** 2, axis=1)
    return 1.0 / (2.0 / r - v2 / sat.gm)


def wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def delta_elements(sat: Saturn, before, after):
    """(Delta a [km], Delta e, Delta varpi [rad, wrapped to (-pi, pi]]) from two arrays of states."""
    b, a = geo_elements(sat, before), geo_elements(sat, after)
    return a["a"] - b["a"], a["e"] - b["e"], wrap_pi(a["varpi"] - b["varpi"])


# ---------------------------------------------------------------- IAS15 with the two moons


def make_sim(sat: Saturn, moon_states, particle_states, epsilon=1e-9, moon_gm_scale=1.0):
    """IAS15 simulation: Saturn free, Prometheus and Pandora (Saturn-centred states (2, 6)), massless particles.

    moon_gm_scale = 0 gives massless moons (a particle far from every moon, for the noise test).
    """
    moons = load_moons()
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=sat.gm)
    for m, st in zip(moons, moon_states):
        sim.add(m=m.gm * moon_gm_scale, x=st[0], y=st[1], z=st[2], vx=st[3], vy=st[4], vz=st[5])
    for s in np.asarray(particle_states):
        sim.add(m=0.0, x=s[0], y=s[1], z=s[2], vx=s[3], vy=s[4], vz=s[5])
    sim.N_active = 3
    sim.integrator = "ias15"
    sim.integrator.epsilon = epsilon
    sim.force_is_velocity_dependent = 0
    sim.additional_forces = make_fast_force(sat)
    return sim


def read_states(sim):
    """Saturn-centred states (N-1, 6) of all bodies except Saturn."""
    n = sim.N
    buf = np.ctypeslib.as_array(ctypes.cast(sim._particles, ctypes.POINTER(ctypes.c_double)), shape=(n, _STRIDE))
    return buf[1:, 0:6] - buf[0, 0:6]


def run_pass(sat: Saturn, moon_states, particle_states, times, epsilon=1e-9, moon_gm_scale=1.0):
    """Integrate and sample at `times` (increasing, times[0] = 0).  Returns (moons (nt,2,6), parts (nt,N,6))."""
    sim = make_sim(sat, moon_states, particle_states, epsilon, moon_gm_scale)
    out = np.empty((len(times), sim.N - 1, 6))
    for j, t in enumerate(times):
        if t > 0:
            sim.integrate(t)
        out[j] = read_states(sim)
    return out[:, :2], out[:, 2:]


def pass_measures(moons, parts, dt, r_hill):
    """Closest approach to Prometheus over the samples (Hermite), its relative speed, and end/start distances.

    Returns dict b_rh, u_kms, d_start_rh, d_end_rh, t_min_s, interior (True if the minimum is not at an end).
    """
    rel = parts[:, :, 0:3] - moons[:, None, 0, 0:3]
    vel = parts[:, :, 3:6] - moons[:, None, 0, 3:6]
    b = hermite_min_distance(rel, vel, dt)
    d = np.linalg.norm(rel, axis=-1)
    k = d.argmin(axis=0)
    # speed at the sampled minimum (close enough for a distribution check)
    u = np.linalg.norm(vel[k, np.arange(d.shape[1])], axis=-1)
    interior = (k > 0) & (k < d.shape[0] - 1)
    return {"b_rh": b / r_hill, "u_kms": u, "d_start_rh": d[0] / r_hill, "d_end_rh": d[-1] / r_hill,
            "t_min_s": k * dt, "interior": interior}


def hill_frame_states(sat: Saturn, moon_state, x_km, y_km, e, m_phase):
    """Particle states at radial offset x_km and along-track offset y_km from Prometheus (Hill frame).

    The position is moon + x r_hat + y phi_hat (r_hat along the moon's position, phi_hat = z_hat x r_hat),
    z = 0.  The velocity is that of an epicycle of amplitude e and phase m_phase through this position,
    so that the guiding radius is a = r / (1 - e cos M) (first-order epicycle, `particles.epicycle_states`).
    """
    from .particles import epicycle_states

    pm = np.asarray(moon_state[:3], dtype=float)
    rh = pm / np.linalg.norm(pm)
    ph = np.cross([0.0, 0.0, 1.0], rh)
    x_km, y_km, e, m_phase = (np.asarray(v, dtype=float) for v in (x_km, y_km, e, m_phase))
    pos = pm[None, :] + x_km[:, None] * rh[None, :] + y_km[:, None] * ph[None, :]
    rp = np.hypot(pos[:, 0], pos[:, 1])
    th = np.arctan2(pos[:, 1], pos[:, 0])
    a = rp / (1 - e * np.cos(m_phase))
    n, kap, _ = frequencies(sat, a)
    lam = th - 2 * e * (n / kap) * np.sin(m_phase)
    return epicycle_states(sat, a, e, 0 * a, m_phase, 0 * a, lam)
