"""Test particles near the F ring core with the two moons (V3).

Same model as `moons.py` (ADR 0004): Saturn is a free massive body, the moons carry their GM, the
J2+J4+J6 force acts on moons and particles relative to Saturn, outputs are Saturn-centred.
The particles are massless, so they do not disturb the moons or each other.
Units: km, km/s, s.
"""
from __future__ import annotations

import ctypes

import numpy as np
import rebound

from .force import frequencies, harmonic_acceleration
from .params import Saturn, load_moons

A0_KM = 140222.43  # core semi-major axis (task V3)
_STRIDE = ctypes.sizeof(rebound.Particle) // 8  # doubles per particle record


def orbit_period(sat: Saturn, a0: float = A0_KM) -> float:
    """One 'orbit' of the V3 files: 2 pi / n(a0), with n from the J2+J4+J6 frequencies (PLAN 7.2), s."""
    return 2 * np.pi / frequencies(sat, a0)[0]


def hill_radii_km(sat: Saturn):
    """Hill radius a (GM_moon / (3 GM_Saturn))^(1/3) of each moon, with a = |r| at the epoch, km."""
    return np.array([np.linalg.norm(m.pos) * (m.gm / (3 * sat.gm)) ** (1 / 3) for m in load_moons()])


def epicycle_states(sat: Saturn, a, e, inc, mean_anom, phase_z, lam):
    """Cartesian states (N, 6) from first-order epicyclic elements about circular orbits of radius a.

    r = a (1 - e cos M), theta = lam + 2 e (n/kappa) sin M, vr = a e kappa sin M,
    vtheta = a n (1 + e cos M), z = a i sin(phi), vz = a i nu cos(phi).
    n, kappa, nu are the exact frequencies at a (PLAN 7.2).  e and i are epicyclic amplitudes, not
    osculating elements.  Saturn-equator frame, z along the spin axis.
    """
    a, e, inc, mean_anom, phase_z, lam = (np.asarray(v, dtype=float) for v in (a, e, inc, mean_anom, phase_z, lam))
    n, kap, nu = frequencies(sat, a)
    r = a * (1 - e * np.cos(mean_anom))
    th = lam + 2 * e * (n / kap) * np.sin(mean_anom)
    vr = a * e * kap * np.sin(mean_anom)
    vt = a * n * (1 + e * np.cos(mean_anom))
    c, s = np.cos(th), np.sin(th)
    out = np.empty((a.size, 6))
    out[:, 0], out[:, 1] = r * c, r * s
    out[:, 2] = a * inc * np.sin(phase_z)
    out[:, 3], out[:, 4] = vr * c - vt * s, vr * s + vt * c
    out[:, 5] = a * inc * nu * np.cos(phase_z)
    return out


def make_fast_force(sat: Saturn, orders=(2, 4, 6)):
    """Same force as force.make_rebound_force, but reads and writes the particle array with numpy."""

    def force(reb_sim):
        sim = reb_sim.contents
        n = sim.N
        buf = np.ctypeslib.as_array(ctypes.cast(sim._particles, ctypes.POINTER(ctypes.c_double)), shape=(n, _STRIDE))
        acc = harmonic_acceleration(sat, buf[1:, 0:3] - buf[0, 0:3], orders)
        buf[1:, 6:9] += acc

    return force


def simulate(sat: Saturn, states, times, epsilon=1e-9):
    """Moons plus massless particles.  states (N, 6) at t = 0; times (nt,) increasing, times[0] = 0.

    Returns (moons (nt, 2, 6), parts (nt, N, 6)), Saturn-centred.
    """
    moons = load_moons()
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=sat.gm)
    for m in moons:
        sim.add(m=m.gm, x=m.pos[0], y=m.pos[1], z=m.pos[2], vx=m.vel[0], vy=m.vel[1], vz=m.vel[2])
    for s in np.asarray(states):
        sim.add(m=0.0, x=s[0], y=s[1], z=s[2], vx=s[3], vy=s[4], vz=s[5])
    sim.N_active = 3
    sim.integrator = "ias15"
    sim.integrator.epsilon = epsilon
    sim.force_is_velocity_dependent = 0
    sim.additional_forces = make_fast_force(sat)
    n = sim.N
    out = np.empty((len(times), n - 1, 6))
    for j, t in enumerate(times):
        if t > 0:
            sim.integrate(t)
        buf = np.ctypeslib.as_array(ctypes.cast(sim._particles, ctypes.POINTER(ctypes.c_double)), shape=(n, _STRIDE))
        out[j] = buf[1:, 0:6] - buf[0, 0:6]
    return out[:, :2], out[:, 2:]


def hermite_min_distance(rel, vel, dt, sub=21):
    """Minimum of |r(t)| over a trajectory sampled every dt, with cubic Hermite interpolation.

    rel, vel: (nt, ..., 3) relative position and velocity.  Returns the minimum over time, shape (...).
    """
    s = np.linspace(0.0, 1.0, sub).reshape((sub,) + (1,) * (rel.ndim))
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    r0, r1 = rel[:-1][None], rel[1:][None]
    v0, v1 = vel[:-1][None] * dt, vel[1:][None] * dt
    p = h00 * r0 + h10 * v0 + h01 * r1 + h11 * v1
    return np.linalg.norm(p, axis=-1).min(axis=(0, 1))
