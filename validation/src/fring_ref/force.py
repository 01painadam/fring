"""Zonal-harmonic acceleration (PLAN 7.1) written from the specification.

Potential:  Phi = -(GM/r) [1 - sum_n J_n (R/r)^n P_n(z/r)],  n = 2, 4, 6.
The harmonic part is Phi_J = (GM/r) sum_n J_n (R/r)^n P_n(z/r) and a = -grad Phi.
"""
from __future__ import annotations

import ctypes

import numpy as np

from .params import Saturn

# Legendre polynomials and derivatives as numpy poly1d-style coefficient lists
_P = {
    2: lambda u: (3 * u**2 - 1) / 2,
    4: lambda u: (35 * u**4 - 30 * u**2 + 3) / 8,
    6: lambda u: (231 * u**6 - 315 * u**4 + 105 * u**2 - 5) / 16,
}
_DP = {
    2: lambda u: 3 * u,
    4: lambda u: (140 * u**3 - 60 * u) / 8,
    6: lambda u: (1386 * u**5 - 1260 * u**3 + 210 * u) / 16,
}


def harmonic_potential(sat: Saturn, pos, orders=(2, 4, 6)):
    """Phi_J at pos (shape (...,3)), km^2 s^-2 (negative of the work to infinity)."""
    pos = np.asarray(pos, dtype=float)
    r = np.linalg.norm(pos, axis=-1)
    u = pos[..., 2] / r
    js = {2: sat.j2, 4: sat.j4, 6: sat.j6}
    return sum(sat.gm / r * js[n] * (sat.radius / r) ** n * _P[n](u) for n in orders)


def harmonic_acceleration(sat: Saturn, pos, orders=(2, 4, 6)):
    """a = -grad Phi_J analytically; pos shape (...,3), result same shape (km s^-2)."""
    pos = np.asarray(pos, dtype=float)
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    u = z / r
    js = {2: sat.j2, 4: sat.j4, 6: sat.j6}
    gx = np.zeros_like(r)
    gy = np.zeros_like(r)
    gz = np.zeros_like(r)
    for n in orders:
        c = sat.gm * js[n] * sat.radius**n
        p, dp = _P[n](u), _DP[n](u)
        # f = r^-(n+1) P_n(z/r);  d(z/r)/dx_k = delta_kz / r - z x_k / r^3
        radial = -(n + 1) * r ** (-(n + 3)) * p - r ** (-(n + 4)) * dp * z
        gx += c * radial * x
        gy += c * radial * y
        gz += c * (radial * z + r ** (-(n + 2)) * dp)
    return -np.stack([gx, gy, gz], axis=-1)


def make_rebound_force(sat: Saturn, orders=(2, 4, 6)):
    """additional_forces callback: J acceleration on bodies 1.. relative to body 0 (Saturn).

    Saturn's own reaction to the harmonic force (relative order m/M ~ 3e-10 for the
    moons) is neglected.  Use with sim.force_is_velocity_dependent = 0.
    """

    def force(reb_sim):
        sim = reb_sim.contents
        n = sim.N
        ps = sim.particles
        s = ps[0]
        pos = np.empty((n - 1, 3))
        for i in range(1, n):
            p = ps[i]
            pos[i - 1] = (p.x - s.x, p.y - s.y, p.z - s.z)
        acc = harmonic_acceleration(sat, pos, orders)
        for i in range(1, n):
            p = ps[i]
            p.ax += acc[i - 1, 0]
            p.ay += acc[i - 1, 1]
            p.az += acc[i - 1, 2]

    return force


def frequencies(sat: Saturn, r: float):
    """n, kappa, nu (rad/s) from PLAN 7.2 (J2, J4, J6), exact for a circular equatorial orbit."""
    q = sat.radius / r
    o2 = sat.gm / r**3
    n2 = o2 * (1 + 1.5 * sat.j2 * q**2 - 15 / 8 * sat.j4 * q**4 + 35 / 16 * sat.j6 * q**6)
    k2 = o2 * (1 - 1.5 * sat.j2 * q**2 + 45 / 8 * sat.j4 * q**4 - 175 / 16 * sat.j6 * q**6)
    v2 = o2 * (1 + 4.5 * sat.j2 * q**2 - 75 / 8 * sat.j4 * q**4 + 245 / 16 * sat.j6 * q**6)
    return np.sqrt(n2), np.sqrt(k2), np.sqrt(v2)
