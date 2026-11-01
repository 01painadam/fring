"""Checks of the reference force: finite difference, PLAN 7.2 frequencies, REBOUNDx."""
import numpy as np
import pytest
import rebound
import reboundx

from fring_ref.force import (frequencies, harmonic_acceleration, harmonic_potential,
                             make_rebound_force)
from fring_ref.params import load_saturn

SAT = load_saturn()


def _points(n=40, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.uniform(70000, 200000, n)
    u = rng.uniform(-0.3, 0.3, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - u**2)
    return r[:, None] * np.stack([s * np.cos(ph), s * np.sin(ph), u], axis=1)


def test_gradient_matches_finite_difference():
    pts = _points()
    h = 1.0  # km, central differences
    num = np.empty_like(pts)
    for k in range(3):
        d = np.zeros(3); d[k] = h
        num[:, k] = -(harmonic_potential(SAT, pts + d) - harmonic_potential(SAT, pts - d)) / (2 * h)
    a = harmonic_acceleration(SAT, pts)
    assert np.max(np.abs(a - num) / np.linalg.norm(a, axis=1)[:, None]) < 1e-8


def test_equatorial_circular_matches_plan_7_2():
    for r in (100000.0, 140222.43, 180000.0):
        n, k, nu = frequencies(SAT, r)
        pos = np.array([r, 0.0, 0.0])
        a = harmonic_acceleration(SAT, pos)[0] - SAT.gm / r**2
        assert abs(-a / r - n**2) / n**2 < 1e-13
        # nu^2 = Phi_zz, kappa^2 from the radial gradient, by central differences on the full potential
        def phi(p):
            return -SAT.gm / np.linalg.norm(p) + harmonic_potential(SAT, p)
        h = 5.0
        pzz = (phi(pos + [0, 0, h]) - 2 * phi(pos) + phi(pos - [0, 0, h])) / h**2
        assert abs(pzz - nu**2) / nu**2 < 1e-6
        prr = (phi(pos + [h, 0, 0]) - 2 * phi(pos) + phi(pos - [h, 0, 0])) / h**2
        assert abs(prr + 3 * n**2 - k**2) / k**2 < 1e-6


def test_matches_reboundx_gravitational_harmonics_j2_j4():
    r0, v0 = np.array([140222.43, 0.0, 300.0]), None
    n = frequencies(SAT, 140222.43)[0]
    v0 = np.array([0.0, 140222.43 * n, 0.2])
    out = []
    for mode in ("mine", "rebx"):
        sim = rebound.Simulation()
        sim.G = 1.0
        sim.add(m=SAT.gm)
        sim.add(m=0.0, x=r0[0], y=r0[1], z=r0[2], vx=v0[0], vy=v0[1], vz=v0[2])
        sim.integrator = "ias15"
        if mode == "mine":
            sim.force_is_velocity_dependent = 0
            f = make_rebound_force(SAT, orders=(2, 4))
            sim.additional_forces = f
        else:
            rebx = reboundx.Extras(sim)
            gh = rebx.load_force("gravitational_harmonics")
            rebx.add_force(gh)
            sim.particles[0].params["J2"] = SAT.j2
            sim.particles[0].params["J4"] = SAT.j4
            sim.particles[0].params["R_eq"] = SAT.radius
        sim.integrate(2 * 86400.0)
        p = sim.particles[1]
        out.append(np.array([p.x, p.y, p.z, p.vx, p.vy, p.vz]))
    assert np.max(np.abs(out[0][:3] - out[1][:3])) < 1e-3  # km after 2 d
