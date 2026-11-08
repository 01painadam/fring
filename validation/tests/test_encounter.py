"""E1-REF: geometric elements (round trip with epicycle_states, flatness in IAS15 with the moons off)."""
import numpy as np

from fring_ref.encounter import geo_elements, guiding_radius, osculating_a, r_hill_prometheus, run_pass, wrap_pi
from fring_ref.params import load_moons, load_saturn
from fring_ref.particles import epicycle_states

SAT = load_saturn()


def test_hill_radius():
    assert abs(r_hill_prometheus(SAT) - 63.3) < 0.1  # ICR-004: 10 R_H = 634 km


def test_round_trip_epicycle_states():
    rng = np.random.default_rng(1)
    n = 500
    a = rng.uniform(139500, 141000, n)
    e = rng.uniform(0, 0.0024, n)
    m = rng.uniform(0, 2 * np.pi, n)
    lam = rng.uniform(0, 2 * np.pi, n)
    g = geo_elements(SAT, epicycle_states(SAT, a, e, 0 * a, m, 0 * a, lam))
    # epicycle_states is first order in e, the geometric a is exact, so a differs by O(a e^2)
    assert np.all(abs(g["a"] - a) < 2.5 * a * e**2 + 1e-6)
    assert abs(g["e"] - e).max() < 1e-5
    big = e > 5e-4  # the phase is defined when e is not tiny
    assert abs(wrap_pi(g["M"] - m))[big].max() < 3e-3


def test_guiding_radius_circular():
    # a circular equatorial orbit of radius r has guiding radius r
    from fring_ref.force import frequencies

    r = np.array([100000.0, 140000.0])
    n = frequencies(SAT, r)[0]
    assert np.allclose(guiding_radius(SAT, r * r * n), r, rtol=1e-13)


def test_geometric_a_is_flat_osculating_is_not():
    ms = np.array([list(m.pos) + list(m.vel) for m in load_moons()])
    t = np.arange(0, 2 * 86400 + 1, 1200.0)
    st = epicycle_states(SAT, [140000.0], [0.0024], [0.0], [1.0], [0.0], [0.5])
    _, pa = run_pass(SAT, ms, st, t, 1e-9, moon_gm_scale=0.0)
    g = geo_elements(SAT, pa[:, 0])
    assert np.ptp(g["a"]) < 1e-8  # km, round-off
    assert np.ptp(osculating_a(SAT, pa[:, 0])) > 1.0  # km, the J2 artefact
