"""V3: recompute the first orbit of 3 particles, regenerate the initial states from the seed, test the compare tool."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(not (REF / "v3_traj.npy").exists(), reason="reference file not generated yet")


def test_first_orbit_matches_committed_reference():
    dpos, dvel = _load("v3_particles").check()
    assert dpos < 1e-6 and dvel < 1e-12


def test_initial_states_regenerate_from_seed():
    m = _load("v3_particles")
    from fring_ref.params import load_saturn
    from fring_ref.particles import epicycle_states
    rng = np.random.default_rng(np.random.PCG64(m.SEED))
    elems = m.draw(rng, m.BATCH)
    rows = [r.split(",") for r in (REF / "v3_particles_meta.csv").read_text().splitlines() if not r.startswith("#")][1:]
    kept = [int(r[0]) for r in rows if r[1] != "-1"]
    ic = np.loadtxt(REF / "v3_ic.csv", delimiter=",", comments="#")[:, 1:]
    states = epicycle_states(load_saturn(), *elems[kept].T)
    assert np.array_equal(states, ic)


def test_epicycle_amplitudes():
    """A reference orbit has a radial range of about 2 a e and a vertical range of about 2 a i."""
    from fring_ref.params import load_saturn
    from fring_ref.particles import A0_KM, epicycle_states, orbit_period, simulate
    sat = load_saturn()
    e, inc = 5e-4, 1e-4
    st = epicycle_states(sat, [A0_KM], [e], [inc], [0.3], [1.0], [2.0])
    t = np.linspace(0, 3 * orbit_period(sat), 301)
    _, p = simulate(sat, st, t)
    r = np.linalg.norm(p[:, 0, 0:3], axis=1)
    assert abs((r.max() - r.min()) / (2 * A0_KM * e) - 1) < 0.1
    assert abs((p[:, 0, 2].max() - p[:, 0, 2].min()) / (2 * A0_KM * inc) - 1) < 0.1


def test_compare_tool_accepts_reference_and_flags_offset(tmp_path):
    c = _load("v3_compare")
    ref = np.load(REF / "v3_traj.npy")
    f = tmp_path / "x.csv"
    rows = [(k, 0.0, i, *ref[k, i]) for k in (0, 500, 1000) for i in range(100)]
    np.savetxt(f, rows, delimiter=",", fmt="%.17g")
    assert c.compare(f, plot=False) == 0.0
    rows = [(k, 0.0, i, ref[k, i, 0] + 2.0, *ref[k, i, 1:]) for k in (0, 500, 1000) for i in range(100)]
    np.savetxt(f, rows, delimiter=",", fmt="%.17g")
    assert abs(c.compare(f, plot=False) - 2.0) < 1e-9
