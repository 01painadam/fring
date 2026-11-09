"""D2 fast checks (no simulation binary needed): statistics, comparison rule, configs."""
import struct
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from fring_ref import arc_stats as A
from fring_ref.force import frequencies
from fring_ref.params import load_saturn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import d2_configs  # noqa: E402

SAT = load_saturn()
A0 = 140222.43


def test_mean_motion_and_kappa_match_plan_7_2():
    for r in (1.0e5, A0, 1.8e5):
        n, k, _ = frequencies(SAT, r)
        assert A.mean_motion(SAT, r) == pytest.approx(n, rel=1e-12)
        assert A.kappa(SAT, r) == pytest.approx(k, rel=1e-12)


def test_guiding_radius_inverts_circular_orbit():
    r = np.array([1.0e5, 139000.0, A0, 141500.0, 1.8e5])
    lz = r**2 * A.mean_motion(SAT, r)
    assert np.allclose(A.guiding_radius(SAT, lz), r, rtol=0, atol=1e-7)


def _frame_from_epicycle(e, varpi, n_part=2000, a=A0, seed=1):
    """Epicycle states (first order in e) written as a rotating-frame Frame."""
    rng = np.random.default_rng(seed)
    n0 = float(A.mean_motion(SAT, A0))
    n, k = float(A.mean_motion(SAT, a)), float(A.kappa(SAT, a))
    lam = rng.uniform(0, 2 * np.pi, n_part)
    m = lam - varpi
    R = a * (1 - e * np.cos(m))
    th = lam + 2 * e * np.sin(m)
    vr = a * e * k * np.sin(m)
    vt = a * n * (1 + e * (k / (2 * n)) * 0 ) + a * e * (k * k / (2 * n)) * np.cos(m) * 0
    # tangential velocity from L_z conservation: R * vt = a^2 n (L_z of the guiding circle)
    vt = a**2 * n / R
    x, y = R * np.cos(th), R * np.sin(th)
    # inertial velocity -> rotating frame
    vxi = vr * np.cos(th) - vt * np.sin(th)
    vyi = vr * np.sin(th) + vt * np.cos(th)
    vx = vxi + n0 * y
    vy = vyi - n0 * x
    pos = np.stack([x, y, np.zeros(n_part)], 1)
    vel = np.stack([vx, vy, np.zeros(n_part)], 1)
    return A.Frame(0, 0.0, A0, n0, 0.0, np.arange(n_part, dtype="u4"), pos, vel)


def test_epicycle_recovers_e_and_apse():
    e, varpi = 0.00235, np.deg2rad(353.78)
    o = A.orbit_elements(SAT, _frame_from_epicycle(e, varpi))
    assert np.mean(o.e) == pytest.approx(e, rel=5e-3)
    z = (o.e * np.exp(1j * o.varpi_rot)).mean()
    assert abs(z) == pytest.approx(e, rel=5e-3)
    assert A.wrap_pi(np.angle(z) - varpi) == pytest.approx(0.0, abs=3e-3)


def test_rotation_does_not_change_rg_or_e_but_changes_apse():
    """The reason for the extra statistics: a wrap leaves r_g and e alone."""
    fr = _frame_from_epicycle(0.00235, 0.3)
    o = A.orbit_elements(SAT, fr)
    ang = np.deg2rad(10.0)
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    fr2 = A.Frame(0, 0.0, A0, fr.n0, 0.0, fr.id, fr.pos @ rot.T, fr.vel @ rot.T)
    o2 = A.orbit_elements(SAT, fr2)
    assert np.allclose(o2.rg, o.rg, atol=1e-6) and np.allclose(o2.e, o.e, atol=1e-10)
    assert A.wrap_pi(np.median(o2.varpi_rot - o.varpi_rot)) == pytest.approx(ang, abs=1e-6)


def test_local_rms_sees_the_coherent_ribbon():
    """Ring: coherent e gives a small local RMS; apses spread over the circle give ae/sqrt(2)."""
    ring = A.Domain(360.0)
    fr = _frame_from_epicycle(0.00235, 0.3, n_part=200000)
    s = A.frame_stats(SAT, fr, ring, A0)
    assert s["loc_rms"] < 5.0 and s["e_coh"] / s["e_mean"] > 0.99
    # the same particles with the apse spread: each particle gets a random rotation k*10 deg
    rng = np.random.default_rng(3)
    k = rng.integers(0, 36, len(fr.id))
    ang = np.deg2rad(10.0) * k
    c, sn = np.cos(ang), np.sin(ang)
    px, py = fr.pos[:, 0] * c - fr.pos[:, 1] * sn, fr.pos[:, 0] * sn + fr.pos[:, 1] * c
    vx, vy = fr.vel[:, 0] * c - fr.vel[:, 1] * sn, fr.vel[:, 0] * sn + fr.vel[:, 1] * c
    z = np.zeros(len(c))
    fr2 = A.Frame(0, 0.0, A0, fr.n0, 0.0, fr.id, np.stack([px, py, z], 1), np.stack([vx, vy, z], 1))
    s2 = A.frame_stats(SAT, fr2, ring, A0)
    assert s2["loc_rms"] == pytest.approx(A0 * 0.00235 / np.sqrt(2), rel=0.02)
    assert s2["e_coh"] / s2["e_mean"] < 0.01
    assert s2["vdisp"] > 20 * s["vdisp"]


def test_longitude_bins_arc_across_180():
    dom = A.Domain(10.0, 180.0)
    th = np.deg2rad([175.5, 179.5, -179.5, -175.5])
    idx, nb = A.longitude_bin(dom, th)
    assert nb == 10 and list(idx) == [0, 4, 5, 9]


def test_compare_and_pass_rule():
    c = A.compare([100.0, 100.4], [101.0, 101.2, 100.8, 101.0])
    assert c["d"] == pytest.approx(1.01 / 1.002 - 1, abs=1e-3) or abs(c["d"]) < 0.02
    assert c["bound"] > abs(c["d"])
    big = A.compare([100.0, 100.2], [150.0, 151.0, 149.0, 150.0])
    assert not A.passes([big["bound"]]) and A.passes([c["bound"]])
    assert A.recommend({5: {"a": False, "b": True}, 10: {"a": True, "b": True}, 30: {"a": True, "b": True}}) == 10
    assert A.recommend({5: {"a": False}, 30: {"a": False}}) is None


def test_predicted_apse_slopes():
    p = A.predicted_slopes(SAT, A0)
    assert p["wrap_slope"] == pytest.approx(1.5 * float(A.mean_motion(SAT, A0)) / A0 * 180 / np.pi * 86400, rel=0.01)
    assert abs(p["wrap_slope"] / p["precession_slope"]) > 50


def test_read_frame_roundtrip(tmp_path):
    n = 3
    hdr = b"FRINGFRM" + struct.pack("<IIII", 1, 72, 1, 1) + struct.pack("<QQ", 5, n) + struct.pack("<dddd", 10.0, A0, 1e-4, 0.0)
    ids = np.array([2, 0, 1], "<u4")
    cols = [np.arange(n, dtype="<f4") + 10 * j for j in range(6)]
    p = tmp_path / "f.bin"
    p.write_bytes(hdr + ids.tobytes() + b"".join(c.tobytes() for c in cols))
    fr = A.read_frame(p)
    assert fr.step == 5 and list(fr.id) == [0, 1, 2] and fr.pos[0, 0] == 1.0 and fr.vel[2, 0] == 30.0


def test_configs_valid():
    names = []
    for name, dom, w, c, seed, core in d2_configs.all_runs(False):
        t = tomllib.loads(d2_configs.config_text(core, dom, w, c, seed, name, 5.0, 22145))
        assert t["collisions"]["model"] == "none" and t["particles"]["count"] == 100000
        assert t["domain"]["kind"] == ("full_ring" if dom == "ring" else "arc")
        assert t["particles"]["distribution"]["eccentricity"] == (0.00235 if core == "ecc" else 0.0)
        names.append(name)
    assert len(names) == len(set(names)) == 28


def test_window_mask_and_paired_compare():
    dom = A.Domain(10.0, 180.0)
    th = np.deg2rad([174.0, 176.0, 180.0, -176.0, -174.0])
    assert list(A.window_mask(dom, th)) == [False, True, True, True, False]
    pairs = [[100.0, 102.0], [100.0, 98.0], [50.0, 52.0], [10.0, 10.2]]
    c = A.compare_paired(pairs, [101.0, 99.0, 51.0, 10.1])
    assert abs(c["d"]) < 1e-12 and c["bound"] < 0.05
    c2 = A.compare_paired(pairs, [150.0, 150.0, 75.0, 15.0])
    assert c2["d"] == pytest.approx(0.5, abs=0.03) and c2["bound"] > 0.5


def test_frame_stats_selection_matches_subset():
    fr = _frame_from_epicycle(0.00235, 0.3, n_part=20000, seed=5)
    o = A.orbit_elements(SAT, fr)
    dom = A.Domain(10.0, 0.0)
    sel = A.window_mask(dom, o.theta)
    s1 = A.frame_stats(SAT, fr, dom, A0, o, sel)
    assert s1["n_core"] == int(sel.sum()) and 400 < s1["n_core"] < 700
    assert s1["loc_nbins"] == 10
