"""D2 (role A6): statistics for the arc against full-ring test, from frames only.

Written from the specification: the force law of PLAN 7.1 (planar part, z = 0),
the epicycle approximation of PLAN 4.2 and the frame file of docs/snapshot-format.md.
No code of the simulation crates is used.

Per particle (positions and velocities in the rotating frame, f32 in the file):
    L_z = R^2 (n0 + thetadot_rot)      exact in Saturn's axisymmetric field
    r_g = guiding radius, the circular orbit with this L_z
    A   = sqrt(2 (E_h - E_circ(L_z))) / kappa(r_g)      epicycle amplitude
    e   = A / r_g
    psi = epicycle angle, xi = R - r_g = -A cos(psi), vr = A kappa sin(psi)
    varpi = theta - 2 e sin(psi) - psi      longitude of pericentre (first order in e)
A rotation about z (the arc wrap) changes theta and leaves L_z, r_g, A, e, psi alone,
so r_g and e cannot see an error in the apse; the local statistics, the coherent
eccentricity and the apse drift can.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .params import Saturn, load_saturn

YEAR_S = 31_557_600.0
P_ZERO = {2: -0.5, 4: 3.0 / 8.0, 6: -5.0 / 16.0}  # P_n(0)


# ---------------------------------------------------------------- planar field

def _terms(sat: Saturn):
    js = {2: sat.j2, 4: sat.j4, 6: sat.j6}
    return [(n, js[n] * sat.radius**n * P_ZERO[n]) for n in (2, 4, 6)]


def phi(sat: Saturn, r):
    """Potential in the equatorial plane, km^2 s^-2: -GM/r + GM sum c_n r^-(n+1)."""
    r = np.asarray(r, dtype=float)
    return -sat.gm / r + sat.gm * sum(c * r ** (-(n + 1)) for n, c in _terms(sat))


def dphi(sat: Saturn, r):
    """dPhi/dr (= r Omega^2), km s^-2."""
    r = np.asarray(r, dtype=float)
    return sat.gm / r**2 - sat.gm * sum((n + 1) * c * r ** (-(n + 2)) for n, c in _terms(sat))


def d2phi(sat: Saturn, r):
    r = np.asarray(r, dtype=float)
    return -2 * sat.gm / r**3 + sat.gm * sum((n + 1) * (n + 2) * c * r ** (-(n + 3)) for n, c in _terms(sat))


def mean_motion(sat: Saturn, r):
    return np.sqrt(dphi(sat, r) / np.asarray(r, dtype=float))


def kappa(sat: Saturn, r):
    r = np.asarray(r, dtype=float)
    return np.sqrt(d2phi(sat, r) + 3 * dphi(sat, r) / r)


def guiding_radius(sat: Saturn, lz, r0: float = 1.4e5):
    """Solve L_z^2 = r^3 dPhi/dr for r (Newton, vectorised)."""
    lz = np.asarray(lz, dtype=float)
    r = np.full_like(lz, r0)
    for _ in range(40):
        f = r**3 * dphi(sat, r) - lz**2
        fp = 3 * r**2 * dphi(sat, r) + r**3 * d2phi(sat, r)
        step = f / fp
        r = r - step
        if np.all(np.abs(step) < 1e-10):
            break
    return r


# ---------------------------------------------------------------------- frames

@dataclass
class Frame:
    step: int
    time_s: float
    a0_km: float
    n0: float
    t_ref_s: float
    id: np.ndarray
    pos: np.ndarray  # (N,3) km, rotating frame, float64 copy of the f32 values
    vel: np.ndarray  # (N,3) km/s, relative to the rotating frame


def read_frame(path) -> Frame:
    """Read a frame file (docs/snapshot-format.md section 5), particles sorted by id."""
    raw = Path(path).read_bytes()
    if raw[:8] != b"FRINGFRM":
        raise ValueError("not a frame file")
    version, header_len, ftype, flags = struct.unpack_from("<IIII", raw, 8)
    step, n = struct.unpack_from("<QQ", raw, 24)
    time_s, a0, n0, t_ref = struct.unpack_from("<dddd", raw, 40)
    if version != 1 or ftype != 1 or not (flags & 1):
        raise ValueError(f"unsupported frame (version {version}, type {ftype}, flags {flags})")
    if len(raw) != header_len + 28 * n:
        raise ValueError("frame size does not match the header")
    o = header_len
    ids = np.frombuffer(raw, "<u4", n, o)
    o += 4 * n
    cols = []
    for _ in range(6):
        cols.append(np.frombuffer(raw, "<f4", n, o).astype(float))
        o += 4 * n
    order = np.argsort(ids)
    pos = np.stack(cols[:3], axis=1)[order]
    vel = np.stack(cols[3:], axis=1)[order]
    return Frame(step, time_s, a0, n0, t_ref, ids[order].copy(), pos, vel)


def frame_files(run_dir) -> list[Path]:
    return sorted((Path(run_dir) / "frames").glob("*.bin"))


# ---------------------------------------------------------------- per particle

@dataclass
class Orbit:
    theta: np.ndarray  # rotating-frame longitude, rad
    R: np.ndarray
    lz: np.ndarray
    rg: np.ndarray
    e: np.ndarray
    psi: np.ndarray
    varpi_rot: np.ndarray  # rad, rotating frame
    kappa: np.ndarray


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def orbit_elements(sat: Saturn, fr: Frame) -> Orbit:
    x, y = fr.pos[:, 0], fr.pos[:, 1]
    vx, vy = fr.vel[:, 0], fr.vel[:, 1]
    R = np.hypot(x, y)
    lz = (x * vy - y * vx) + fr.n0 * R**2  # inertial L_z
    vr = (x * vx + y * vy) / R
    rg = guiding_radius(sat, lz)
    k = kappa(sat, rg)
    e_h = 0.5 * vr**2 + 0.5 * lz**2 / R**2 + phi(sat, R)
    e_c = 0.5 * lz**2 / rg**2 + phi(sat, rg)
    amp = np.sqrt(np.maximum(2 * (e_h - e_c), 0.0)) / k
    ecc = amp / rg
    xi = R - rg
    psi = np.arctan2(vr / k, -xi)
    theta = np.arctan2(y, x)
    varpi = theta - 2 * ecc * np.sin(psi) - psi
    return Orbit(theta, R, lz, rg, ecc, psi, varpi, k)


# ------------------------------------------------------------------ statistics

def pw(v) -> float:
    """(P84 - P16) / 2, the 1-sigma width of a Gaussian."""
    p16, p84 = np.percentile(v, [15.865, 84.135])
    return 0.5 * (p84 - p16)


def _pooled_rms(values: np.ndarray, group: np.ndarray, ngroups: int) -> tuple[float, int]:
    """Pooled within-group RMS (unbiased variance); values (N,) or (N,k) summed over k."""
    v = values.reshape(len(values), -1)
    n = np.bincount(group, minlength=ngroups).astype(float)
    ss = 0.0
    for j in range(v.shape[1]):
        s = np.bincount(group, v[:, j], ngroups)
        q = np.bincount(group, v[:, j] ** 2, ngroups)
        ss = ss + (q - np.where(n > 0, s**2 / np.maximum(n, 1), 0.0))
    ok = n >= 2
    return float(np.sqrt(ss[ok].sum() / (n[ok] - 1).sum())), int(ok.sum())


@dataclass(frozen=True)
class Domain:
    """width_deg = 360 for the full ring; centre_deg is the rotating-frame longitude."""
    width_deg: float
    centre_deg: float = 0.0

    @property
    def ring(self) -> bool:
        return self.width_deg >= 360.0


def longitude_bin(dom: Domain, theta, bin_deg: float = 1.0):
    """Index of the longitude bin and the number of bins. The arc starts at centre - width/2."""
    if dom.ring:
        s = np.mod(theta, 2 * np.pi)
        nb = int(round(360.0 / bin_deg))
    else:
        s = wrap_pi(theta - np.deg2rad(dom.centre_deg)) + np.deg2rad(dom.width_deg) / 2
        nb = int(round(dom.width_deg / bin_deg))
    idx = np.clip(np.floor(np.rad2deg(s) / bin_deg).astype(int), 0, nb - 1)
    return idx, nb


def frame_stats(sat: Saturn, fr: Frame, dom: Domain, a0: float, orb: Orbit | None = None) -> dict:
    """The D2 statistics of one frame (km, km/s)."""
    o = orb if orb is not None else orbit_elements(sat, fr)
    ok = np.isfinite(o.rg) & (np.abs(o.rg - a0) < 500.0)
    n_all = len(o.rg)
    d = {"n": n_all, "n_core": int(ok.sum())}
    rg = o.rg[ok]
    d["sigma_rg"] = float(np.std(rg, ddof=1))
    d["pw_rg"] = pw(rg)
    d["e_mean"] = float(o.e[ok].mean())
    d["e_pw"] = pw(o.e[ok])
    z = (o.e * np.exp(1j * o.varpi_rot))[ok]
    d["e_coh"] = float(abs(z.mean()))
    d["varpi_coh_rot_deg"] = float(np.rad2deg(np.angle(z.mean())))
    d["varpi_r"] = float(abs(np.exp(1j * o.varpi_rot[ok]).mean()))
    # local statistics: 1 deg longitude bins
    idx, nb = longitude_bin(dom, o.theta[ok])
    d["loc_rms"], d["loc_nbins"] = _pooled_rms(o.R[ok] - a0, idx, nb)
    n_b = np.bincount(idx, minlength=nb)
    w = []
    for b in range(nb):
        if n_b[b] >= 20:
            w.append((pw(o.R[ok][idx == b]), n_b[b]))
    d["loc_pw"] = float(sum(a * c for a, c in w) / sum(c for _, c in w)) if w else float("nan")
    # velocity dispersion (RMS of the 3-vector v about the cell mean, km/s): cells of 1 deg x 5 km in R
    rb = np.floor((o.R[ok] - a0) / 5.0).astype(int)
    rb -= rb.min()
    cell = idx * (rb.max() + 1) + rb
    d["vdisp"], d["vdisp_ncells"] = _pooled_rms(fr.vel[ok], cell, nb * (rb.max() + 1))
    return d


def apse_slope(sat: Saturn, orb0: Orbit, orb: Orbit, fr: Frame, a0: float) -> dict:
    """Fit of the apse drift against the guiding radius at t = 0.

    d_varpi = varpi(t) - varpi(0) (inertial, mod 2 pi) against rho = r_g(0) - a0.
    Returns the slope in deg per day per km, the common drift in deg per day, and the
    residual scatter. The residual is taken about the circular mean.
    """
    t_days = (fr.time_s - fr.t_ref_s) / 86400.0
    ok = np.isfinite(orb0.rg) & (np.abs(orb0.rg - a0) < 500.0)
    dv = wrap_pi((orb.varpi_rot - orb0.varpi_rot) + fr.n0 * (fr.time_s - fr.t_ref_s))[ok]
    mu = np.angle(np.exp(1j * dv).mean())
    res = wrap_pi(dv - mu)
    rho = (orb0.rg - a0)[ok]
    A = np.vstack([rho, np.ones_like(rho)]).T
    (m, c), *_ = np.linalg.lstsq(A, res, rcond=None)
    scatter = float(np.std(res - (m * rho + c)))
    deg = 180 / np.pi
    return {
        "t_days": t_days,
        "apse_slope_deg_day_km": float(m * deg / t_days),
        "apse_common_deg_day": float((mu + 0.0) * deg / t_days),
        "apse_scatter_deg": scatter * deg,
    }


def predicted_slopes(sat: Saturn, a0: float) -> dict:
    """Predictions in deg per day per km at a0 (central differences, 1 km)."""
    h = 1.0
    n = lambda r: float(mean_motion(sat, r))
    kp = lambda r: float(kappa(sat, r))
    dn = (n(a0 + h) - n(a0 - h)) / (2 * h)
    dprec = ((n(a0 + h) - kp(a0 + h)) - (n(a0 - h) - kp(a0 - h))) / (2 * h)
    f = 180 / np.pi * 86400.0
    return {
        "wrap_slope": -dn * f,  # the arc: apse rate -(n(r) - n0)
        "precession_slope": dprec * f,  # the real differential precession, d(n - kappa)/dr
    }


# ------------------------------------------------------------------ comparison

def compare(ring: list[float], arcs: list[float]) -> dict:
    """Relative difference of the arc statistic (mean over centres) from the ring mean.

    ring: values of the ring runs (seeds); arcs: values of the arc runs (centres).
    bound = |mean d| + 2 sqrt(sem_arc^2 + noise_ring^2), where sem_arc is the standard
    error over centres and noise_ring the standard error of the ring mean (half the
    spread for 2 seeds, relative).
    """
    ring = np.asarray(ring, float)
    arcs = np.asarray(arcs, float)
    rm = ring.mean()
    d = arcs / rm - 1.0
    sem_arc = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    noise = (ring.std(ddof=1) / np.sqrt(len(ring)) / abs(rm)) if len(ring) > 1 else 0.0
    mean_d = float(d.mean())
    return {
        "ring": float(rm), "arc": float(arcs.mean()), "d": mean_d,
        "d_max": float(np.abs(d).max()), "sem_arc": float(sem_arc), "noise_ring": float(noise),
        "bound": float(abs(mean_d) + 2 * np.hypot(sem_arc, noise)),
    }


def passes(bounds: list[float], limit: float = 0.05) -> bool:
    return bool(all(np.isfinite(b) and b < limit for b in bounds))


def recommend(table: dict[float, dict[str, bool]]) -> float | None:
    """Smallest width for which every statistic passes; None if none does."""
    ok = [w for w, st in sorted(table.items()) if all(st.values())]
    return ok[0] if ok else None
