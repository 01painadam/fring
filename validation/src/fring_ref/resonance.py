"""Analysis of the Prometheus-Pandora run: mean elements, resonant angles, chaos indicators.

"Mean" here means orbit-averaged, not geometric: the Saturn-centred two-body
eccentricity vector (mu = GM_Saturn + GM_moon) and the azimuth are averaged over one
orbital period with a boxcar window (exact integral of a cubic spline), which removes
the J2 forced short-period terms.  The mean longitude is the window average of the
unwrapped azimuth; the mean longitude of pericentre is the direction of the averaged
eccentricity vector.  These are not Borderies-Longaretti geometric elements.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .params import Saturn, load_moons

DAY = 86400.0
YEAR = 365.25 * DAY


def split(arr):
    t = arr[:, 0]
    return t, [(arr[:, 1 + 6 * k:4 + 6 * k], arr[:, 4 + 6 * k:7 + 6 * k]) for k in (0, 1)]


def window_average(t, y, width, tout):
    """Boxcar average of y(t) over [tout - width/2, tout + width/2] using a cubic spline."""
    F = CubicSpline(t, y, axis=0).antiderivative()
    return (F(tout + width / 2) - F(tout - width / 2)) / width


def mean_elements(sat: Saturn, arr, tout):
    """Return per-moon dict: lam (mean longitude, rad, unwrapped), varpi (rad, unwrapped), ecc, n (rad/s, fit)."""
    t, states = split(arr)
    moons = load_moons()
    out = []
    for (pos, vel), m in zip(states, moons):
        mu = sat.gm + m.gm
        r = np.linalg.norm(pos, axis=1)
        h = np.cross(pos, vel)
        evec = np.cross(vel, h) / mu - pos / r[:, None]
        theta = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
        n0 = (theta[-1] - theta[0]) / (t[-1] - t[0])
        width = 2 * np.pi / n0
        lam = window_average(t, theta, width, tout)
        ebar = window_average(t, evec[:, :2], width, tout)
        varpi = np.unwrap(np.arctan2(ebar[:, 1], ebar[:, 0]))
        out.append(dict(name=m.name, lam=lam, varpi=varpi, ecc=np.hypot(ebar[:, 0], ebar[:, 1]), n_mean=n0, period_s=width))
    return out


def mean_motion(t, lam, half_window_s):
    """Central-difference mean motion (rad/s) of the mean longitude over +-half_window_s."""
    dt = t[1] - t[0]
    k = int(round(half_window_s / dt))
    n = np.full_like(lam, np.nan)
    n[k:-k] = (lam[2 * k:] - lam[:-2 * k]) / (2 * k * dt)
    return n


def resonant_angles(lam_pr, lam_pd, vp_pr, vp_pd):
    """phi_k = 121 lam_Pd - 118 lam_Pr - k varpi_Pr - (3-k) varpi_Pd, k = 0..3, wrapped to (-pi, pi]."""
    base = 121 * lam_pd - 118 * lam_pr
    return [np.angle(np.exp(1j * (base - k * vp_pr - (3 - k) * vp_pd))) for k in range(4)]


def conjunctions(t, pos_pr, pos_pd):
    """Times of conjunctions (azimuth difference crossing multiples of 2 pi) and the radial separation there."""
    d = np.unwrap(np.arctan2(pos_pr[:, 1], pos_pr[:, 0])) - np.unwrap(np.arctan2(pos_pd[:, 1], pos_pd[:, 0]))
    k0 = np.ceil(d[0] / (2 * np.pi))
    k1 = np.floor(d[-1] / (2 * np.pi))
    ks = np.arange(k0, k1 + 1)
    tc, sep = [], []
    rpr, rpd = np.linalg.norm(pos_pr, axis=1), np.linalg.norm(pos_pd, axis=1)
    for k in ks:
        i = np.searchsorted(d, 2 * np.pi * k) - 1
        if i < 0 or i >= len(t) - 1:
            continue
        f = (2 * np.pi * k - d[i]) / (d[i + 1] - d[i])
        tc.append(t[i] + f * (t[i + 1] - t[i]))
        sep.append((1 - f) * (rpd[i] - rpr[i]) + f * (rpd[i + 1] - rpr[i + 1]))
    return np.array(tc), np.array(sep)


def divergence(main, shadow):
    """Position differences |dr| of each moon between two runs on the same time grid (km)."""
    dpr = np.linalg.norm(main[:, 1:4] - shadow[:, 1:4], axis=1)
    dpd = np.linalg.norm(main[:, 7:10] - shadow[:, 7:10], axis=1)
    return dpr, dpd


def first_crossing(t, y, level):
    """First time y reaches level, linear interpolation; nan if never."""
    idx = np.nonzero(y >= level)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return t[0]
    return t[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (t[i] - t[i - 1])


def lyapunov(t, d, lo=1e-2, hi=1e2, env_days=10.0):
    """Exponential growth rate (per year) of the envelope of d(t) between lo and hi km.

    The envelope is the running maximum over env_days, to remove the orbital oscillation.
    Returns (rate_per_year, t_start, t_end, r2) or None.
    """
    dt = t[1] - t[0]
    w = max(1, int(round(env_days * DAY / dt)))
    from scipy.ndimage import maximum_filter1d
    env = maximum_filter1d(d, size=w)
    t0, t1 = first_crossing(t, env, lo), first_crossing(t, env, hi)
    if np.isnan(t0) or np.isnan(t1) or t1 <= t0:
        return None
    sel = (t >= t0) & (t <= t1)
    p = np.polyfit(t[sel], np.log(env[sel]), 1)
    resid = np.log(env[sel]) - np.polyval(p, t[sel])
    r2 = 1 - resid.var() / np.log(env[sel]).var()
    return p[0] * YEAR, t0, t1, r2


def growth_rate(t, d, t_from_yr, t_to_yr, env_days=10.0):
    """Slope (per year) of ln of the running-max envelope of d(t) between two times (finite-time rate)."""
    from scipy.ndimage import maximum_filter1d
    dt = t[1] - t[0]
    env = maximum_filter1d(d, size=max(1, int(round(env_days * DAY / dt))))
    sel = (t >= t_from_yr * YEAR) & (t <= t_to_yr * YEAR) & (env > 0)
    p = np.polyfit(t[sel] / YEAR, np.log(env[sel]), 1)
    return float(p[0])
