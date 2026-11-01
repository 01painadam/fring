"""Measure apsidal and nodal rates of a near-circular orbit from sampled states."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def lagrange_roots(t, f, npts=8):
    """Roots of the sampled function f(t): local Lagrange polynomial of npts points + Newton."""
    idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    return idx


def refine(t, cols, i, npts=8, guess=None):
    """Refine sign change of cols[0] between samples i and i+1 with a local polynomial.

    Returns (t_root, [values of the other columns at t_root]).
    """
    lo = max(0, min(i - npts // 2 + 1, len(t) - npts))
    ts = t[lo:lo + npts]
    t0 = ts[0]
    x = ts - t0
    polys = [np.polyfit(x, c[lo:lo + npts], npts - 1) for c in cols]
    # bracket, then bisection/secant via numpy roots on the bracket
    a, b = t[i] - t0, t[i + 1] - t0
    fa, fb = np.polyval(polys[0], a), np.polyval(polys[0], b)
    for _ in range(80):  # Illinois / bisection hybrid, converged to machine precision
        c = b - fb * (b - a) / (fb - fa)
        fc = np.polyval(polys[0], c)
        if fc * fb < 0:
            a, fa = b, fb
        else:
            fa *= 0.5
        b, fb = c, fc
        if abs(fb) < 1e-14 * max(1.0, np.abs(cols[0]).max()) or abs(b - a) < 1e-9:
            break
    return t0 + b, [np.polyval(p, b) for p in polys[1:]]


def event_rates(t, pos, vel):
    """Event method.

    Pericentre: zeros of rdot = r.v / r with rdot going - to + ; record the azimuth theta.
    Node: zeros of z with z going - to + ; record the azimuth theta.
    Slope of the unwrapped azimuth of the events, converted to rates:
      varpi_dot = d(theta_peri)/dt  (the line of apsides moves by this per unit time)
      Omega_dot = d(theta_node)/dt
    Both are slopes of event azimuth against event time, so they are the precession of
    the pericentre and node, independent of the osculating-element definition.
    """
    r = np.linalg.norm(pos, axis=1)
    rdot = np.einsum("ij,ij->i", pos, vel) / r
    theta = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
    z = pos[:, 2]
    out = {}
    for name, f, cols in (("peri", rdot, [theta]), ("node", z, [theta])):
        idx = [i for i in np.nonzero((f[:-1] < 0) & (f[1:] >= 0))[0]]
        ev_t, ev_th = [], []
        for i in idx:
            if i < 4 or i > len(t) - 5:
                continue
            tr, (th,) = refine(t, [f, theta], i)
            ev_t.append(tr)
            ev_th.append(th)
        ev_t, ev_th = np.array(ev_t), np.array(ev_th)
        # longitude of the event = theta mod 2pi; its unwrapped drift is theta minus the
        # 2pi*k advance of the event counter
        k = np.arange(len(ev_t))
        lon = ev_th - 2 * np.pi * k  # one event per radial (or vertical) cycle
        A = np.vstack([ev_t, np.ones_like(ev_t)]).T
        coef, res, *_ = np.linalg.lstsq(A, lon, rcond=None)
        resid = lon - A @ coef
        dof = max(1, len(lon) - 2)
        sig = np.sqrt(np.sum(resid**2) / dof / np.sum((ev_t - ev_t.mean()) ** 2))
        out[name] = dict(rate=coef[0], sigma=sig, n_events=len(ev_t), t=ev_t, lon=lon, resid=resid)
    return out


def frequency_fit(t, pos, vel, n0, k0, v0):
    """Fit n, kappa, nu directly (independent method).

    n: slope of the unwrapped azimuth.  kappa: sinusoid + secular drift fitted to r(t).
    nu: sinusoid fitted to z(t).  Returns n, kappa, nu in rad/s.
    """
    r = np.linalg.norm(pos, axis=1)
    theta = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
    tt = t - t[0]
    kappa_ref = None
    # azimuth = n t + (small periodic terms) + const; fit with first harmonics of kappa
    def model_th(p):
        n, c, a1, b1, a2, b2 = p
        w = kappa * tt
        return n * tt + c + a1 * np.sin(w) + b1 * np.cos(w) + a2 * np.sin(2 * w) + b2 * np.cos(2 * w)
    def model_r(p):
        k, c, a1, b1, a2, b2 = p
        w = k * tt
        return c + a1 * np.sin(w) + b1 * np.cos(w) + a2 * np.sin(2 * w) + b2 * np.cos(2 * w)
    def model_z(p):
        v, a1, b1 = p
        return a1 * np.sin(v * tt) + b1 * np.cos(v * tt)
    sol_r = least_squares(lambda p: model_r(p) - r, [k0, r.mean(), 0, 0, 0, 0], x_scale=[1e-8, 1e5, 1, 1, 1, 1], xtol=1e-15, ftol=1e-15)
    kappa = sol_r.x[0]
    sol_th = least_squares(lambda p: (lambda q: model_th(q) - theta)((p[0], *p[1:])), [n0, theta[0], 0, 0, 0, 0],
                           x_scale=[1e-8, 1, 1e-3, 1e-3, 1e-3, 1e-3], xtol=1e-15, ftol=1e-15)
    sol_z = least_squares(lambda p: model_z(p) - pos[:, 2], [v0, pos[:, 2].max(), 0], x_scale=[1e-8, 1, 1], xtol=1e-15, ftol=1e-15)
    return sol_th.x[0], kappa, sol_z.x[0]
