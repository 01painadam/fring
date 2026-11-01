"""Minimax coefficients for the f32 sine and cosine of fring-gpu (ADR 0006).

Method: the Remez exchange algorithm (E. Remez 1934; see for example
J.-M. Muller, "Elementary Functions", 3rd ed., chapter 3) for a weighted
polynomial approximation in t = x^2 on [0, (pi/4)^2]:

- sin x = x + x^3 P(t), P of degree 2, minimising the relative error of sin;
- cos x = 1 - t/2 + t^2 Q(t), Q of degree 2, minimising the absolute error.

The coefficients are printed as f32 values (rounded to nearest) with the
maximum error of the rounded polynomial, evaluated in f64 on a dense grid.
Run: uv run --no-project --with numpy python3 docs/adr/0006/remez_sincos.py
"""

import numpy as np

T_MAX = (np.pi / 4) ** 2


def remez(g, w, deg, n_iter=30):
    """Minimax p (degree deg) of max |w(t) (p(t) - g(t))| on [0, T_MAX]."""
    n = deg + 2
    # Chebyshev nodes as the first reference.
    k = np.arange(n)
    ref = 0.5 * T_MAX * (1 - np.cos(np.pi * k / (n - 1)))
    grid = np.linspace(0.0, T_MAX, 200001)
    for _ in range(n_iter):
        a = np.zeros((n, n))
        for j in range(deg + 1):
            a[:, j] = ref**j
        a[:, deg + 1] = (-1.0) ** k / w(ref)
        sol = np.linalg.solve(a, g(ref))
        c = sol[: deg + 1]
        err = w(grid) * (np.polyval(c[::-1], grid) - g(grid))
        # New reference: one extremum per sign change interval.
        sign = np.sign(err)
        idx = np.concatenate(([0], np.nonzero(np.diff(sign))[0] + 1, [len(grid)]))
        ext = [s + np.argmax(np.abs(err[s:e])) for s, e in zip(idx[:-1], idx[1:])]
        if len(ext) != n:
            break
        ref = grid[ext]
    return c


def sin_g(t):
    x = np.sqrt(t)
    out = np.empty_like(t)
    small = x < 1e-3
    out[small] = -1 / 6 + t[small] / 120 - t[small] ** 2 / 5040
    xs = x[~small]
    out[~small] = (np.sin(xs) - xs) / xs**3
    return out


def sin_w(t):
    # relative error of sin: x^3 / sin x
    x = np.sqrt(t)
    out = np.ones_like(t)
    nz = x > 0
    out[nz] = x[nz] ** 3 / np.sin(x[nz])
    out[~nz] = 0.0
    return np.maximum(out, 1e-300)


def cos_g(t):
    x = np.sqrt(t)
    out = np.empty_like(t)
    small = x < 1e-3
    out[small] = 1 / 24 - t[small] / 720 + t[small] ** 2 / 40320
    xs = x[~small]
    out[~small] = (np.cos(xs) - 1 + xs**2 / 2) / xs**4
    return out


def cos_w(t):
    return np.maximum(t**2, 1e-300)


ps = np.float32(remez(sin_g, sin_w, 2))
pc = np.float32(remez(cos_g, cos_w, 2))
x = np.linspace(-np.pi / 4, np.pi / 4, 400001)
x = x[x != 0]
t = x * x
s = x + x**3 * np.polyval(ps[::-1].astype(np.float64), t)
c = 1 - t / 2 + t**2 * np.polyval(pc[::-1].astype(np.float64), t)
print("sin coefficients (t^0, t^1, t^2):", [f"{v:.9e}" for v in ps])
print("cos coefficients (t^0, t^1, t^2):", [f"{v:.9e}" for v in pc])
print("f32 bits sin:", [hex(v) for v in ps.view(np.uint32)])
print("f32 bits cos:", [hex(v) for v in pc.view(np.uint32)])
print(f"max relative error of sin: {np.max(np.abs(s / np.sin(x) - 1)):.3e}")
print(f"max absolute error of cos: {np.max(np.abs(c - np.cos(x))):.3e}")
