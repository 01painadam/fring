#!/usr/bin/env python3
"""Derive n^2, kappa^2, nu^2 for the zonal potential of PLAN section 7.1.

Needs only sympy.  Run:  python3 docs/literature/derivations/n_kappa_nu.py

Part 1 derives the frequencies of near-circular equatorial orbits from the
potential Phi(r, z) = -(GM/r) [1 - sum_n J_n (R/r)^n P_n(z/r)] (spherical r),
with n = 2, 4 for the expansions of PLAN 7.2 and n = 2..12 for the numerical table.
Part 2 prints the apse precession varpi_dot = n - kappa at the core with the
parameters of params/saturn.toml and params/ring.toml (copied below; change them
here if the files change).

Definitions (cylindrical radius r in the equatorial plane):
  n^2     = (1/r) dPhi/dr
  kappa^2 = d^2Phi/dr^2 + (3/r) dPhi/dr        (= r^-3 d(r^4 n^2)/dr)
  nu^2    = d^2Phi/dz^2
"""

import math

import sympy as sp

r, z, GM, R = sp.symbols("r z GM R", positive=True)
J = {n: sp.Symbol(f"J{n}") for n in range(2, 13, 2)}


def potential(jmax):
    rho = sp.sqrt(r**2 + z**2)
    u = z / rho
    s = 1 - sum(J[n] * (R / rho) ** n * sp.legendre(n, u) for n in range(2, jmax + 1, 2))
    return -GM / rho * s


def frequencies_squared(jmax):
    phi = potential(jmax)
    n2 = (sp.diff(phi, r) / r).subs(z, 0)
    k2 = (sp.diff(phi, r, 2) + 3 * sp.diff(phi, r) / r).subs(z, 0)
    nu2 = sp.diff(phi, z, 2).subs(z, 0)
    return [sp.simplify(x) for x in (n2, k2, nu2)]


# ---- Part 1: symbolic expansion, J2 and J4 ----
q = sp.symbols("q", positive=True)  # q = R / r
omega2 = GM / r**3
print("Expansions in q = R/r, divided by Omega_K^2 = GM/r^3 (J2 and J4 only):")
for name, expr in zip(("n^2", "kappa^2", "nu^2"), frequencies_squared(4)):
    coeffs = sp.collect(sp.expand(sp.simplify((expr / omega2).subs(R, q * r))), q)
    print(f"  {name:8s}/Omega_K^2 = {coeffs}")

# ---- Part 2: numbers (parameters as in params/*.toml) ----
GM_VAL = 37931206.234  # km^3 s^-2
R_VAL = 60330.0  # km
J_VALS = {2: 1.6290615e-2, 4: -9.35119e-4, 6: 8.6764e-5, 8: -1.3931e-5, 10: 5.426e-6, 12: -5.37e-7}
A_CORE = 140222.43  # km
RAD_DAY = 180.0 / math.pi * 86400.0  # rad/s -> deg/day
OBSERVED = 2.70025  # deg/day, Albers et al. 2012 via Hedman 2017
OBSERVED_ERR = 0.00029


def numeric(jmax):
    subs = {GM: GM_VAL, R: R_VAL}
    subs.update({J[n]: (J_VALS[n] if n <= jmax else 0.0) for n in J})
    fn = [sp.lambdify(r, e.subs(subs), "math") for e in frequencies_squared(jmax)]
    return fn


print()
print(f"At a = {A_CORE} km (GM = {GM_VAL}, R = {R_VAL}):")
print("  Jmax   n[deg/d]      kappa[deg/d]   varpi_dot[deg/d]  minus observed   node_dot[deg/d]")
for jmax in (2, 4, 6, 8, 12):
    f = numeric(jmax)
    n = math.sqrt(f[0](A_CORE)) * RAD_DAY
    k = math.sqrt(f[1](A_CORE)) * RAD_DAY
    nu = math.sqrt(f[2](A_CORE)) * RAD_DAY
    print(
        f"  {jmax:4d}  {n:12.6f}  {k:12.6f}  {n - k:14.6f}  {n - k - OBSERVED:+14.6f}  {-(nu - n):14.6f}"
    )

f4, f12 = numeric(4), numeric(12)
fj6 = numeric(6)
n4 = math.sqrt(f4[0](A_CORE)) * RAD_DAY
n6 = math.sqrt(fj6[0](A_CORE)) * RAD_DAY
n12 = math.sqrt(f12[0](A_CORE)) * RAD_DAY
for label, nj in (("J6", n6), ("J6..J12", n12)):
    dn = nj - n4
    da = 2.0 / 3.0 * A_CORE * dn / n4
    print(f"  Effect of {label} on n: {dn:+.3e} deg/d = {dn / n4:+.2e} relative; a shifts by {da:.3f} km")

days_30yr = 30 * 365.25
for jmax, fj in ((4, f4), (12, f12)):
    nn = math.sqrt(fj[0](A_CORE)) * RAD_DAY
    kk = math.sqrt(fj[1](A_CORE)) * RAD_DAY
    gap = OBSERVED - (nn - kk)
    print(
        f"  Jmax {jmax}: observed minus computed = {gap:+.5f} deg/d ({gap / OBSERVED_ERR:.1f} sigma); "
        f"over 30 yr = {gap * days_30yr:+.1f} deg"
    )

# ---- Corotation resonance check: n_F = n_P - kappa_P / m, m = 110 (Cuzzi et al. 2024) ----
N_PROM = 587.285415  # deg/day, Cuzzi et al. 2024
M = 110
for jmax, fj in ((4, f4), (12, f12)):
    # find a_P with n(a_P) = N_PROM
    a = 139_400.0
    for _ in range(60):
        n_a = math.sqrt(fj[0](a)) * RAD_DAY
        a *= (n_a / N_PROM) ** (2.0 / 3.0)
    kappa_p = math.sqrt(fj[1](a)) * RAD_DAY
    n_f = N_PROM - kappa_p / M
    # a at which n(a) = n_f
    b = A_CORE
    for _ in range(60):
        n_b = math.sqrt(fj[0](b)) * RAD_DAY
        b *= (n_b / n_f) ** (2.0 / 3.0)
    print(
        f"  Jmax {jmax}: Prometheus mean-motion radius {a:.2f} km, kappa_P = {kappa_p:.6f} deg/d, "
        f"n_F = {n_f:.6f} deg/d, CER radius {b:.3f} km (Cuzzi 2024: {A_CORE} km)"
    )
