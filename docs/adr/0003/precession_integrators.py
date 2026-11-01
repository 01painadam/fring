import math, numpy as np
GM, R = 37931206.234, 60330.0
J = {2: 1.6290615e-2, 4: -9.35119e-4, 6: 8.6764e-5}
def acc(x, pm=True):
    r2 = x @ x; r = math.sqrt(r2); u = x[2] / r; u2 = u*u; q2 = (R/r)**2
    p = {2: (3*u2-1)/2, 4: ((35*u2-30)*u2+3)/8, 6: (((231*u2-315)*u2+105)*u2-5)/16}
    d = {2: 3*u, 4: (35*u2-15)*u/2, 6: (((1386*u2-1260)*u2+210)*u)/16}
    A = sum((n+1)*J[n]*q2**(n//2)*p[n] for n in J); B = sum(J[n]*q2**(n//2)*d[n] for n in J)
    rad = (1.0 if pm else 0.0) - A - u*B
    k = -GM/(r2*r)
    return k*np.array([rad*x[0], rad*x[1], rad*x[2] + B*r])
def freqs(r):
    q = R/r; W2 = GM/r**3
    n2 = W2*(1 + 1.5*J[2]*q**2 - 15/8*J[4]*q**4 + 35/16*J[6]*q**6)
    k2 = W2*(1 - 1.5*J[2]*q**2 + 45/8*J[4]*q**4 - 175/16*J[6]*q**6)
    return math.sqrt(n2), math.sqrt(k2)
def kepler_drift(x, v, dt):
    r0 = math.sqrt(x@x); a = 1/(2/r0 - (v@v)/GM); nk = math.sqrt(GM/a**3)
    ec, es = 1 - r0/a, (x@v)/math.sqrt(GM*a); M = nk*dt; dE = M
    for _ in range(30):
        f = dE - ec*math.sin(dE) + es*(1-math.cos(dE)) - M
        fp = 1 - ec*math.cos(dE) + es*math.sin(dE); s = f/fp; dE -= s
        if abs(s) < 1e-15: break
    f = 1 - a/r0*(1-math.cos(dE)); g = dt - (dE - math.sin(dE))/nk
    x1 = f*x + g*v; r1 = math.sqrt(x1@x1)
    fd = -math.sqrt(GM*a)*math.sin(dE)/(r1*r0); gd = 1 - a/r1*(1-math.cos(dE))
    return x1, fd*x + gd*v
def lf(x, v, h):
    x = x + v*h/2; v = v + acc(x)*h; return x + v*h/2, v
w1 = 1/(2 - 2**(1/3)); w0 = -2**(1/3)*w1
def y4(x, v, h):
    for w in (w1, w0, w1): x, v = lf(x, v, w*h)
    return x, v
def wh(x, v, h):
    x, v = kepler_drift(x, v, h/2); v = v + acc(x, pm=False)*h; return kepler_drift(x, v, h/2)
def varpi_dot(step, steps_per_orbit, a0=140222.43, orbits=300, e=1e-3):
    n, k = freqs(a0); h = 2*math.pi/n/steps_per_orbit
    x = np.array([a0, 0.0, 0.0]); v = np.array([e*a0*k, a0*n, 0.0])
    ts, ws = [], []; t = 0.0; rv_prev = x@v
    for _ in range(orbits*steps_per_orbit):
        x1, v1 = step(x, v, h); rv = x1@v1
        if rv_prev < 0 <= rv:
            f = -rv_prev/(rv - rv_prev); xp = x + f*(x1 - x)
            ts.append(t + f*h); ws.append(math.atan2(xp[1], xp[0]))
        x, v, t, rv_prev = x1, v1, t + h, rv
    return np.polyfit(ts, np.unwrap(np.array(ws)), 1)[0]
n, k = freqs(140222.43); D = 180/math.pi*86400
print(f"oracle varpi_dot at a0 = {(n-k)*D:.5f} deg/d")
for name, st, sp in (("leapfrog", lf, 150), ("leapfrog", lf, 600), ("Yoshida4 (3 kicks)", y4, 150), ("Wisdom-Holman", wh, 150)):
    w = varpi_dot(st, sp)*D
    print(f"{name:20s} {sp:4d} steps/orbit: {w:.5f} deg/d, error {100*(w/((n-k)*D)-1):+.4f} %")
