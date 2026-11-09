"""K4: reduce one run directory (frames with velocities + stats.csv) to a time series CSV.

Usage: uv run python scripts/k4_reduce.py <rundir> <out.csv>
Frame format: docs/snapshot-format.md section 5 (rotating frame, f32).
Columns: orbit, c_rho, c_phi, c_z (km/s), sigma2_km2 (variance of rho minus c_rho^2/kappa^2),
n (particles), n0_rad_s (mean motion), and from the stats.csv rows with steps_total in (previous frame, this frame]:
coll (accepted pairs), loss (sum of energy_loss_km2_s2), deep, gmax_exceeded.
"""
import sys, struct, csv, glob, os
import numpy as np

QUANT_KM = 2.0 ** -6   # f32 spacing at rho = 1.4e5 km (131072 <= rho < 262144: 2^17 * 2^-23 = 2^-6)

def read_frame(path):
    b = open(path, "rb").read()
    assert b[:8] == b"FRINGFRM"
    ver, hl, ft, flags = struct.unpack_from("<IIII", b, 8)
    step, n = struct.unpack_from("<QQ", b, 24)
    t, a0, n0, tref = struct.unpack_from("<dddd", b, 40)
    assert flags & 1
    o = hl
    ids = np.frombuffer(b, "<u4", n, o); o += 4 * n
    arr = [np.frombuffer(b, "<f4", n, o + 4 * n * k).astype(np.float64) for k in range(6)]
    return step, t, n0, a0, arr

def stats_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def main():
    rundir, out = sys.argv[1], sys.argv[2]
    rows = stats_rows(os.path.join(rundir, "stats.csv"))
    files = sorted(glob.glob(os.path.join(rundir, "frames", "*.bin")), key=lambda p: int(os.path.basename(p)[:-4]))
    res = []
    prev_step = 0
    for p in files:
        step, t, n0, a0, (x, y, z, vx, vy, vz) = read_frame(p)
        rho = np.hypot(x, y); ph = np.arctan2(y, x)
        vr = vx * np.cos(ph) + vy * np.sin(ph)
        vp = -vx * np.sin(ph) + vy * np.cos(ph)
        c_rho2 = np.mean((vr - vr.mean()) ** 2)
        c_z2 = np.mean((vz - vz.mean()) ** 2)
        # c_phi: per 0.2 km bin of rho, remove a linear mean flow, residual variance; correct for f32 rounding of rho
        corr = (1.5 * n0) ** 2 * QUANT_KM ** 2 / 12.0
        num = 0.0; cnt = 0
        b = np.floor((rho - rho.min()) / 0.2).astype(int)
        for k in np.unique(b):
            m = b == k
            if m.sum() < 50: continue
            r = rho[m]; v = vp[m]
            dr = r - r.mean(); dv = v - v.mean()
            slope = np.dot(dr, dv) / np.dot(dr, dr)   # closed form; avoids the spurious matmul warnings of numpy+Accelerate
            res_ = dv - slope * dr
            num += np.sum(res_ ** 2) - m.sum() * corr; cnt += m.sum()
        c_phi2 = num / cnt
        kappa = n0
        s2 = np.var(rho) - c_rho2 / kappa ** 2
        # stats rows in (prev_step, step]
        sel = [r for r in rows if prev_step < int(r["steps_total"]) <= step]
        nrows = len(sel)
        coll = sum(int(r["collisions_since_last"]) for r in sel)
        loss = sum(float(r.get("energy_loss_km2_s2", 0) or 0) for r in sel)
        deep = sum(int(r.get("deep_collisions", 0) or 0) for r in sel)
        gx = sum(int(r.get("gmax_exceeded", 0) or 0) for r in sel)
        res.append(dict(step=step, t_s=t, orbit=t * n0 / (2 * np.pi), n=len(x), n0_rad_s=n0,
                        c_rho=np.sqrt(c_rho2), c_phi=np.sqrt(max(c_phi2, 0)), c_z=np.sqrt(c_z2),
                        sigma2_km2=s2, coll=coll, loss=loss, deep=deep, gmax_exceeded=gx, dsteps=step - prev_step))
        prev_step = step
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader(); w.writerows(res)

main()
