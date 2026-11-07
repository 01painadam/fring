"""A5-ANALYSIS (b): summary of the `fring-analyse closest` runs on eccentric_band.

usage: python3 a5_closest.py <runs_dir> <out.csv>
Reads <runs_dir>/a5-cl-{ecc,circ}-bNN.csv (id, r0, min distance to each moon, class).
Writes, per run, the counts of particles closer than 1, 4.1 and 6.3 Hill radii to
each moon, by class (chaotic or not), with the binomial error sqrt(N p (1 - p)),
and the chaotic fraction with and without an encounter within 6.3 R_H of either moon.
"""
import csv, math, sys
from pathlib import Path

HILL = {"prometheus": 63.40, "pandora": 61.41}  # km, printed by `closest`
LIMITS = [1.0, 4.1, 6.3]
BINS = [0, 1, 2, 10, 20, 21]

def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def berr(k, n):
    p = k / n
    return math.sqrt(n * p * (1 - p))

def main(runs, out):
    rows = []
    for kind in ("ecc", "circ"):
        for b in BINS:
            d = read(Path(runs) / f"a5-cl-{kind}-b{b:02d}.csv")
            n = len(d)
            assert all(r["class"] for r in d), "unmatched class"
            chaotic = [r["class"] == "chaotic" for r in d]
            lo = 140150 + 25 * b
            base = {"kind": kind, "bin": b, "a_lo_km": lo, "n": n, "n_chaotic": sum(chaotic),
                    "frac_chaotic": round(sum(chaotic) / n, 4)}
            for moon, hill in HILL.items():
                for k in LIMITS:
                    inside = [float(r[f"min_{moon}_km"]) < k * hill for r in d]
                    nin = sum(inside)
                    nin_ch = sum(i and c for i, c in zip(inside, chaotic))
                    nin_no = nin - nin_ch
                    base[f"{moon}_{k}RH_n"] = nin
                    base[f"{moon}_{k}RH_err"] = round(berr(nin, n), 1)
                    base[f"{moon}_{k}RH_chaotic"] = nin_ch
                    base[f"{moon}_{k}RH_other"] = nin_no
            enc = [any(float(r[f"min_{m}_km"]) < 6.3 * h for m, h in HILL.items()) for r in d]
            ne = sum(enc)
            ne_ch = sum(e and c for e, c in zip(enc, chaotic))
            nq = n - ne
            nq_ch = sum((not e) and c for e, c in zip(enc, chaotic))
            base.update({
                "enc6.3_n": ne, "enc6.3_chaotic": ne_ch,
                "enc6.3_frac_chaotic": round(ne_ch / ne, 4) if ne else "",
                "noenc6.3_n": nq, "noenc6.3_chaotic": nq_ch,
                "noenc6.3_frac_chaotic": round(nq_ch / nq, 4) if nq else "",
                "noenc6.3_err": round(math.sqrt(nq_ch / nq * (1 - nq_ch / nq) / nq), 4) if nq else "",
            })
            rows.append(base)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
