"""Compare a Rust trajectory with the V3 REBOUND reference.

Usage (from validation/):
  uv run python scripts/v3_compare.py --rust path/to/rust.csv [--tol-km 1.0]
The Rust file uses the long CSV format of reference/v3_traj_orbits.csv:
  columns k,t_s,id,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s (lines starting with # are ignored).
k is the index on the T0/10 grid (the file may hold only some k; every full orbit is k = 0, 10, ...), id is 0 to 99.
Writes reports/v3_rust_vs_rebound.png and exits 0 if the largest position difference is below the tolerance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compare(rust_csv, tol_km=1.0, plot=True):
    ref = np.load(ROOT / "reference" / "v3_traj.npy")                  # (1001, 100, 6)
    d = np.loadtxt(rust_csv, delimiter=",", comments="#")
    k, ids = d[:, 0].astype(int), d[:, 2].astype(int)
    diff = np.linalg.norm(d[:, 3:6] - ref[k, ids, 0:3], axis=1)
    ks = np.unique(k)
    worst = np.array([diff[k == kk].max() for kk in ks])
    per_particle = np.array([diff[ids == i].max() for i in range(ref.shape[1]) if (ids == i).any()])
    print(f"{len(d)} rows, k {ks.min()}..{ks.max()}: max |dr| = {diff.max():.3e} km (tolerance {tol_km} km); "
          f"worst particle id {ids[diff.argmax()]}; {(per_particle > tol_km).sum()} particles above tolerance")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].semilogy(ks / 10, np.maximum(worst, 1e-12), label="worst particle")
        ax[0].axhline(tol_km, color="k", ls="--", lw=1, label="tolerance")
        ax[0].set(xlabel="orbits", ylabel="|r(Rust) - r(REBOUND)| (km)", title="Position difference")
        ax[0].legend()
        ax[1].hist(np.log10(np.maximum(per_particle, 1e-12)), bins=20)
        ax[1].axvline(np.log10(tol_km), color="k", ls="--")
        ax[1].set(xlabel="log10 of max |dr| over the run (km)", ylabel="particles", title="Per particle")
        fig.tight_layout()
        fig.savefig(ROOT / "reports" / "v3_rust_vs_rebound.png", dpi=120)
    return float(diff.max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", required=True)
    ap.add_argument("--tol-km", type=float, default=1.0)
    a = ap.parse_args()
    sys.exit(0 if compare(a.rust, a.tol_km) < a.tol_km else 1)
