"""E1-REF stand-in for the Rust WH results: REBOUND WHFast with the same force (NOT the CPU backend).

Usage (from validation/):
  uv run python scripts/e1_standin.py layer1 --div 1  --out ../runs/standin_l1_h1.csv    # h = 356.2537 s ("sub-stepping off")
  uv run python scripts/e1_standin.py layer1 --div 16 --out ../runs/standin_l1_h16.csv   # h / 16 everywhere ("on")
  uv run python scripts/e1_standin.py layer2 --div 1 --out ../runs/standin_l2_h1.csv
Writes the WH result files of the contract in scripts/e1_compare.py.  WHFast is a different code from the Rust
scheme (Jacobi coordinates, the moons are integrated by WHFast too, the J force is a kick).  It shows that the
comparison script and its power check work; it says nothing about the Rust CPU backend.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import rebound

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_ref  # noqa: E402
from fring_ref import encounter as enc  # noqa: E402
from fring_ref.params import load_moons  # noqa: E402
from fring_ref.particles import make_fast_force  # noqa: E402

SAT = e1_ref.SAT


def whfast_sim(moon, states, dt):
    moons = load_moons()
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=SAT.gm)
    for m, s in zip(moons, moon):
        sim.add(m=m.gm, x=s[0], y=s[1], z=s[2], vx=s[3], vy=s[4], vz=s[5])
    for s in states:
        sim.add(m=0.0, x=s[0], y=s[1], z=s[2], vx=s[3], vy=s[4], vz=s[5])
    sim.N_active = 3
    sim.integrator = "whfast"
    sim.dt = dt
    sim.force_is_velocity_dependent = 0
    sim.additional_forces = make_fast_force(SAT)
    return sim


def steps_for(duration_s, h):
    n = int(round(duration_s / h))
    return n, duration_s / n


def _job1(args):
    ids, moon, states, div = args
    n, dt = steps_for(enc.WINDOW_S, enc.STEP_S / div)
    sim = whfast_sim(moon, states, dt)
    sim.exact_finish_time = 0
    sim.steps(n) if hasattr(sim, "steps") else sim.integrate(n * dt)
    return ids, enc.read_states(sim)[2:]


def layer1(div, out, workers):
    st = e1_ref.read_states()
    _, moons = e1_ref.read_moons(e1_ref.REF / "e1_moons_t0.csv")
    tasks = [(ids, moons[st["t0_idx"][ids[0]]], st["state"][ids], div) for ids in e1_ref.chunks(st["t0_idx"])]
    end = np.zeros((len(st["id"]), 6))
    with mp.Pool(workers) as pool:
        for ids, s in pool.imap_unordered(_job1, tasks):
            end[ids] = s
    with open(out, "w") as fh:
        fh.write(f"# STAND-IN, not the Rust CPU backend: REBOUND WHFast, h = {enc.STEP_S}/{div} s (window {enc.WINDOW_S:.0f} s in whole steps)\n")
        fh.write("# columns: id,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s\n")
        for i in st["id"]:
            fh.write(f"{i}," + ",".join(repr(float(v)) for v in end[i]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", choices=["layer1", "layer2"])
    ap.add_argument("--div", type=float, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.layer == "layer1":
        layer1(a.div, a.out, a.workers)
    else:
        from e1_l2 import standin_layer2

        standin_layer2(a.div, a.out, a.workers)
