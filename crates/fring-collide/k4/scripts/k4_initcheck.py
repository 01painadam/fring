"""K4: are the ST and HS initial states of the same seed identical?

Usage: uv run python scripts/k4_initcheck.py <set> [--runs-dir ../../../runs/k4]
Compares frame 000000.bin (t = 0, before any step) of <set>-st-s<k> and <set>-hs-s<k> byte for byte,
and the six f32 arrays and the ids separately. Also compares the two seeds' ST frames (they must differ).
"""
import os, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("set"); ap.add_argument("--runs-dir", default="../../../runs/k4")
a = ap.parse_args()

def frame(m, s):
    return open(os.path.join(a.runs_dir, f"{a.set}-{m}-s{s}", "frames", "000000.bin"), "rb").read()

for s in sorted(int(d.rsplit('-s', 1)[1]) for d in os.listdir(a.runs_dir) if d.startswith(f'{a.set}-hs-s')):
    st, hs = frame("st", s), frame("hs", s)
    n = int.from_bytes(st[24 + 8:24 + 16], "little")
    hl = int.from_bytes(st[12:16], "little")
    ids_equal = st[hl:hl + 4 * n] == hs[hl:hl + 4 * n]
    arr = [np.array_equal(np.frombuffer(st, "<f4", n, hl + 4 * n * (1 + k)), np.frombuffer(hs, "<f4", n, hl + 4 * n * (1 + k)))
           for k in range(6)]
    print(f"seed {s}: N={n}, whole file identical={st == hs}, header identical={st[:hl] == hs[:hl]}, ids identical={ids_equal}, "
          f"x,y,z,vx,vy,vz identical={arr}")
print("ST seed 1 equals ST seed 2 (expected False):", frame("st", 1) == frame("st", 2))
