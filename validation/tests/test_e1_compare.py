"""E1-REF comparison script (ICR-004 amendment 2, part 3): limits for --run, --substep reported only, exit codes."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import e1_compare as c  # noqa: E402


def test_layer1_limits():
    ref = c.load_ref1()
    de, dv = ref["de"], ref["dv"]
    # 1 % error everywhere passes (median < 5 %, p95 < 25 %)
    assert c.verdict(c.layer1_stats(ref, ref["da"] * 1.01, de, dv))
    # 30 % error everywhere fails
    assert not c.verdict(c.layer1_stats(ref, ref["da"] * 1.30, de, dv))
    # stratum 0 alone bad: reported, not a pass criterion
    assert c.verdict(c.layer1_stats(ref, ref["da"] * np.where(ref["stratum"] == 0, 3.0, 1.0), de, dv))
    # a bad stratum 3 alone fails
    assert not c.verdict(c.layer1_stats(ref, ref["da"] * np.where(ref["stratum"] == 3, 1.30, 1.0), de, dv))
    # p95 above 25 % in stratum 3 only, median 0: 10 % of its encounters are 30 % off
    rng = np.random.default_rng(1)
    bad = (ref["stratum"] == 3) & (rng.random(len(ref["da"])) < 0.10)
    rows = c.layer1_stats(ref, ref["da"] * np.where(bad, 1.30, 1.0), de, dv)
    assert rows[3]["p95"] > c.P95_LIMIT and not c.verdict(rows)


def test_no_power_check_left():
    assert not hasattr(c, "power_off") and not hasattr(c, "verdict_on")


def test_empty_stratum_fails():
    ref = c.load_ref1()
    keep = ref["stratum"] != 2
    sub = {k: v[keep] for k, v in ref.items()}
    rows = c.layer1_stats(sub, sub["da"], sub["de"], sub["dv"])
    assert rows[2]["n"] == 0 and np.isnan(rows[2]["median"])
    assert not c.verdict(rows)


def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPTS / "e1_compare.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=SCRIPTS.parent)


def l1_file(path, shift_km):
    """A layer 1 WH file: the start states of every id, moved radially by shift_km (a bad run)."""
    st = np.loadtxt(c.REF / "e1_l1_states.csv", delimiter=",", comments="#")
    out = np.column_stack([st[:, 0], st[:, 4:10]])
    out[:, 1] += shift_km
    np.savetxt(path, out, delimiter=",", header="id,x,y,z,vx,vy,vz")
    return path


def test_cli_layer1_exit_codes(tmp_path):
    bad = l1_file(tmp_path / "bad.csv", 50.0)
    r = run_cli("layer1", "--run", bad)
    assert r.returncode == 1 and "layer 1: FAILED" in r.stdout
    assert "power" not in r.stdout
    # old flags and a missing --run are rejected
    assert run_cli("layer1", "--on", bad).returncode == 2
    assert run_cli("layer1").returncode == 2
    # missing and empty files give exit 2
    assert run_cli("layer1", "--run", tmp_path / "none.csv").returncode == 2
    empty = tmp_path / "empty.csv"
    empty.write_text("# nothing\n")
    assert run_cli("layer1", "--run", empty).returncode == 2
    # --substep is reported only and has its own JSON key
    r = run_cli("layer1", "--run", bad, "--substep", bad, "--json", tmp_path / "o.json")
    assert r.returncode == 1 and "reported only" in r.stdout
    assert set(json.loads((tmp_path / "o.json").read_text())) == {"run", "substep"}


def test_cli_layer2(tmp_path):
    ref = c.REF / "e1_l2_ref.csv"
    r = run_cli("layer2", "--run", ref)
    assert r.returncode == 0 and "layer 2: OK" in r.stdout
    d = np.loadtxt(ref, delimiter=",", comments="#")
    d[:, 1] = 1.0  # every closest approach inside 1 R_H: the counts fail
    bad = tmp_path / "bad.csv"
    np.savetxt(bad, d, delimiter=",")
    assert run_cli("layer2", "--run", bad).returncode == 1
    # a bad --substep with a good --run: exit 0, reported only
    r = run_cli("layer2", "--run", ref, "--substep", bad)
    assert r.returncode == 0 and "reported only" in r.stdout
    # a bad --run with a good --substep: exit 1
    assert run_cli("layer2", "--run", bad, "--substep", ref).returncode == 1


def test_layer2_identical_passes():
    ref = c.load_l2(c.REF / "e1_l2_ref.csv")
    start = np.loadtxt(c.REF / "e1_l2_states.csv", delimiter=",", comments="#")[:, 1:7]
    ok, o = c.layer2_stats(ref, ref, start, 63.3)
    assert ok and "counts" in o["limits"]
