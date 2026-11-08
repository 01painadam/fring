"""E1-REF comparison script: limits and power check on synthetic numbers (the reference itself, scaled)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e1_compare as c  # noqa: E402


def test_layer1_limits_and_power():
    ref = c.load_ref1()
    de, dv = ref["de"], ref["dv"]
    # 1 % error everywhere: passes the limits (median < 5 %, p95 < 25 %), has no power
    rows = c.layer1_stats(ref, ref["da"] * 1.01, de, dv)
    assert c.verdict_on(rows) and not c.power_off(rows)
    # 30 % error everywhere: fails the limits, has power
    rows = c.layer1_stats(ref, ref["da"] * 1.30, de, dv)
    assert not c.verdict_on(rows) and c.power_off(rows)
    # stratum 0 alone bad: not a pass criterion
    bad0 = ref["da"] * np.where(ref["stratum"] == 0, 3.0, 1.0)
    rows = c.layer1_stats(ref, bad0, de, dv)
    assert c.verdict_on(rows) and not c.power_off(rows)


def test_layer2_identical_passes():
    ref = c.load_l2(c.REF / "e1_l2_ref.csv") if (c.REF / "e1_l2_ref.csv").exists() else None
    if ref is None:
        return
    start = np.loadtxt(c.REF / "e1_l2_states.csv", delimiter=",", comments="#")[:, 1:7]
    ok, _ = c.layer2_stats(ref, ref, start, 63.3)
    assert ok
