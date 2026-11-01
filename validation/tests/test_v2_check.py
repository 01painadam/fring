"""Recompute the first day of V2 and compare with the committed 6 h reference file."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v2_moons.py"


def test_first_day_matches_committed_reference():
    if not (SCRIPT.parent.parent / "reference" / "v2_moons_6h.npy").exists():
        pytest.skip("reference file not generated yet")
    spec = importlib.util.spec_from_file_location("v2_moons", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dpos, dvel = mod.check()
    assert dpos < 1e-6 and dvel < 1e-12
