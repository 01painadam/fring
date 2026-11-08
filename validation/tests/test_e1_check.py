"""E1-REF: the committed layer 1 reference reproduces (two chunks, a few seconds)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e1_ref  # noqa: E402


def test_e1_layer1_check():
    assert e1_ref.check() == 0
