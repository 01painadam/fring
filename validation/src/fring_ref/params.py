"""Read Saturn's field from params/saturn.toml (the only source of constants)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Saturn:
    gm: float  # km^3 s^-2
    radius: float  # km, reference radius of the J coefficients
    j2: float
    j4: float
    j6: float


def load_saturn() -> Saturn:
    with open(REPO / "params" / "saturn.toml", "rb") as f:
        s = tomllib.load(f)["saturn"]
    return Saturn(s["gm_km3_s2"], s["reference_radius_km"], s["j2"], s["j4"], s["j6"])


@dataclass(frozen=True)
class Moon:
    name: str
    gm: float  # km^3 s^-2
    pos: tuple  # km, Saturn-centred, Saturn-equator frame
    vel: tuple  # km s^-1


def load_moons() -> list[Moon]:
    with open(REPO / "params" / "moons.toml", "rb") as f:
        t = tomllib.load(f)
    return [Moon(m["name"], m["gm_km3_s2"], tuple(m["position_km"]), tuple(m["velocity_km_s"]))
            for m in t["moon"]]
