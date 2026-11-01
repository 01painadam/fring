//! `params/saturn.toml` (task L2, ICR-001) loads into `SaturnParams`.
//!
//! `SaturnParams` has `deny_unknown_fields`, so this test fails if the
//! `[saturn]` table gains or loses a field without a matching code change.

use fring_model::SaturnParams;
use serde::Deserialize;

/// The parts of `params/saturn.toml` that code reads. Other top-level tables
/// (for example `[meta]`) are ignored.
#[derive(Deserialize)]
struct SaturnFile {
    saturn: SaturnParams,
}

const SATURN_TOML: &str = include_str!("../../../params/saturn.toml");

#[test]
fn saturn_toml_loads_into_saturn_params() {
    let file: SaturnFile = toml::from_str(SATURN_TOML).expect("params/saturn.toml must parse");
    let s = file.saturn;
    s.validate().unwrap();
    // Jacobson 2022 (SAT441), see params/sources.md.
    assert_eq!(s.gm_km3_s2, 37_931_206.234);
    assert_eq!(s.reference_radius_km, 60_330.0);
    assert_eq!(s.j2, 1.6290615e-2);
    assert_eq!(s.j4, -9.35119e-4);
    assert_eq!(s.j6, 8.6764e-5);
}
