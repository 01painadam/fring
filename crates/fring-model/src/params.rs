//! Loading `params/saturn.toml` and `params/moons.toml` from text.
//!
//! No file I/O: the caller reads the file (or uses `include_str!`) and passes
//! the text, so the same code works in the web build. Unknown tables and fields
//! (`[meta]`, `naif_id`, `radius_km`) are ignored.

use crate::physics::MoonInit;
use crate::{BodyState, Error, Result, SaturnParams, Vec3};
use serde::Deserialize;

#[derive(Deserialize)]
struct SaturnFile {
    saturn: SaturnParams,
}

#[derive(Deserialize)]
struct MoonsFile {
    moon: Vec<MoonEntry>,
}

#[derive(Deserialize)]
struct MoonEntry {
    name: String,
    gm_km3_s2: f64,
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
}

/// Reads the `[saturn]` table of `params/saturn.toml` and checks it
/// ([`SaturnParams::validate`]).
pub fn saturn_from_toml_str(text: &str) -> Result<SaturnParams> {
    let file: SaturnFile =
        toml::from_str(text).map_err(|e| Error::Config(format!("saturn params: {e}")))?;
    file.saturn.validate()?;
    Ok(file.saturn)
}

/// Reads the `[[moon]]` entries of `params/moons.toml`, in file order. The
/// states are the initial conditions at t = 0 (km and km/s, Saturn-centred).
pub fn moons_from_toml_str(text: &str) -> Result<Vec<MoonInit>> {
    let file: MoonsFile =
        toml::from_str(text).map_err(|e| Error::Config(format!("moon params: {e}")))?;
    let finite = |a: &[f64; 3]| a.iter().all(|x| x.is_finite());
    file.moon
        .into_iter()
        .map(|m| {
            if !(m.gm_km3_s2.is_finite()
                && m.gm_km3_s2 > 0.0
                && finite(&m.position_km)
                && finite(&m.velocity_km_s))
            {
                return Err(Error::Config(format!(
                    "moon params: moon {:?} has a non-finite value or GM <= 0",
                    m.name
                )));
            }
            Ok(MoonInit {
                name: m.name,
                gm_km3_s2: m.gm_km3_s2,
                state: BodyState {
                    position_km: Vec3::from(m.position_km),
                    velocity_km_s: Vec3::from(m.velocity_km_s),
                },
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_bad_gm() {
        let text = "[[moon]]\nname='a'\ngm_km3_s2=-1.0\nposition_km=[1.0,0.0,0.0]\nvelocity_km_s=[0.0,1.0,0.0]\n";
        assert!(matches!(moons_from_toml_str(text), Err(Error::Config(_))));
    }

    #[test]
    fn reports_a_missing_field() {
        assert!(matches!(
            moons_from_toml_str("[[moon]]\nname='a'\n"),
            Err(Error::Config(_))
        ));
    }
}
