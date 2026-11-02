//! Loading `params/saturn.toml`, `params/moons.toml` and `params/ring.toml` from text.
//!
//! No file I/O: the caller reads the file (or uses `include_str!`) and passes
//! the text, so the same code works in the web build. Unknown tables and fields
//! (`[meta]`, `naif_id`, and `radius_km` except in [`moon_radii_from_toml_str`])
//! are ignored.

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
struct RingFile {
    core: RingCore,
    targets: RingTargets,
}

#[derive(Deserialize)]
struct RingCore {
    semi_major_axis_km: f64,
    eccentricity: f64,
    apse_longitude_deg: f64,
    apse_rate_deg_per_day: f64,
    inclination_deg: f64,
}

#[derive(Deserialize)]
struct RingTargets {
    dust_strand_width_min_km: f64,
    dust_strand_width_max_km: f64,
}

/// The F ring values of `params/ring.toml` that the code uses. Degrees as in
/// the file (the unit is in each name); convert to radians where you compute.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingParams {
    /// Semi-major axis a0 of the F ring core, km. The rotating frame turns at n(a0).
    pub semi_major_axis_km: f64,
    /// Eccentricity of the core.
    pub eccentricity: f64,
    /// Longitude of pericentre at the epoch, degrees. UNVERIFIED in
    /// `params/sources.md`: good to a few degrees at best.
    pub apse_longitude_deg: f64,
    /// Apse precession rate, degrees per day.
    pub apse_rate_deg_per_day: f64,
    /// Inclination of the core, degrees.
    pub inclination_deg: f64,
    /// Narrowest dust strand seen in images, km (a target, not an orbit value).
    pub dust_strand_width_min_km: f64,
    /// Widest dust strand seen in images, km.
    pub dust_strand_width_max_km: f64,
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

#[derive(Deserialize)]
struct MoonRadiiFile {
    moon: Vec<MoonRadiusEntry>,
}

#[derive(Deserialize)]
struct MoonRadiusEntry {
    name: String,
    radius_km: f64,
}

/// Reads the name and mean radius (km) of each `[[moon]]` entry of
/// `params/moons.toml`, in file order. Checks that each radius is finite and
/// positive. The orbit code does not use the radii; the web uses them for the
/// drop rule "inside a moon" (ICR-006, `GpuStepper::set_moon_radii_km`). Match
/// them to the moon tracks by name, not by position.
pub fn moon_radii_from_toml_str(text: &str) -> Result<Vec<(String, f64)>> {
    let file: MoonRadiiFile =
        toml::from_str(text).map_err(|e| Error::Config(format!("moon radii: {e}")))?;
    file.moon
        .into_iter()
        .map(|m| {
            if !(m.radius_km.is_finite() && m.radius_km > 0.0) {
                return Err(Error::Config(format!(
                    "moon radii: moon {:?} has radius_km {} (must be finite and > 0)",
                    m.name, m.radius_km
                )));
            }
            Ok((m.name, m.radius_km))
        })
        .collect()
}

/// Reads the `[core]` and `[targets]` tables of `params/ring.toml`. Checks that
/// every value is finite, a0 > 0, 0 <= e < 1, and 0 <= min width <= max width.
pub fn ring_from_toml_str(text: &str) -> Result<RingParams> {
    let f: RingFile =
        toml::from_str(text).map_err(|e| Error::Config(format!("ring params: {e}")))?;
    let r = RingParams {
        semi_major_axis_km: f.core.semi_major_axis_km,
        eccentricity: f.core.eccentricity,
        apse_longitude_deg: f.core.apse_longitude_deg,
        apse_rate_deg_per_day: f.core.apse_rate_deg_per_day,
        inclination_deg: f.core.inclination_deg,
        dust_strand_width_min_km: f.targets.dust_strand_width_min_km,
        dust_strand_width_max_km: f.targets.dust_strand_width_max_km,
    };
    let all_finite = [
        r.semi_major_axis_km,
        r.eccentricity,
        r.apse_longitude_deg,
        r.apse_rate_deg_per_day,
        r.inclination_deg,
        r.dust_strand_width_min_km,
        r.dust_strand_width_max_km,
    ]
    .iter()
    .all(|x| x.is_finite());
    let ok = all_finite
        && r.semi_major_axis_km > 0.0
        && (0.0..1.0).contains(&r.eccentricity)
        && (0.0..=r.dust_strand_width_max_km).contains(&r.dust_strand_width_min_km);
    if !ok {
        return Err(Error::Config("ring params: value out of range".into()));
    }
    Ok(r)
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

    const MOONS_TOML: &str = include_str!("../../../params/moons.toml");

    #[test]
    fn reads_the_moon_radii_in_file_order() {
        let r = moon_radii_from_toml_str(MOONS_TOML).unwrap();
        assert_eq!(
            r,
            vec![
                ("prometheus".to_string(), 43.09),
                ("pandora".to_string(), 40.63)
            ]
        );
        // The orbit loader still reads the same file (it ignores radius_km).
        assert_eq!(moons_from_toml_str(MOONS_TOML).unwrap().len(), 2);
    }

    #[test]
    fn rejects_a_bad_or_missing_moon_radius() {
        let bad = MOONS_TOML.replace("radius_km = 43.09", "radius_km = -1.0");
        assert!(matches!(
            moon_radii_from_toml_str(&bad),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            moon_radii_from_toml_str("[[moon]]\nname='a'\n"),
            Err(Error::Config(_))
        ));
    }

    const RING_TOML: &str = include_str!("../../../params/ring.toml");

    #[test]
    fn reads_the_ring_params_file() {
        let r = ring_from_toml_str(RING_TOML).unwrap();
        assert_eq!(r.semi_major_axis_km, 140222.43);
        assert_eq!(r.eccentricity, 0.00235);
        assert_eq!(r.apse_longitude_deg, 353.78);
        assert_eq!(r.apse_rate_deg_per_day, 2.70025);
        assert_eq!(r.inclination_deg, 0.0065);
        assert_eq!(r.dust_strand_width_min_km, 10.0);
        assert_eq!(r.dust_strand_width_max_km, 30.0);
    }

    #[test]
    fn rejects_a_bad_ring_value() {
        let bad_e = RING_TOML.replace("eccentricity = 0.00235", "eccentricity = 1.5");
        assert!(matches!(ring_from_toml_str(&bad_e), Err(Error::Config(_))));
        let bad_w = RING_TOML.replace(
            "dust_strand_width_min_km = 10.0",
            "dust_strand_width_min_km = 40.0",
        );
        assert!(matches!(ring_from_toml_str(&bad_w), Err(Error::Config(_))));
    }

    #[test]
    fn reports_a_missing_ring_table() {
        assert!(matches!(
            ring_from_toml_str("[core]\nsemi_major_axis_km = 1.0\n"),
            Err(Error::Config(_))
        ));
    }
}
