//! Units, frames and time: the conventions for every `fring` crate.
//!
//! # Units
//!
//! - Length in kilometres (km), time in seconds (s), mass in kilograms (kg).
//! - Gravitational parameters as GM in km³ s⁻². Code does not use G or masses
//!   directly.
//! - Angles in radians in code. Degrees only in config files and plots. Convert
//!   at the boundary with [`deg_to_rad`].
//! - Every physical quantity has its unit in its name (`step_s`,
//!   `position_km`, `gm_km3_s2`) or in its doc comment.
//!
//! The constants in this module are unit definitions, not physical constants.
//! Physical constants come only from `params/` (see `params/sources.md`).
//!
//! # Frames
//!
//! See `docs/PLAN.md` section 4.2.
//!
//! - **Inertial:** origin at Saturn's centre, x-y plane is Saturn's equatorial
//!   plane, z along the spin axis. The x axis is the ascending node of
//!   Saturn's equator on the ICRF equator at the epoch (the JPL Horizons
//!   "body equator" convention used for the moon states in `params/`). The
//!   frame is fixed: Saturn's pole precession (about 0.01° in 30 years) is
//!   ignored. All forces are defined here. All state that crosses the
//!   [`Backend`](crate::Backend) interface is inertial and f64.
//! - **Rotating:** turns about z at the mean motion n₀ of the reference orbit
//!   (semi-major axis a₀, from
//!   [`FrameConfig`](crate::config::FrameConfig)). Used for output, collision
//!   cells and the arc domain.
//! - **Local (guiding-centre):** relative to the reference circular orbit at a₀
//!   in the rotating frame. x is the radial offset, y the azimuthal arc length
//!   offset, z the vertical offset, all in km. Used by the GPU (f32) and by
//!   collision cells.
//!
//! # Time
//!
//! `time_s` is the number of seconds since the epoch of the parameter set. Task
//! L2 states the epoch and its time scale in `params/`. Durations in config
//! files are in Julian years ([`SECONDS_PER_JULIAN_YEAR`]).

/// Seconds in one day (86,400 s).
pub const SECONDS_PER_DAY: f64 = 86_400.0;

/// Days in one Julian year (365.25 d).
pub const DAYS_PER_JULIAN_YEAR: f64 = 365.25;

/// Seconds in one Julian year (31,557,600 s).
pub const SECONDS_PER_JULIAN_YEAR: f64 = SECONDS_PER_DAY * DAYS_PER_JULIAN_YEAR;

/// Converts an angle from degrees to radians.
pub fn deg_to_rad(angle_deg: f64) -> f64 {
    angle_deg.to_radians()
}

/// Converts an angle from radians to degrees.
pub fn rad_to_deg(angle_rad: f64) -> f64 {
    angle_rad.to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn julian_year_in_seconds() {
        assert_eq!(SECONDS_PER_JULIAN_YEAR, 31_557_600.0);
    }

    #[test]
    fn degree_round_trip() {
        assert_eq!(deg_to_rad(180.0), std::f64::consts::PI);
        assert!((rad_to_deg(deg_to_rad(37.5)) - 37.5).abs() < 1e-12);
    }
}
