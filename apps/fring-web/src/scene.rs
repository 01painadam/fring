//! The particle set-up and the analytic motion that W1 draws.
//!
//! No integration happens here. Each particle follows its first-order
//! epicycle (`fring_model::physics::epicycle_state`, with the exact n, kappa
//! and nu of Saturn's field). The inertial position is turned into the frame
//! that rotates at n0, the mean motion of the F ring core, with
//! `fring_model::physics::inertial_to_rotating`.
//!
//! The scene is plain Rust with no web types, so the unit tests run on the host.
//!
//! Units: km and s. Angles in radians. Time `t_s` is seconds since the epoch of
//! `params/` (2008-01-01 TDB); the rotating frame and the inertial frame
//! coincide at `t_s = 0`.
//!
//! Moons are schematic in W1: each moon is on a circular orbit with the radius
//! of its `params/moons.toml` position and the mean motion of the field at that
//! radius. W2 uses the real ephemeris.

use fring_model::config::Distribution;
use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{
    EpicycleElements, SaturnField, epicycle_state, frames::inertial_to_rotating,
    frames::reference_mean_motion, frequencies,
};
use fring_model::rng::{Counter, Philox, Stream, box_muller};
use fring_model::{Error, Result, Vec3};
use serde::Deserialize;

/// `params/saturn.toml`, `params/moons.toml` and `params/ring.toml`, compiled in.
const SATURN_TOML: &str = include_str!("../../../params/saturn.toml");
const MOONS_TOML: &str = include_str!("../../../params/moons.toml");
const RING_TOML: &str = include_str!("../../../params/ring.toml");

/// Seed of the particle set-up. The page always draws the same particles.
pub const SEED: u64 = 20_081_101;

/// Number of floats in one instance: x, y (km, relative to the eye), size
/// (pixels) and colour id.
pub const FLOATS_PER_INSTANCE: usize = 4;

/// Colour id of a particle.
pub const COLOUR_PARTICLE: f32 = 0.0;
/// Colour id of Saturn.
pub const COLOUR_SATURN: f32 = 1.0;
/// Colour id of the first moon. Moon `k` has `COLOUR_MOON_FIRST + k`.
pub const COLOUR_MOON_FIRST: f32 = 2.0;

/// Size on the screen of a particle, in pixels.
pub const PARTICLE_SIZE_PX: f32 = 2.0;
/// Size on the screen of a moon dot, in pixels.
pub const MOON_SIZE_PX: f32 = 10.0;

/// The values of `params/ring.toml` that W1 uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingParams {
    /// Semi-major axis a0 of the F ring core, km. The rotating frame turns at n(a0).
    pub a0_km: f64,
    /// Epicyclic amplitude e (radial excursion `a e`).
    pub eccentricity: f64,
    /// Longitude of pericentre at the epoch, degrees.
    pub apse_longitude_deg: f64,
    /// Narrowest dust strand of the F ring, km. Used as the display width of the core.
    pub dust_strand_width_min_km: f64,
}

#[derive(Deserialize)]
struct RingFile {
    core: CoreEntry,
    targets: TargetsEntry,
}

#[derive(Deserialize)]
struct CoreEntry {
    semi_major_axis_km: f64,
    eccentricity: f64,
    apse_longitude_deg: f64,
}

#[derive(Deserialize)]
struct TargetsEntry {
    dust_strand_width_min_km: f64,
}

impl RingParams {
    /// Reads the `[core]` and `[targets]` tables of `params/ring.toml`.
    pub fn from_toml_str(text: &str) -> Result<Self> {
        let f: RingFile =
            toml::from_str(text).map_err(|e| Error::Config(format!("ring params: {e}")))?;
        let ok = f.core.semi_major_axis_km > 0.0
            && (0.0..1.0).contains(&f.core.eccentricity)
            && f.targets.dust_strand_width_min_km >= 0.0;
        if !ok {
            return Err(Error::Config("ring params: value out of range".into()));
        }
        Ok(Self {
            a0_km: f.core.semi_major_axis_km,
            eccentricity: f.core.eccentricity,
            apse_longitude_deg: f.core.apse_longitude_deg,
            dust_strand_width_min_km: f.targets.dust_strand_width_min_km,
        })
    }
}

/// The initial band of the particles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// Narrow core: Gaussian in a (display width from the dust strands), with the
    /// eccentricity and apse of the F ring.
    Core,
    /// Circular orbits, uniform in a over `a0 +- a0 e` (the radial excursion of the ring).
    Band,
}

impl Preset {
    /// The `Distribution` (the type of the config schema) of this preset.
    pub fn distribution(self, ring: &RingParams) -> Distribution {
        match self {
            Preset::Core => Distribution::Core {
                semi_major_axis_km: ring.a0_km,
                sigma_semi_major_axis_km: ring.dust_strand_width_min_km,
                eccentricity: ring.eccentricity,
                apse_longitude_deg: ring.apse_longitude_deg,
            },
            Preset::Band => {
                let half = ring.a0_km * ring.eccentricity;
                Distribution::Band {
                    min_semi_major_axis_km: ring.a0_km - half,
                    max_semi_major_axis_km: ring.a0_km + half,
                }
            }
        }
    }
}

/// One moon on a schematic circular orbit.
#[derive(Clone, Debug, PartialEq)]
pub struct MoonDot {
    /// Name, for example "prometheus".
    pub name: String,
    /// Orbit radius, km.
    pub radius_km: f64,
    /// Inertial longitude at t = 0, rad.
    pub longitude0_rad: f64,
    /// Mean motion at `radius_km`, rad/s.
    pub n_rad_s: f64,
}

/// Everything the page draws.
pub struct Scene {
    field: SaturnField<f64>,
    n0_rad_s: f64,
    a0_km: f64,
    saturn_radius_km: f64,
    elements: Vec<EpicycleElements<f64>>,
    moons: Vec<MoonDot>,
}

/// The epicycle elements of particle `id` for `dist`. This follows
/// `fring_model::physics::initial_particle` for `Core` and `Band` (same RNG
/// counters), but keeps the elements, so that the motion can run in time. A test
/// checks that both give the same state at t = 0.
///
/// # Panics
/// If `dist` is `Clump`, which the page does not use.
pub fn elements_for(dist: &Distribution, seed: u64, id: u32) -> EpicycleElements<f64> {
    let rng = Philox::from_seed(seed);
    let counter = |draw: u16| Counter {
        index: id,
        step: 0,
        stream: Stream::INITIAL_CONDITIONS,
        draw,
    };
    let two_pi = std::f64::consts::TAU;
    match *dist {
        Distribution::Core {
            semi_major_axis_km,
            sigma_semi_major_axis_km,
            eccentricity,
            apse_longitude_deg,
        } => {
            let [u1, u2] = rng.uniform_f64(counter(0));
            let (g, _) = box_muller(u1, u2);
            let [u_lambda, _] = rng.uniform_f64(counter(1));
            let lambda = two_pi * u_lambda;
            EpicycleElements {
                a_km: semi_major_axis_km + sigma_semi_major_axis_km * g,
                e: eccentricity,
                i_rad: 0.0,
                mean_anomaly_rad: lambda - apse_longitude_deg.to_radians(),
                z_phase_rad: 0.0,
                mean_longitude_rad: lambda,
            }
        }
        Distribution::Band {
            min_semi_major_axis_km,
            max_semi_major_axis_km,
        } => {
            let [u_a, u_lambda] = rng.uniform_f64(counter(0));
            EpicycleElements {
                a_km: min_semi_major_axis_km
                    + (max_semi_major_axis_km - min_semi_major_axis_km) * u_a,
                e: 0.0,
                i_rad: 0.0,
                mean_anomaly_rad: 0.0,
                z_phase_rad: 0.0,
                mean_longitude_rad: two_pi * u_lambda,
            }
        }
        Distribution::Clump { .. } => panic!("the web scene has no clump preset"),
    }
}

impl Scene {
    /// Builds the scene from the `params/` files compiled into the crate.
    pub fn builtin(preset: Preset, count: u32) -> Result<Self> {
        Self::from_texts(SATURN_TOML, MOONS_TOML, RING_TOML, preset, count, SEED)
    }

    /// Builds the scene from the text of the three `params/` files.
    pub fn from_texts(
        saturn_toml: &str,
        moons_toml: &str,
        ring_toml: &str,
        preset: Preset,
        count: u32,
        seed: u64,
    ) -> Result<Self> {
        let saturn = saturn_from_toml_str(saturn_toml)?;
        let ring = RingParams::from_toml_str(ring_toml)?;
        let field = SaturnField::<f64>::from_params(&saturn);
        let n0_rad_s = reference_mean_motion(&field, ring.a0_km);
        let dist = preset.distribution(&ring);
        let elements = (0..count).map(|id| elements_for(&dist, seed, id)).collect();
        let moons = moons_from_toml_str(moons_toml)?
            .into_iter()
            .map(|m| {
                let p = m.state.position_km;
                // Radius in the equatorial plane: the schematic orbit has z = 0.
                let radius_km = p.x.hypot(p.y);
                MoonDot {
                    name: m.name,
                    radius_km,
                    longitude0_rad: p.y.atan2(p.x),
                    n_rad_s: frequencies(&field, radius_km).n_rad_s,
                }
            })
            .collect();
        Ok(Self {
            field,
            n0_rad_s,
            a0_km: ring.a0_km,
            saturn_radius_km: saturn.reference_radius_km,
            elements,
            moons,
        })
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// True if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// The reference semi-major axis a0, km.
    pub fn a0_km(&self) -> f64 {
        self.a0_km
    }

    /// The mean motion n0 of the rotating frame, rad/s.
    pub fn n0_rad_s(&self) -> f64 {
        self.n0_rad_s
    }

    /// The moons.
    pub fn moons(&self) -> &[MoonDot] {
        &self.moons
    }

    /// Position of particle `i` in the rotating frame at time `t_s`, km.
    pub fn particle_rotating_km(&self, i: usize, t_s: f64) -> Vec3<f64> {
        let mut el = self.elements[i];
        let f = frequencies(&self.field, el.a_km);
        el.mean_anomaly_rad =
            (el.mean_anomaly_rad + f.kappa_rad_s * t_s).rem_euclid(std::f64::consts::TAU);
        el.mean_longitude_rad =
            (el.mean_longitude_rad + f.n_rad_s * t_s).rem_euclid(std::f64::consts::TAU);
        let (pos, vel) = epicycle_state(&self.field, &el);
        inertial_to_rotating(pos, vel, self.n0_rad_s, t_s).0
    }

    /// Position of moon `k` in the rotating frame at time `t_s`, km.
    pub fn moon_rotating_km(&self, k: usize, t_s: f64) -> Vec3<f64> {
        let m = &self.moons[k];
        let lon = (m.longitude0_rad + (m.n_rad_s - self.n0_rad_s) * t_s)
            .rem_euclid(std::f64::consts::TAU);
        Vec3::new(m.radius_km * lon.cos(), m.radius_km * lon.sin(), 0.0)
    }

    /// Fills `out` with the instances to draw: Saturn, the moons, then the
    /// particles; `FLOATS_PER_INSTANCE` floats each (x, y, size in pixels, colour id).
    ///
    /// The positions are in km, rotating frame, **relative to the eye** (the
    /// camera centre). The subtraction is done here in f64 and only then
    /// rounded to f32. So the f32 error is at most 6e-8 times the distance to the
    /// eye. On the screen that is at most about 1e-4 pixel at any zoom, because the
    /// distance in pixels is limited by the size of the window.
    pub fn write_instances(&self, t_s: f64, eye_km: [f64; 2], px_per_km: f64, out: &mut Vec<f32>) {
        out.clear();
        out.reserve((1 + self.moons.len() + self.elements.len()) * FLOATS_PER_INSTANCE);
        let mut push = |p: Vec3<f64>, size_px: f32, colour: f32| {
            out.extend_from_slice(&[
                (p.x - eye_km[0]) as f32,
                (p.y - eye_km[1]) as f32,
                size_px,
                colour,
            ]);
        };
        push(
            Vec3::new(0.0, 0.0, 0.0),
            (2.0 * self.saturn_radius_km * px_per_km) as f32,
            COLOUR_SATURN,
        );
        for k in 0..self.moons.len() {
            push(
                self.moon_rotating_km(k, t_s),
                MOON_SIZE_PX,
                COLOUR_MOON_FIRST + k as f32,
            );
        }
        for i in 0..self.elements.len() {
            push(
                self.particle_rotating_km(i, t_s),
                PARTICLE_SIZE_PX,
                COLOUR_PARTICLE,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fring_model::physics::initial_particle;

    fn scene(preset: Preset, count: u32) -> Scene {
        Scene::builtin(preset, count).unwrap()
    }

    #[test]
    fn ring_params_come_from_the_params_file() {
        let r = RingParams::from_toml_str(RING_TOML).unwrap();
        assert_eq!(r.a0_km, 140222.43);
        assert_eq!(r.eccentricity, 0.00235);
        assert_eq!(r.dust_strand_width_min_km, 10.0);
    }

    #[test]
    fn bad_ring_params_are_an_error() {
        assert!(RingParams::from_toml_str("[core]\nsemi_major_axis_km = 1.0").is_err());
    }

    /// The elements must give the same state at t = 0 as the shared initial
    /// conditions of the CPU runs (same RNG counters).
    #[test]
    fn elements_agree_with_initial_particle_at_t0() {
        let saturn = saturn_from_toml_str(SATURN_TOML).unwrap();
        let field = SaturnField::<f64>::from_params(&saturn);
        let ring = RingParams::from_toml_str(RING_TOML).unwrap();
        for preset in [Preset::Core, Preset::Band] {
            let dist = preset.distribution(&ring);
            for id in 0..200 {
                let (p, v) = epicycle_state(&field, &elements_for(&dist, SEED, id));
                let (p0, v0) = initial_particle(&field, &dist, SEED, id);
                // Same formulas and same inputs: only rounding may differ.
                assert!((p - p0).norm() < 1e-9, "{preset:?} id {id}");
                assert!((v - v0).norm() < 1e-12, "{preset:?} id {id}");
            }
        }
    }

    #[test]
    fn scene_is_reproducible() {
        let a = scene(Preset::Core, 50);
        let b = scene(Preset::Core, 50);
        assert_eq!(a.len(), 50);
        for i in 0..50 {
            assert_eq!(
                a.particle_rotating_km(i, 1.0e6),
                b.particle_rotating_km(i, 1.0e6)
            );
        }
    }

    /// n0 is the exact mean motion at a0 (`reference_mean_motion` uses J2, J4
    /// and J6, the same field as the particles). So a circular orbit at a0 is at
    /// rest in the rotating frame. The only error is rounding: the phase n t is at
    /// most 1.2e-4 rad/s * 1e7 s = 1.2e3 rad, so the phase error is about
    /// 1e-16 * 1.2e3 = 1.3e-13 rad, which is 2e-8 km at 1.4e5 km. The tolerance is 1e-6 km.
    #[test]
    fn circular_orbit_at_a0_is_fixed_in_the_rotating_frame() {
        let mut s = scene(Preset::Band, 1);
        s.elements[0] = EpicycleElements {
            a_km: s.a0_km,
            e: 0.0,
            i_rad: 0.0,
            mean_anomaly_rad: 0.0,
            z_phase_rad: 0.0,
            mean_longitude_rad: 1.0,
        };
        let p0 = s.particle_rotating_km(0, 0.0);
        for t in [1.0e4, 1.0e6, 1.0e7] {
            let p = s.particle_rotating_km(0, t);
            assert!(
                (p - p0).norm() < 1e-6,
                "t = {t} s, moved {} km",
                (p - p0).norm()
            );
        }
    }

    /// A particle 100 km outside a0 falls behind at (n(a) - n0). Kepler gives
    /// dn = -1.5 n da / a. J2 changes the log slope by about 0.3 % (-1.5 to -1.504),
    /// so the tolerance on the drift angle is 1 %.
    #[test]
    fn outer_particle_drifts_backwards_in_the_rotating_frame() {
        let mut s = scene(Preset::Band, 1);
        let da = 100.0;
        s.elements[0] = EpicycleElements {
            a_km: s.a0_km + da,
            e: 0.0,
            i_rad: 0.0,
            mean_anomaly_rad: 0.0,
            z_phase_rad: 0.0,
            mean_longitude_rad: 0.0,
        };
        let t = 86_400.0;
        let p = s.particle_rotating_km(0, t);
        let lon = p.y.atan2(p.x);
        let expected = -1.5 * s.n0_rad_s() * da / s.a0_km() * t;
        assert!(expected < 0.0);
        assert!(
            ((lon - expected) / expected).abs() < 0.01,
            "lon {lon}, expected {expected}"
        );
    }

    #[test]
    fn moons_start_at_their_params_position_and_keep_their_radius() {
        let s = scene(Preset::Core, 1);
        let moons = moons_from_toml_str(MOONS_TOML).unwrap();
        assert_eq!(s.moons().len(), 2);
        for (k, m) in moons.iter().enumerate() {
            let p = s.moon_rotating_km(k, 0.0);
            assert!((p.x - m.state.position_km.x).abs() < 1e-6);
            assert!((p.y - m.state.position_km.y).abs() < 1e-6);
            // The orbit is circular in the equatorial plane; the small z of the params state is not drawn.
            for t in [0.0, 1.0e5, 3.0e7] {
                assert!(
                    (s.moon_rotating_km(k, t).norm()
                        - m.state.position_km.x.hypot(m.state.position_km.y))
                    .abs()
                        < 1e-6
                );
            }
        }
    }

    #[test]
    fn instance_layout() {
        let s = scene(Preset::Core, 10);
        let mut out = Vec::new();
        s.write_instances(0.0, [0.0, 0.0], 1e-3, &mut out);
        assert_eq!(out.len(), (1 + 2 + 10) * FLOATS_PER_INSTANCE);
        assert_eq!(out[3], COLOUR_SATURN);
        assert_eq!(out[7], COLOUR_MOON_FIRST);
        assert_eq!(out[11], COLOUR_MOON_FIRST + 1.0);
        assert_eq!(out[15], COLOUR_PARTICLE);
        // Saturn diameter in pixels: 2 R * scale.
        let r = saturn_from_toml_str(SATURN_TOML)
            .unwrap()
            .reference_radius_km;
        assert!((out[2] as f64 - 2.0 * r * 1e-3).abs() < 1e-3);
    }

    /// The camera-relative f32 positions are exact to a small part of a pixel at
    /// any zoom. The bound: f32 has a relative error of 6e-8, and every
    /// instance in the window is at most about 2000 px from the eye, so the
    /// error is at most 2000 * 6e-8 = 1.2e-4 px. The tolerance is 1e-3 px.
    #[test]
    fn camera_relative_f32_error_is_below_a_pixel_fraction() {
        let s = scene(Preset::Core, 2000);
        let t = 3.0e6;
        let height_px = 1000.0;
        let tracked = s.particle_rotating_km(7, t);
        for half_height_km in [1.8e5, 1.0e4, 300.0, 5.0] {
            let px_per_km = 0.5 * height_px / half_height_km;
            let eye = [tracked.x, tracked.y];
            let mut out = Vec::new();
            s.write_instances(t, eye, px_per_km, &mut out);
            let mut checked = 0;
            for i in 0..s.len() {
                let p = s.particle_rotating_km(i, t);
                let (rx, ry) = (p.x - eye[0], p.y - eye[1]);
                if rx.abs() * px_per_km > 2000.0 || ry.abs() * px_per_km > 2000.0 {
                    continue;
                }
                let base = (3 + i) * FLOATS_PER_INSTANCE;
                let err_px = ((out[base] as f64 - rx).hypot(out[base + 1] as f64 - ry)) * px_per_km;
                assert!(
                    err_px < 1e-3,
                    "half height {half_height_km} km: error {err_px} px"
                );
                checked += 1;
            }
            assert!(
                checked >= 1,
                "no particle in the window at {half_height_km} km"
            );
        }
    }

    /// For comparison, the plain f32 position (no eye subtraction) is much worse at
    /// high zoom: this is why the code subtracts the eye in f64 first.
    #[test]
    fn plain_f32_would_be_too_coarse_at_high_zoom() {
        let s = scene(Preset::Core, 1);
        let p = s.particle_rotating_km(0, 3.0e6);
        let err_km = (p.x as f32 as f64 - p.x)
            .abs()
            .max((p.y as f32 as f64 - p.y).abs());
        let px_per_km = 0.5 * 1000.0 / 5.0;
        // Spacing of f32 near 1.4e5 is 2^-6 km = 0.0156 km, which is 3 px at this zoom.
        assert!(err_km * px_per_km > 1e-3 || err_km == 0.0);
        assert!(p.norm() > 1.0e5);
    }
}
