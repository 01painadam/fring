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
//!
//! Three kinds of view (`ViewFrame`): inertial (fixed camera on Saturn), rotating
//! (the rotating frame itself: the camera turns with the core) and a moon view
//! (centred on one moon and turning with it).

use crate::view::ViewTransform;
use fring_model::config::Distribution;
use fring_model::params::{
    RingParams, moons_from_toml_str, ring_from_toml_str, saturn_from_toml_str,
};
use fring_model::physics::{
    EpicycleElements, SaturnField, epicycle_state, frames::inertial_to_rotating,
    frames::reference_mean_motion, frequencies,
};
use fring_model::rng::{Counter, Philox, Stream, box_muller};
use fring_model::{Result, Vec3};

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

/// Writes the floats as little-endian bytes into `out` (the layout of a GPU
/// buffer on every WebGPU device). Replaces the old contents of `out`.
pub fn floats_to_le_bytes(data: &[f32], out: &mut Vec<u8>) {
    out.clear();
    out.extend(data.iter().flat_map(|x| x.to_le_bytes()));
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
                semi_major_axis_km: ring.semi_major_axis_km,
                sigma_semi_major_axis_km: ring.dust_strand_width_min_km,
                eccentricity: ring.eccentricity,
                apse_longitude_deg: ring.apse_longitude_deg,
            },
            Preset::Band => {
                let half = ring.semi_major_axis_km * ring.eccentricity;
                Distribution::Band {
                    min_semi_major_axis_km: ring.semi_major_axis_km - half,
                    max_semi_major_axis_km: ring.semi_major_axis_km + half,
                }
            }
        }
    }
}

/// What the view follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewFrame {
    /// Fixed camera, centred on Saturn: everything orbits anticlockwise.
    Inertial,
    /// The frame that turns at n0, centred on Saturn: the camera turns with the
    /// ring core, so the core stays still.
    Rotating,
    /// Centred on moon `k` (index into `Scene::moons`) and turning with it:
    /// Saturn is straight down and the orbital motion goes to the left.
    Moon(usize),
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
        let ring = ring_from_toml_str(ring_toml)?;
        let field = SaturnField::<f64>::from_params(&saturn);
        let n0_rad_s = reference_mean_motion(&field, ring.semi_major_axis_km);
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
            a0_km: ring.semi_major_axis_km,
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

    /// Inertial position (km) and velocity (km/s) of particle `i` at time `t_s`.
    fn particle_inertial_state(&self, i: usize, t_s: f64) -> (Vec3<f64>, Vec3<f64>) {
        let mut el = self.elements[i];
        let f = frequencies(&self.field, el.a_km);
        el.mean_anomaly_rad =
            (el.mean_anomaly_rad + f.kappa_rad_s * t_s).rem_euclid(std::f64::consts::TAU);
        el.mean_longitude_rad =
            (el.mean_longitude_rad + f.n_rad_s * t_s).rem_euclid(std::f64::consts::TAU);
        epicycle_state(&self.field, &el)
    }

    /// Position of particle `i` in the inertial frame at time `t_s`, km.
    pub fn particle_inertial_km(&self, i: usize, t_s: f64) -> Vec3<f64> {
        self.particle_inertial_state(i, t_s).0
    }

    /// Position of particle `i` in the rotating frame at time `t_s`, km.
    pub fn particle_rotating_km(&self, i: usize, t_s: f64) -> Vec3<f64> {
        let (pos, vel) = self.particle_inertial_state(i, t_s);
        inertial_to_rotating(pos, vel, self.n0_rad_s, t_s).0
    }

    /// Position of moon `k` in the rotating frame at time `t_s`, km.
    pub fn moon_rotating_km(&self, k: usize, t_s: f64) -> Vec3<f64> {
        let m = &self.moons[k];
        let lon = (m.longitude0_rad + (m.n_rad_s - self.n0_rad_s) * t_s)
            .rem_euclid(std::f64::consts::TAU);
        Vec3::new(m.radius_km * lon.cos(), m.radius_km * lon.sin(), 0.0)
    }

    /// Distance from the orbit of moon `k` to the ring core a0, km.
    pub fn moon_gap_km(&self, k: usize) -> f64 {
        (self.moons[k].radius_km - self.a0_km).abs()
    }

    /// The map from the rotating frame to the view plane at time `t_s`.
    ///
    /// For `Inertial` the plane turns by `+n0 t` (anticlockwise), which undoes
    /// the turn of the rotating frame (the two frames agree at t = 0). The angle
    /// is the same f64 product `n0 * t` as in `inertial_to_rotating`, with no
    /// reduction by 2 pi, so the same cos and sin come back and the two turns
    /// cancel to rounding (about 1e-16 of the radius, 1e-11 km at the ring).
    ///
    /// For `Moon(k)` the moon goes to the origin, and the plane turns by
    /// `pi/2 - phi`, where `phi` is the moon's longitude in the rotating frame.
    /// So the moon's radius vector points up (Saturn is straight down), and the
    /// prograde direction `(-sin phi, cos phi)` points to the left (-x).
    ///
    /// # Panics
    /// If `k` is not a moon index.
    pub fn view_transform(&self, frame: ViewFrame, t_s: f64) -> ViewTransform {
        match frame {
            ViewFrame::Inertial => ViewTransform::new([0.0, 0.0], self.n0_rad_s * t_s),
            ViewFrame::Rotating => ViewTransform::IDENTITY,
            ViewFrame::Moon(k) => {
                let p = self.moon_rotating_km(k, t_s);
                let phi = p.y.atan2(p.x);
                ViewTransform::new([p.x, p.y], std::f64::consts::FRAC_PI_2 - phi)
            }
        }
    }

    /// Fills `out` with the instances to draw: Saturn, the moons, then the
    /// particles; `FLOATS_PER_INSTANCE` floats each (x, y, size in pixels, colour id).
    ///
    /// The positions are in km, in the view plane of `view`, **relative to the
    /// eye** (the camera centre). The transform and the subtraction are done
    /// here in f64 and only then rounded to f32. So the f32 error is at most 6e-8
    /// times the distance to the eye. On the screen that is at most about 1e-4
    /// pixel at any zoom, because the distance in pixels is limited by the size
    /// of the window.
    pub fn write_instances(
        &self,
        t_s: f64,
        view: &ViewTransform,
        eye_km: [f64; 2],
        px_per_km: f64,
        out: &mut Vec<f32>,
    ) {
        out.clear();
        out.reserve((1 + self.moons.len() + self.elements.len()) * FLOATS_PER_INSTANCE);
        let mut push = |p: Vec3<f64>, size_px: f32, colour: f32| {
            let v = view.apply([p.x, p.y]);
            out.extend_from_slice(&[
                (v[0] - eye_km[0]) as f32,
                (v[1] - eye_km[1]) as f32,
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
        let r = ring_from_toml_str(RING_TOML).unwrap();
        assert_eq!(r.semi_major_axis_km, 140222.43);
        assert_eq!(r.eccentricity, 0.00235);
        assert_eq!(r.dust_strand_width_min_km, 10.0);
    }

    #[test]
    fn bad_ring_params_are_an_error() {
        assert!(ring_from_toml_str("[core]\nsemi_major_axis_km = 1.0").is_err());
    }

    /// The elements must give the same state at t = 0 as the shared initial
    /// conditions of the CPU runs (same RNG counters).
    #[test]
    fn elements_agree_with_initial_particle_at_t0() {
        let saturn = saturn_from_toml_str(SATURN_TOML).unwrap();
        let field = SaturnField::<f64>::from_params(&saturn);
        let ring = ring_from_toml_str(RING_TOML).unwrap();
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
    fn float_bytes_are_little_endian() {
        let mut out = vec![9u8];
        floats_to_le_bytes(&[1.0, -2.5], &mut out);
        assert_eq!(out, [0, 0, 0x80, 0x3f, 0, 0, 0x20, 0xc0]);
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

    /// Longitude of moon `k` in the rotating frame, rad.
    fn moon_longitude(s: &Scene, k: usize, t_s: f64) -> f64 {
        let p = s.moon_rotating_km(k, t_s);
        p.y.atan2(p.x)
    }

    /// In the rotating view a moon inside a0 moves ahead (anticlockwise) and a moon
    /// outside a0 falls behind (clockwise), like the particles. The schematic
    /// radii are the radii at the epoch: Prometheus 139,160 km (a0 - 1,062 km),
    /// Pandora 142,228 km (a0 + 2,006 km). Kepler to first order gives
    /// n - n0 = -1.5 n0 (r - a0) / a0: +6.6 deg/day and -12.5 deg/day. The
    /// second-order term changes this by 1.25 |r - a0| / a0 (0.9 % for
    /// Prometheus, 1.8 % for Pandora) and J2 by about 0.3 %, so the tolerance is 3 %.
    #[test]
    fn prometheus_moves_ahead_and_pandora_falls_behind_in_the_rotating_view() {
        let s = scene(Preset::Core, 1);
        assert_eq!(s.moons()[0].name, "prometheus");
        assert_eq!(s.moons()[1].name, "pandora");
        let day = 86_400.0;
        for (k, sign) in [(0, 1.0), (1, -1.0)] {
            let r = s.moons()[k].radius_km;
            let drift = moon_longitude(&s, k, day) - moon_longitude(&s, k, 0.0);
            let kepler = -1.5 * s.n0_rad_s() * (r - s.a0_km()) / s.a0_km() * day;
            assert_eq!(drift.signum(), sign, "moon {k}: drift {drift} rad/day");
            assert!(
                ((drift - kepler) / kepler).abs() < 0.03,
                "moon {k}: drift {drift}, Kepler {kepler} rad/day"
            );
        }
    }

    /// The inertial view turns the rotating-frame positions back by n0 t, so it
    /// must give the inertial positions of the epicycles. The two turns use the
    /// same f64 cos and sin, so the error is rounding only: about 1e-16 of
    /// 1.4e5 km per operation, 1e-10 km at most. The tolerance is 1e-8 km. The
    /// last time is about 100 years (n0 t = 3.8e5 rad).
    #[test]
    fn inertial_view_gives_the_inertial_positions() {
        let s = scene(Preset::Core, 50);
        for t in [0.0, 1.0e5, 3.0e7, 3.2e9] {
            let view = s.view_transform(ViewFrame::Inertial, t);
            for i in 0..s.len() {
                let p = s.particle_rotating_km(i, t);
                let v = view.apply([p.x, p.y]);
                let q = s.particle_inertial_km(i, t);
                let err = (v[0] - q.x).hypot(v[1] - q.y);
                assert!(err < 1e-8, "t = {t} s, particle {i}: error {err} km");
            }
        }
    }

    /// In the inertial view each moon goes anticlockwise at its own mean motion:
    /// in one hour Prometheus turns by n t = 0.43 rad. The tolerance 1e-9 rad is
    /// far above rounding (about 1e-15 rad).
    #[test]
    fn moons_orbit_anticlockwise_at_their_own_mean_motion_in_the_inertial_view() {
        use std::f64::consts::{PI, TAU};
        let s = scene(Preset::Core, 1);
        let (t0, dt) = (5.0e6, 3600.0);
        for k in 0..s.moons().len() {
            let lon = |t: f64| {
                let m = s.moon_rotating_km(k, t);
                let v = s.view_transform(ViewFrame::Inertial, t).apply([m.x, m.y]);
                v[1].atan2(v[0])
            };
            let turned = (lon(t0 + dt) - lon(t0) + PI).rem_euclid(TAU) - PI;
            let expected = s.moons()[k].n_rad_s * dt;
            assert!(expected > 0.0);
            assert!(
                (turned - expected).abs() < 1e-9,
                "moon {k}: turned {turned} rad, expected {expected} rad"
            );
        }
    }

    /// In a moon view the moon is at the origin and Saturn is straight down, at
    /// the moon's orbit radius, at every time. Tolerances: the inputs are about
    /// 1.4e5 km, so f64 rounding is about 3e-11 km; 1e-6 km leaves a wide margin.
    #[test]
    fn moon_view_puts_the_moon_at_the_centre_and_saturn_below() {
        let s = scene(Preset::Core, 1);
        for k in 0..s.moons().len() {
            let r = s.moons()[k].radius_km;
            for t in [0.0, 1.0e5, 3.0e6] {
                let view = s.view_transform(ViewFrame::Moon(k), t);
                let m = s.moon_rotating_km(k, t);
                let mv = view.apply([m.x, m.y]);
                assert!(mv[0].abs() < 1e-6 && mv[1].abs() < 1e-6, "moon {k}: {mv:?}");
                let sat = view.apply([0.0, 0.0]);
                assert!(sat[0].abs() < 1e-6, "moon {k}: Saturn at {sat:?}");
                assert!((sat[1] + r).abs() < 1e-6, "moon {k}: Saturn at {sat:?}");
            }
        }
    }

    /// In a moon view a point 1 mrad ahead of the moon on its orbit is to the
    /// left of the moon: x = -r sin(1e-3), y = -r (1 - cos(1e-3)).
    #[test]
    fn moon_view_shows_orbital_motion_to_the_left() {
        let s = scene(Preset::Core, 1);
        let t = 2.0e6;
        for k in 0..s.moons().len() {
            let r = s.moons()[k].radius_km;
            let lon = moon_longitude(&s, k, t) + 1e-3;
            let ahead = [r * lon.cos(), r * lon.sin()];
            let v = s.view_transform(ViewFrame::Moon(k), t).apply(ahead);
            assert!((v[0] + r * 1e-3_f64.sin()).abs() < 1e-6, "moon {k}: {v:?}");
            assert!(
                (v[1] + r * (1.0 - 1e-3_f64.cos())).abs() < 1e-6,
                "moon {k}: {v:?}"
            );
        }
    }

    #[test]
    fn moon_gap_is_the_distance_to_the_core() {
        let s = scene(Preset::Core, 1);
        for k in 0..s.moons().len() {
            assert_eq!(s.moon_gap_km(k), (s.moons()[k].radius_km - s.a0_km()).abs());
        }
        // Prometheus is inside the core, Pandora outside; both within 3,000 km.
        assert!(s.moons()[0].radius_km < s.a0_km() && s.moons()[1].radius_km > s.a0_km());
        assert!(s.moon_gap_km(0) < 3000.0 && s.moon_gap_km(1) < 3000.0);
    }

    #[test]
    fn instance_layout() {
        let s = scene(Preset::Core, 10);
        let mut out = Vec::new();
        s.write_instances(0.0, &ViewTransform::IDENTITY, [0.0, 0.0], 1e-3, &mut out);
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
    /// error is at most 2000 * 6e-8 = 1.2e-4 px. The tolerance is 1e-3 px. The
    /// same holds in a moon view, because the turn is done in f64 too.
    #[test]
    fn camera_relative_f32_error_is_below_a_pixel_fraction() {
        let s = scene(Preset::Core, 2000);
        let t = 3.0e6;
        let height_px = 1000.0;
        let mut out = Vec::new();
        for frame in [ViewFrame::Inertial, ViewFrame::Rotating, ViewFrame::Moon(0)] {
            let view = s.view_transform(frame, t);
            // Exact (f64) view-plane positions, to compare with the f32 output.
            let exact: Vec<[f64; 2]> = (0..s.len())
                .map(|i| {
                    let p = s.particle_rotating_km(i, t);
                    view.apply([p.x, p.y])
                })
                .collect();
            let eye = exact[7];
            for half_height_km in [1.8e5, 1.0e4, 300.0, 5.0] {
                let px_per_km = 0.5 * height_px / half_height_km;
                s.write_instances(t, &view, eye, px_per_km, &mut out);
                let mut checked = 0;
                for (i, p) in exact.iter().enumerate() {
                    let (rx, ry) = (p[0] - eye[0], p[1] - eye[1]);
                    if rx.abs() * px_per_km > 2000.0 || ry.abs() * px_per_km > 2000.0 {
                        continue;
                    }
                    let base = (3 + i) * FLOATS_PER_INSTANCE;
                    let err_px =
                        ((out[base] as f64 - rx).hypot(out[base + 1] as f64 - ry)) * px_per_km;
                    assert!(
                        err_px < 1e-3,
                        "{frame:?}, half height {half_height_km} km: error {err_px} px"
                    );
                    checked += 1;
                }
                assert!(
                    checked >= 1,
                    "{frame:?}: no particle in the window at {half_height_km} km"
                );
            }
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
