//! DET-SINCOS guard: the bits of every function that takes the sine and
//! cosine of one angle must not depend on the build profile.
//!
//! In release, LLVM can fuse `sin` and `cos` of one argument into macOS
//! `___sincos_stret`, which differs from separate `sin` and `cos` by up to
//! 2 ulp (GOLDEN-REL). `crate::real::sin_cos` stops that. The hash below was
//! recorded in a debug build, where nothing is fused. CI runs this test in
//! release too (`cargo test -p fring-model --release --lib`).

use super::arc::ArcGeometry;
use super::frames::{
    LocalState, inertial_to_rotating, local_to_rotating, rotating_to_inertial, rotating_to_local,
};
use super::gravity::SaturnField;
use super::initial::{EpicycleElements, epicycle_state, initial_particle};
use super::moon_radius::{MoonsAtKick, particle_level};
use crate::config::{Distribution, MoonletVelocity};
use crate::params::saturn_from_toml_str;
use crate::rng::box_muller;
use crate::{Real, Vec3};

/// FNV-1a over the bits of the values.
struct Hash(u64);

impl Hash {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn f64(&mut self, x: f64) {
        self.bytes(&x.to_bits().to_le_bytes());
    }
    fn real<T: Real>(&mut self, x: T) {
        self.f64(x.to_f64());
    }
    fn vec<T: Real>(&mut self, v: Vec3<T>) {
        self.real(v.x);
        self.real(v.y);
        self.real(v.z);
    }
}

fn field() -> SaturnField<f64> {
    let s = saturn_from_toml_str(include_str!("../../../../params/saturn.toml")).unwrap();
    SaturnField::from_params(&s)
}

/// Frames, generic over f64 and f32.
fn frames_into<T: Real>(h: &mut Hash) {
    let a0 = T::from_f64(140_222.43);
    let n0 = T::from_f64(5.8e-4);
    for i in 0..300_u32 {
        let u = f64::from(i);
        let phi = u * 0.731;
        let pos = Vec3::new(
            T::from_f64(140_000.0 * phi.cos()),
            T::from_f64(140_000.0 * phi.sin()),
            T::from_f64(3.0 * (u * 0.3).sin()),
        );
        let vel = Vec3::new(
            T::from_f64(-0.5 * phi.sin()),
            T::from_f64(0.5 * phi.cos()),
            T::from_f64(1e-3 * u),
        );
        let t = T::from_f64(1.0e5 * u);
        let (p, v) = inertial_to_rotating(pos, vel, n0, t);
        h.vec(p);
        h.vec(v);
        let (p2, v2) = rotating_to_inertial(p, v, n0, t);
        h.vec(p2);
        h.vec(v2);
        let l = rotating_to_local(p, v, a0);
        for x in [l.x_km, l.y_km, l.z_km, l.vx_km_s, l.vy_km_s, l.vz_km_s] {
            h.real(x);
        }
        let (p3, v3) = local_to_rotating(&l, a0);
        h.vec(p3);
        h.vec(v3);
        let l2 = LocalState {
            y_km: T::from_f64(u * 977.0),
            ..l
        };
        let (p4, v4) = local_to_rotating(&l2, a0);
        h.vec(p4);
        h.vec(v4);
    }
}

fn distributions() -> Vec<Distribution> {
    let mut d = vec![
        Distribution::Core {
            semi_major_axis_km: 140_220.0,
            sigma_semi_major_axis_km: 5.0,
            eccentricity: 0.0024,
            apse_longitude_deg: 40.0,
            inclination_deg: 0.0,
        },
        Distribution::Band {
            min_semi_major_axis_km: 139_000.0,
            max_semi_major_axis_km: 141_500.0,
        },
        Distribution::EccentricBand {
            min_semi_major_axis_km: 139_000.0,
            max_semi_major_axis_km: 141_500.0,
            eccentricity: 0.01,
            apse_longitude_deg: 70.0,
        },
        Distribution::Clump {
            semi_major_axis_km: 140_222.43,
            longitude_deg: 200.0,
            sigma_km: 20.0,
        },
    ];
    for velocity in [
        MoonletVelocity::Synchronous,
        MoonletVelocity::Centre,
        MoonletVelocity::Circular,
    ] {
        d.push(Distribution::Moonlet {
            semi_major_axis_km: 140_222.43,
            longitude_deg: 200.0,
            packing_radius_km: 50.0,
            velocity,
        });
    }
    d
}

#[test]
fn sin_cos_users_bits_do_not_depend_on_the_profile() {
    let mut h = Hash::new();
    let f = field();

    // box_muller, f64 and f32.
    for i in 0..2000_u32 {
        let u1 = (f64::from(i) + 0.5) / 2000.0;
        let u2 = (f64::from(i) * 0.618_033_988_7).fract().max(1e-9);
        let (a, b) = box_muller(u1, u2);
        h.f64(a);
        h.f64(b);
        let (a, b) = box_muller(u1 as f32, u2 as f32);
        h.real(a);
        h.real(b);
    }

    // Epicycle states with e > 0, i > 0 and a vertical phase.
    for i in 0..500_u32 {
        let u = f64::from(i);
        let (pos, vel) = epicycle_state(
            &f,
            &EpicycleElements {
                a_km: 139_000.0 + 2.0 * u,
                e: 0.01 + 1e-5 * u,
                i_rad: 1e-4 * (1.0 + u * 0.01),
                mean_anomaly_rad: u * 0.9,
                z_phase_rad: u * 1.7,
                mean_longitude_rad: u * 2.3,
            },
        );
        h.vec(pos);
        h.vec(vel);
    }

    // Every initial distribution (core, band, eccentric band, clump, moonlet).
    for dist in distributions() {
        for id in 0..400 {
            let (p, v) = initial_particle(&f, &dist, 20_261_101, id);
            h.vec(p);
            h.vec(v);
        }
    }

    // Frame conversions.
    frames_into::<f64>(&mut h);
    frames_into::<f32>(&mut h);

    // Arc fold and wrap (also the centre, width and half-width angles).
    let n0 = 5.8e-4;
    for (width, centre) in [
        (0.3, 1.1),
        (1.0, -2.0),
        (2.5, 0.2),
        (std::f64::consts::TAU, 0.0),
    ] {
        let g = ArcGeometry::new(width, centre, n0).unwrap();
        for i in 0..300_u32 {
            let u = f64::from(i);
            let lon = u * 0.713;
            let pos = Vec3::new(140_000.0 * lon.cos(), 140_000.0 * lon.sin(), 2.0);
            let vel = Vec3::new(-0.5 * lon.sin(), 0.5 * lon.cos(), 1e-3);
            let t = 3.0e5 * u;
            if let Some(w) = g.fold(pos, vel, t) {
                h.vec(w.pos_km);
                h.vec(w.vel_km_s);
                h.bytes(&w.k.to_le_bytes());
            }
            if let Some(w) = g.wrap(pos, vel, t) {
                h.vec(w.pos_km);
                h.vec(w.vel_km_s);
                h.bytes(&w.k.to_le_bytes());
            }
            if let Some(w) = g.fold(
                Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
                Vec3::new(vel.x as f32, vel.y as f32, vel.z as f32),
                t,
            ) {
                h.vec(w.pos_km);
                h.vec(w.vel_km_s);
                h.bytes(&w.k.to_le_bytes());
            }
            let (c, s) = g.centre_cos_sin(t);
            h.f64(c);
            h.f64(s);
        }
        for (c, s) in [g.half_width_cos_sin(), g.width_cos_sin()] {
            h.f64(c);
            h.f64(s);
        }
    }

    // Sub-step level with the arc images (moon_radius).
    let g = ArcGeometry::new(0.4, 0.5, n0).unwrap();
    let moon = [Vec3::new(139_380.0, 0.0, 0.0)];
    let mv = [Vec3::new(0.0, 16.5, 0.0)];
    let gm = [0.01071];
    let rad = [43.0];
    let m = MoonsAtKick {
        pos_km: &moon,
        vel_km_s: &mv,
        gm_km3_s2: &gm,
        radius_km: &rad,
    };
    for i in 0..400_u32 {
        let u = f64::from(i);
        let q = Vec3::new(139_380.0 + 3.0 * u, -60.0 + 0.3 * u, 0.5);
        let w = Vec3::new(0.0, 16.4, 0.0);
        h.bytes(&[particle_level(356.25, 534.4, q, w, &m, 0.1, 4, Some(&g))]);
    }

    assert_eq!(h.0, EXPECTED, "hash {:#018x}", h.0);
}

const EXPECTED: u64 = 0x06c3_bdb7_b1b1_d749;
