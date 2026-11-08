//! The moon-radius rule and the sub-step level rule (ICR-004, E1).
//!
//! Generic over [`Real`], no `std`-only items, so it builds for wasm32. These
//! functions are the reference of the f32 twins in `fring-gpu` (ICR-004
//! section 4). Distances are in km, speeds in km s⁻¹, times in s, GM in
//! km³ s⁻².
//!
//! * [`segment_min_d2`], [`drift_hits_moon`]: the segment test of ICR-006. A
//!   drift between two kicks is a straight line in the frame of the moon;
//!   the particle is lost if the line passes inside the radius.
//! * [`hermite_moon`]: the moon position and velocity at a sub-kick time
//!   (cubic Hermite between the two nearest table rows).
//! * [`substep_level`], [`particle_level`]: the level `L` of the next step
//!   (`2^L` Wisdom-Holman sub-steps).

use super::arc::ArcGeometry;
use super::hermite::hermite3;
use crate::{Real, Vec3};

/// Smallest squared distance (km²) from the origin to the segment from `a` to
/// `b` (positions relative to the moon, km). The order of operations is that
/// of the f32 twin `fring_gpu::wh::segment_min_d2`.
pub fn segment_min_d2<T: Real>(a: [T; 3], b: [T; 3]) -> T {
    let dot = |u: [T; 3], v: [T; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let dd = dot(d, d);
    let mut s = T::ZERO;
    if dd > T::ZERO {
        s = ((-dot(a, d)) / dd).max(T::ZERO).min(T::ONE);
    }
    let c = [a[0] + d[0] * s, a[1] + d[1] * s, a[2] + d[2] * s];
    dot(c, c)
}

/// True if a drift from `rel_a` to `rel_b` (positions relative to a moon, km)
/// comes closer to the moon centre than `radius_km`.
pub fn drift_hits_moon<T: Real>(rel_a: [T; 3], rel_b: [T; 3], radius_km: T) -> bool {
    segment_min_d2(rel_a, rel_b) < radius_km * radius_km
}

/// Moon position (km) and velocity (km s⁻¹) at time `t_s`, by cubic Hermite
/// interpolation between the table rows at `t0_s` and `t1_s`. At `t0_s` and
/// `t1_s` the rows are returned exactly.
pub fn hermite_moon<T: Real>(
    t_s: T,
    t0_s: T,
    t1_s: T,
    x0: Vec3<T>,
    v0: Vec3<T>,
    x1: Vec3<T>,
    v1: Vec3<T>,
) -> (Vec3<T>, Vec3<T>) {
    if t_s == t0_s {
        return (x0, v0);
    }
    if t_s == t1_s {
        return (x1, v1);
    }
    let h = t1_s - t0_s;
    hermite3(x0, v0, x1, v1, h, (t_s - t0_s) / h)
}

/// The level `L` (0 to `max_level`) for one particle and one moon.
///
/// `d_km` is the distance and `u_km_s` the relative speed at the last kick,
/// `dt_s` the time to the end of the next step, `radius_km` the moon radius
/// (the floor of the effective distance), `gm_km3_s2` the moon GM. With
///
/// ```text
/// d_eff  = max(d - u dt, radius)
/// v_eff² = u² + 2 GM / d_eff
/// ```
///
/// the result is the smallest `L` with `h v_eff / (2^L d_eff) <= accuracy`,
/// or `max_level` if there is none. The test is `h² v_eff² <= acc² 4^L d_eff²`
/// (no logarithm, no square root).
#[allow(clippy::too_many_arguments)]
pub fn substep_level<T: Real>(
    h_s: T,
    d_km: T,
    u_km_s: T,
    dt_s: T,
    radius_km: T,
    gm_km3_s2: T,
    accuracy: T,
    max_level: u8,
) -> u8 {
    let d_eff = (d_km - u_km_s * dt_s).max(radius_km);
    let v2 = u_km_s * u_km_s + T::from_f64(2.0) * gm_km3_s2 / d_eff;
    let lhs = h_s * h_s * v2;
    let mut rhs = accuracy * accuracy * d_eff * d_eff;
    let four = T::from_f64(4.0);
    for level in 0..max_level {
        if lhs <= rhs {
            return level;
        }
        rhs *= four;
    }
    max_level
}

/// What [`particle_level`] needs about the moons at the last kick. All slices
/// have one entry per moon.
#[derive(Clone, Copy, Debug)]
pub struct MoonsAtKick<'a, T: Real> {
    /// Moon positions, km.
    pub pos_km: &'a [Vec3<T>],
    /// Moon velocities, km s⁻¹.
    pub vel_km_s: &'a [Vec3<T>],
    /// Moon GM, km³ s⁻².
    pub gm_km3_s2: &'a [T],
    /// Moon radii, km.
    pub radius_km: &'a [T],
}

/// `v` turned about z by the angle with cosine `c` and sine `s`.
fn turn<T: Real>(v: Vec3<T>, c: T, s: T) -> Vec3<T> {
    Vec3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

/// The level of the next step of a particle: the largest [`substep_level`]
/// over the moons. With `arc = Some(..)` (ICR-004 section 3, D1) the state is
/// also turned about z by `-Δθ` and `+Δθ` (the periodic images, as
/// `ArcGeometry::wrap` does) and the largest level of the three is used,
/// because the particle can wrap before the step that uses the level. With
/// `arc = None` only the particle itself is used.
#[allow(clippy::too_many_arguments)]
pub fn particle_level<T: Real>(
    h_s: T,
    dt_s: T,
    pos_km: Vec3<T>,
    vel_km_s: Vec3<T>,
    moons: &MoonsAtKick<'_, T>,
    accuracy: T,
    max_level: u8,
    arc: Option<&ArcGeometry>,
) -> u8 {
    let one = |p: Vec3<T>, w: Vec3<T>| -> u8 {
        let mut level = 0u8;
        for j in 0..moons.pos_km.len() {
            let d = (p - moons.pos_km[j]).norm();
            let u = (w - moons.vel_km_s[j]).norm();
            let l = substep_level(
                h_s,
                d,
                u,
                dt_s,
                moons.radius_km[j],
                moons.gm_km3_s2[j],
                accuracy,
                max_level,
            );
            level = level.max(l);
        }
        level
    };
    let mut level = one(pos_km, vel_km_s);
    if let Some(g) = arc {
        let w = g.width_rad();
        let (c, s) = (T::from_f64(w.cos()), T::from_f64(w.sin()));
        for sign in [-T::ONE, T::ONE] {
            let s = s * sign;
            level = level.max(one(turn(pos_km, c, s), turn(vel_km_s, c, s)));
        }
    }
    level
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{build_ephemeris, interpolate_moons};

    const H: f64 = 356.25;
    const GM_PROM: f64 = 0.01071;

    #[test]
    fn t1_segment_cases_f64_and_f32() {
        let d = segment_min_d2([-70.0, 20.0, 0.0], [70.0, 20.0, 0.0]);
        assert!((d - 400.0).abs() < 1e-9, "{d}");
        let d = segment_min_d2([-70.0, 20.0, 0.0], [-30.0, 20.0, 0.0]);
        assert!((d - 1300.0).abs() < 1e-9, "{d}");
        assert_eq!(segment_min_d2([3.0, 4.0, 0.0], [3.0, 4.0, 0.0]), 25.0);
        // f32 agrees with f64 to 1e-6 relative.
        let cases = [
            ([-70.0, 20.0, 3.0], [70.0, 25.0, -2.0]),
            ([-700.0, 200.0, 3.0], [300.0, 250.0, -2.0]),
            ([5.0, 5.0, 5.0], [-5.0, 7.0, 1.0]),
        ];
        for (a, b) in cases {
            let d64 = segment_min_d2(a, b);
            let d32 = segment_min_d2(a.map(|x| x as f32), b.map(|x| x as f32)) as f64;
            assert!((d32 - d64).abs() / d64 < 1e-6, "{d32} {d64}");
        }
    }

    #[test]
    fn t2_drift_hits_moon_at_radius_plus_and_minus_one_km() {
        for r in [43.09, 40.63] {
            let miss = drift_hits_moon([-60.0, r + 1.0, 0.0], [60.0, r + 1.0, 0.0], r);
            let hit = drift_hits_moon([-60.0, r - 1.0, 0.0], [60.0, r - 1.0, 0.0], r);
            assert!(!miss && hit, "radius {r}");
            let miss = drift_hits_moon(
                [-60f32, r as f32 + 1.0, 0.0],
                [60.0, r as f32 + 1.0, 0.0],
                r as f32,
            );
            let hit = drift_hits_moon(
                [-60f32, r as f32 - 1.0, 0.0],
                [60.0, r as f32 - 1.0, 0.0],
                r as f32,
            );
            assert!(!miss && hit, "f32 radius {r}");
        }
    }

    fn hermite_errors() -> (f64, f64) {
        let saturn =
            crate::params::saturn_from_toml_str(include_str!("../../../../params/saturn.toml"))
                .unwrap();
        let moons =
            crate::params::moons_from_toml_str(include_str!("../../../../params/moons.toml"))
                .unwrap();
        let eph = build_ephemeris(&saturn, &moons, 1000.0 * H + 7200.0, 1800.0).unwrap();
        let mut worst_p = 0.0f64;
        let mut worst_v = 0.0f64;
        let t_start = 1800.0;
        for i in 0..16 {
            // Node pair 62 h apart in 1000 steps; sub-times inside the gap.
            let k = 60 * i + 5;
            let t0 = t_start + k as f64 * H;
            let t1 = t0 + H;
            let (a, b) = (
                interpolate_moons(&eph, t0).unwrap(),
                interpolate_moons(&eph, t1).unwrap(),
            );
            let t = t0 + H * (0.13 + 0.05 * i as f64);
            let ex = interpolate_moons(&eph, t).unwrap();
            for m in 0..2 {
                let (p, v) = hermite_moon(
                    t,
                    t0,
                    t1,
                    a.moons[m].position_km,
                    a.moons[m].velocity_km_s,
                    b.moons[m].position_km,
                    b.moons[m].velocity_km_s,
                );
                worst_p = worst_p.max((p - ex.moons[m].position_km).norm());
                worst_v = worst_v.max((v - ex.moons[m].velocity_km_s).norm());
                let (p1, v1) = hermite_moon(
                    t1,
                    t0,
                    t1,
                    a.moons[m].position_km,
                    a.moons[m].velocity_km_s,
                    b.moons[m].position_km,
                    b.moons[m].velocity_km_s,
                );
                assert_eq!((p1, v1), (b.moons[m].position_km, b.moons[m].velocity_km_s));
            }
        }
        (worst_p, worst_v)
    }

    #[test]
    fn t3_hermite_moon_against_the_ephemeris() {
        let (worst_p, worst_v) = hermite_errors();
        eprintln!("hermite_moon worst position {worst_p:e} km, velocity {worst_v:e} km/s");
        assert!(worst_p < 5e-3, "position error {worst_p} km");
        // ICR-004 amendment 2: a cubic Hermite with nodes h apart gives
        // h^3 x'''' / 125 = 9.6e-6 km/s (measured 1.0e-5), so the limit is
        // 3e-5 km/s. The velocity enters the level rule only through
        // u = |w - mdot| (0.03 km/s and more), a relative error of 3e-4.
        assert!(worst_v < 3e-5, "velocity error {worst_v} km/s");
    }

    /// Independent form with a logarithm; used away from the level edges.
    fn level_by_log(d: f64, u: f64, r: f64, gm: f64, acc: f64, max: u8, dt: f64) -> (u8, f64) {
        let de = (d - u * dt).max(r);
        let v = (u * u + 2.0 * gm / de).sqrt();
        let x = H * v / (acc * de); // L = ceil(log2 x)
        let l = if x <= 1.0 { 0.0 } else { x.log2().ceil() };
        let frac = (x.log2() - x.log2().round()).abs();
        ((l as u8).min(max), frac)
    }

    #[test]
    fn t4_level_table() {
        // Hand cases: (d, u, radius, accuracy, max, expected).
        let hand: [(f64, f64, f64, f64, u8, u8); 10] = [
            (2000.0, 0.001, 43.09, 0.1, 4, 0),
            (2000.0, 0.2, 43.09, 0.1, 4, 0),
            (500.0, 0.05, 43.09, 0.1, 4, 0),
            // The floor: d - u dt < radius, so d_eff = radius.
            (43.09, 0.15, 43.09, 0.1, 4, 4),
            (43.09, 0.15, 43.09, 0.1, 2, 2),
            (43.09, 0.15, 43.09, 0.1, 0, 0),
            // accuracy 1.0 (loosest) and 0.07
            (50.0, 0.03, 43.09, 1.0, 4, 0),
            (50.0, 0.03, 43.09, 0.07, 6, 3),
            (50.0, 0.03, 43.09, 0.1, 4, 2),
            (60.0, 0.0, 43.09, 0.1, 4, 1),
        ];
        for (d, u, r, acc, max, want) in hand {
            let got = substep_level(H, d, u, 1.5 * H, r, GM_PROM, acc, max);
            assert_eq!(got, want, "d {d} u {u} r {r} acc {acc} max {max}");
        }
        // The case (60, 0): focusing term alone, h v_eff / d = 0.112 > 0.1.
        // Grid against the log form (40 cases), away from the edges.
        let mut n = 0;
        for d in [20.0, 45.0, 63.0, 90.0, 150.0, 400.0, 1000.0, 2000.0] {
            for u in [0.001, 0.01, 0.03, 0.07, 0.15, 0.2] {
                for acc in [0.07, 0.1, 1.0] {
                    let (want, frac) = level_by_log(d, u, 43.09, GM_PROM, acc, 4, 1.5 * H);
                    if frac < 0.02 {
                        continue; // on the edge of a level
                    }
                    n += 1;
                    let got = substep_level(H, d, u, 1.5 * H, 43.09, GM_PROM, acc, 4);
                    assert_eq!(got, want, "d {d} u {u} acc {acc}");
                    let g32 = substep_level(
                        H as f32,
                        d as f32,
                        u as f32,
                        1.5 * H as f32,
                        43.09f32,
                        GM_PROM as f32,
                        acc as f32,
                        4,
                    );
                    assert_eq!(g32, want, "f32 d {d} u {u} acc {acc}");
                }
            }
        }
        assert!(n >= 40, "only {n} grid cases");
    }

    #[test]
    fn arc_image_nearer_the_moon_sets_the_level() {
        // Moon on the +x axis at 140,000 km; particle 5,000 km away along the
        // ring (far), but its image turned by -Δθ lands 60 km from the moon.
        let r = 140_000.0f64;
        let dth = 5000.0 / r;
        let g = ArcGeometry::new(dth, 0.0, 1.1756e-4).unwrap();
        let moon = [Vec3::new(r, 0.0, 0.0)];
        let mv = [Vec3::new(0.0, 0.0, 0.0)];
        let (gm, rad) = ([GM_PROM], [43.09]);
        let m = MoonsAtKick {
            pos_km: &moon,
            vel_km_s: &mv,
            gm_km3_s2: &gm,
            radius_km: &rad,
        };
        // Image: p turned by -dth must be at (r, -60) -> p = r*(cos dth, sin dth) + offset.
        let (c, s) = (dth.cos(), dth.sin());
        let target = Vec3::new(r, -60.0, 0.0);
        // p = target turned by +dth (so that turning p by -dth gives target).
        let p = Vec3::new(
            c * target.x - s * target.y,
            s * target.x + c * target.y,
            0.0,
        );
        let w = Vec3::new(-s * 10.0, c * 10.0, 0.0) * 0.0; // at rest in inertial frame
        let far = particle_level(H, 1.5 * H, p, w, &m, 0.1, 4, None);
        let with_arc = particle_level(H, 1.5 * H, p, w, &m, 0.1, 4, Some(&g));
        assert_eq!(far, 0);
        let image = particle_level(H, 1.5 * H, target, w, &m, 0.1, 4, None);
        assert!(image > 0);
        assert_eq!(with_arc, image);
        // A particle whose images are all far: same level with and without.
        let q = Vec3::new(r, 2000.0, 0.0);
        let a = particle_level(H, 1.5 * H, q, w, &m, 0.1, 4, None);
        let b = particle_level(H, 1.5 * H, q, w, &m, 0.1, 4, Some(&g));
        assert_eq!(a, b);
    }
}
