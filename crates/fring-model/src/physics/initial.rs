//! Initial conditions from the config distributions (`docs/PLAN.md` section
//! 7.4), made with the Philox RNG.
//!
//! Particle `id` uses the counter `{ index: id, step: 0, stream:
//! INITIAL_CONDITIONS, draw }`. So a particle does not depend on the other
//! particles, on `count` or on the order of generation.
//!
//! The states are first-order epicycles about circular equatorial orbits
//! with the exact frequencies n, kappa, nu of Saturn's field (section 7.2),
//! as in the V3 reference. They are not osculating Keplerian elements:
//! in this field the osculating e of a circular orbit is about 4.5e-3 at the
//! F ring. The config `eccentricity` is the epicyclic amplitude (radial
//! excursion `a e`). Time is 0 and the frame is inertial; the rotating frame
//! coincides with it at time 0.
//!
//! All particles have `z = 0` and `vz = 0`, except the clump and the
//! moonlet, which have a z offset (and `vz = 0`): the distributions have no
//! inclination field yet.
//!
//! | Distribution | a | e, apse | mean longitude |
//! |---|---|---|---|
//! | `Core` | Gaussian (mean, sigma) | config e, config apse | uniform |
//! | `Band` | uniform in [min, max] | 0 | uniform |
//! | `Clump` | see below | | |
//! | `Moonlet` | see below | | |
//!
//! `Core`: the epicycle phase is `M = lambda - apse` (the pericentre is where
//! the guiding centre is at the apse longitude).
//!
//! `Clump`: three Gaussian offsets (standard deviation `sigma_km` each, in
//! the inertial x, y, z directions) about the point on the circle of radius
//! `a` at the given inertial longitude. Every particle gets the circular
//! velocity of that centre point (no velocity dispersion).
//!
//! `Moonlet` (ICR-008): uniform in the sphere of radius `packing_radius_km`
//! about the point on the circle of radius `a` at the given inertial
//! longitude (`r = R u^(1/3)`, `cos(theta) = 1 - 2u`, `phi = 2 pi u`). The
//! velocity is one of three ([`MoonletVelocity`]): `synchronous`,
//! `v = n z x r` with the exact n at `a` (at rest in the frame that turns
//! with the centre); `centre`, the circular velocity of the centre point (as
//! `Clump`); `circular`, the circular velocity at the cylindrical radius of
//! the particle.

use super::circular::{circular_speed, frequencies};
use super::gravity::SaturnField;
use crate::config::{Distribution, MoonletVelocity};
use crate::rng::{Counter, Philox, Stream, box_muller};
use crate::{Particles, Real, Result, Vec3};

/// Epicyclic elements of one particle about a circular orbit (angles in radians).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpicycleElements<T: Real> {
    /// Radius of the guiding circle in km.
    pub a_km: T,
    /// Radial epicycle amplitude e (radial excursion `a e`).
    pub e: T,
    /// Vertical epicycle amplitude i in rad (height amplitude `a i`).
    pub i_rad: T,
    /// Phase of the radial epicycle (0 at pericentre).
    pub mean_anomaly_rad: T,
    /// Phase of the vertical oscillation (0 at the equator, going up).
    pub z_phase_rad: T,
    /// Longitude of the guiding centre from the x axis.
    pub mean_longitude_rad: T,
}

/// Position and velocity (inertial, Saturn-centred, km and km/s) of the
/// first-order epicycle `el`, with the exact n, kappa, nu at `el.a_km`:
///
/// `r = a (1 - e cos M)`, `theta = lambda + 2 e (n/kappa) sin M`,
/// `vr = a e kappa sin M`, `vtheta = a n (1 + e cos M)`,
/// `z = a i sin(phi)`, `vz = a i nu cos(phi)`.
pub fn epicycle_state<T: Real>(
    field: &SaturnField<T>,
    el: &EpicycleElements<T>,
) -> (Vec3<T>, Vec3<T>) {
    let f = frequencies(field, el.a_km);
    let (n, kappa, nu) = (f.n_rad_s, f.kappa_rad_s, f.nu_rad_s);
    let a = el.a_km;
    let (sm, cm) = (el.mean_anomaly_rad.sin(), el.mean_anomaly_rad.cos());
    let r = a * (T::ONE - el.e * cm);
    let two = T::from_f64(2.0);
    let theta = el.mean_longitude_rad + two * el.e * (n / kappa) * sm;
    let vr = a * el.e * kappa * sm;
    let vt = a * n * (T::ONE + el.e * cm);
    let (s, c) = (theta.sin(), theta.cos());
    let pos = Vec3::new(r * c, r * s, a * el.i_rad * el.z_phase_rad.sin());
    let vel = Vec3::new(
        vr * c - vt * s,
        vr * s + vt * c,
        a * el.i_rad * nu * el.z_phase_rad.cos(),
    );
    (pos, vel)
}

fn counter(id: u32, draw: u16) -> Counter {
    Counter {
        index: id,
        step: 0,
        stream: Stream::INITIAL_CONDITIONS,
        draw,
    }
}

/// The initial state (position km, velocity km/s) of particle `id` for the
/// distribution `dist`, run seed `seed` and Saturn's field `field`.
///
/// `dist` must be valid (`RunConfig::validate` checks it).
pub fn initial_particle(
    field: &SaturnField<f64>,
    dist: &Distribution,
    seed: u64,
    id: u32,
) -> (Vec3<f64>, Vec3<f64>) {
    let rng = Philox::from_seed(seed);
    let two_pi = 2.0 * std::f64::consts::PI;
    match *dist {
        Distribution::Core {
            semi_major_axis_km,
            sigma_semi_major_axis_km,
            eccentricity,
            apse_longitude_deg,
        } => {
            let [u1, u2] = rng.uniform_f64(counter(id, 0));
            let (g, _) = box_muller(u1, u2);
            let a_km = semi_major_axis_km + sigma_semi_major_axis_km * g;
            let [u_lambda, _] = rng.uniform_f64(counter(id, 1));
            let lambda = two_pi * u_lambda;
            epicycle_state(
                field,
                &EpicycleElements {
                    a_km,
                    e: eccentricity,
                    i_rad: 0.0,
                    mean_anomaly_rad: lambda - apse_longitude_deg.to_radians(),
                    z_phase_rad: 0.0,
                    mean_longitude_rad: lambda,
                },
            )
        }
        Distribution::Band {
            min_semi_major_axis_km,
            max_semi_major_axis_km,
        } => {
            let [u_a, u_lambda] = rng.uniform_f64(counter(id, 0));
            epicycle_state(
                field,
                &EpicycleElements {
                    a_km: min_semi_major_axis_km
                        + (max_semi_major_axis_km - min_semi_major_axis_km) * u_a,
                    e: 0.0,
                    i_rad: 0.0,
                    mean_anomaly_rad: 0.0,
                    z_phase_rad: 0.0,
                    mean_longitude_rad: two_pi * u_lambda,
                },
            )
        }
        Distribution::EccentricBand {
            min_semi_major_axis_km,
            max_semi_major_axis_km,
            eccentricity,
            apse_longitude_deg,
        } => {
            // The same draws as `Band` (ICR-007), so the circular control with
            // the same seed has the same a and mean longitude.
            let [u_a, u_lambda] = rng.uniform_f64(counter(id, 0));
            let lambda = two_pi * u_lambda;
            epicycle_state(
                field,
                &EpicycleElements {
                    a_km: min_semi_major_axis_km
                        + (max_semi_major_axis_km - min_semi_major_axis_km) * u_a,
                    e: eccentricity,
                    i_rad: 0.0,
                    mean_anomaly_rad: lambda - apse_longitude_deg.to_radians(),
                    z_phase_rad: 0.0,
                    mean_longitude_rad: lambda,
                },
            )
        }
        Distribution::Clump {
            semi_major_axis_km,
            longitude_deg,
            sigma_km,
        } => {
            let [u1, u2] = rng.uniform_f64(counter(id, 0));
            let (gx, gy) = box_muller(u1, u2);
            let [u3, u4] = rng.uniform_f64(counter(id, 1));
            let (gz, _) = box_muller(u3, u4);
            let (s, c) = longitude_deg.to_radians().sin_cos();
            let v = circular_speed(field, semi_major_axis_km);
            (
                Vec3::new(
                    semi_major_axis_km * c + sigma_km * gx,
                    semi_major_axis_km * s + sigma_km * gy,
                    sigma_km * gz,
                ),
                Vec3::new(-v * s, v * c, 0.0),
            )
        }
        Distribution::Moonlet {
            semi_major_axis_km,
            longitude_deg,
            packing_radius_km,
            velocity,
        } => {
            let [u_r, u_mu] = rng.uniform_f64(counter(id, 0));
            let [u_phi, _] = rng.uniform_f64(counter(id, 1));
            let r = packing_radius_km * u_r.cbrt();
            let mu = 1.0 - 2.0 * u_mu;
            let rho = (1.0 - mu * mu).max(0.0).sqrt();
            let (sp, cp) = (two_pi * u_phi).sin_cos();
            let (s, c) = longitude_deg.to_radians().sin_cos();
            let pos = Vec3::new(
                semi_major_axis_km * c + r * rho * cp,
                semi_major_axis_km * s + r * rho * sp,
                r * mu,
            );
            let vel = match velocity {
                MoonletVelocity::Synchronous => {
                    let n = frequencies(field, semi_major_axis_km).n_rad_s;
                    Vec3::new(-n * pos.y, n * pos.x, 0.0)
                }
                MoonletVelocity::Centre => {
                    let v = circular_speed(field, semi_major_axis_km);
                    Vec3::new(-v * s, v * c, 0.0)
                }
                MoonletVelocity::Circular => {
                    let r_cyl = pos.x.hypot(pos.y);
                    let v = circular_speed(field, r_cyl) / r_cyl;
                    Vec3::new(-v * pos.y, v * pos.x, 0.0)
                }
            };
            (pos, vel)
        }
    }
}

/// The initial particles with ids `0..count` (see [`initial_particle`]).
pub fn generate_particles(
    field: &SaturnField<f64>,
    dist: &Distribution,
    count: u32,
    seed: u64,
) -> Result<Particles> {
    let mut p = Particles::with_capacity(count as usize);
    for id in 0..count {
        let (pos, vel) = initial_particle(field, dist, seed, id);
        p.push(id, pos, vel);
    }
    p.validate()?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::saturn_from_toml_str;

    const N: u32 = 20_000;
    const SEED: u64 = 20_261_101;

    fn field() -> SaturnField<f64> {
        let s = saturn_from_toml_str(include_str!("../../../../params/saturn.toml")).unwrap();
        SaturnField::from_params(&s)
    }

    fn core() -> Distribution {
        Distribution::Core {
            semi_major_axis_km: 140_220.0,
            sigma_semi_major_axis_km: 5.0,
            eccentricity: 0.0024,
            apse_longitude_deg: 40.0,
        }
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn std(v: &[f64]) -> f64 {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    }

    /// Kolmogorov-Smirnov distance of values in [0, 1] from the uniform law.
    fn ks_uniform(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len() as f64;
        v.iter()
            .enumerate()
            .map(|(i, &x)| {
                ((i as f64 + 1.0) / n - x)
                    .abs()
                    .max((x - i as f64 / n).abs())
            })
            .fold(0.0, f64::max)
    }

    /// 99.9 % critical value of the KS distance, 1.95 / sqrt(n).
    fn ks_limit() -> f64 {
        1.95 / f64::from(N).sqrt()
    }

    fn wrap(x: f64) -> f64 {
        x.rem_euclid(2.0 * std::f64::consts::PI)
    }

    #[test]
    fn circular_epicycle_is_the_circular_orbit() {
        let f = field();
        let el = EpicycleElements {
            a_km: 140_222.43,
            e: 0.0,
            i_rad: 0.0,
            mean_anomaly_rad: 1.0,
            z_phase_rad: 2.0,
            mean_longitude_rad: 0.7,
        };
        let (p, v) = epicycle_state(&f, &el);
        assert!((p.norm() - 140_222.43).abs() < 1e-9);
        assert!((v.norm() - circular_speed(&f, 140_222.43)).abs() < 1e-12);
        assert!(p.dot(v).abs() < 1e-7);
    }

    #[test]
    fn band_is_uniform_and_circular() {
        let f = field();
        let (lo, hi) = (139_000.0, 141_800.0);
        let p = generate_particles(
            &f,
            &Distribution::Band {
                min_semi_major_axis_km: lo,
                max_semi_major_axis_km: hi,
            },
            N,
            SEED,
        )
        .unwrap();
        let (mut ua, mut ul) = (Vec::new(), Vec::new());
        for i in 0..p.len() {
            let (pos, vel) = (p.position_km(i), p.velocity_km_s(i));
            let r = pos.norm();
            assert!(r >= lo && r <= hi);
            assert_eq!(pos.z, 0.0);
            assert!((vel.norm() - circular_speed(&f, r)).abs() < 1e-11);
            assert!(pos.dot(vel).abs() < 1e-6);
            ua.push((r - lo) / (hi - lo));
            ul.push(wrap(pos.y.atan2(pos.x)) / (2.0 * std::f64::consts::PI));
        }
        assert!(ks_uniform(ua) < ks_limit());
        assert!(ks_uniform(ul) < ks_limit());
    }

    #[test]
    fn eccentric_band_shares_the_band_draws_and_has_the_apse() {
        // ICR-007: same a and mean longitude as `band` with the same seed;
        // e = 0 gives the band states; e > 0 gives r = a (1 - e cos M) with
        // M = lambda - apse, and radial speed a e kappa sin M.
        let f = field();
        let (lo, hi, e, apse_deg) = (140_150.0, 140_700.0, 0.0024, 353.78_f64);
        let ecc = |e: f64| Distribution::EccentricBand {
            min_semi_major_axis_km: lo,
            max_semi_major_axis_km: hi,
            eccentricity: e,
            apse_longitude_deg: apse_deg,
        };
        let band = Distribution::Band {
            min_semi_major_axis_km: lo,
            max_semi_major_axis_km: hi,
        };
        let n = 5_000;
        let pb = generate_particles(&f, &band, n, SEED).unwrap();
        let p0 = generate_particles(&f, &ecc(0.0), n, SEED).unwrap();
        let pe = generate_particles(&f, &ecc(e), n, SEED).unwrap();
        let apse = apse_deg.to_radians();
        let (mut outward, mut checked) = (0, 0);
        for i in 0..n as usize {
            assert_eq!(p0.position_km(i), pb.position_km(i), "e = 0 position {i}");
            assert_eq!(
                p0.velocity_km_s(i),
                pb.velocity_km_s(i),
                "e = 0 velocity {i}"
            );
            let pc = pb.position_km(i);
            let (a, lambda) = (pc.norm(), pc.y.atan2(pc.x));
            let m = lambda - apse;
            let (pos, vel) = (pe.position_km(i), pe.velocity_km_s(i));
            let x = (pos.norm() - a) / (a * e);
            assert!(
                (x + m.cos()).abs() < 1e-9,
                "radius of {i}: {x} vs {}",
                -m.cos()
            );
            if m.sin().abs() > 0.1 {
                checked += 1;
                let vr = pos.dot(vel) / pos.norm();
                if (vr > 0.0) == (m.sin() > 0.0) {
                    outward += 1;
                }
            }
        }
        assert!(checked > 4_000, "only {checked} particles checked");
        assert_eq!(outward, checked, "radial speed has the sign of sin M");
    }

    #[test]
    fn core_statistics_of_a_e_and_apse() {
        let f = field();
        let p = generate_particles(&f, &core(), N, SEED).unwrap();
        let (ac, sigma, e_cfg, apse) = (140_220.0, 5.0, 0.0024, 40.0_f64.to_radians());
        let fc = frequencies(&f, ac);
        // Local slope of the specific angular momentum L(a) = a v_circ(a).
        let l = |a: f64| a * circular_speed(&f, a);
        let dl_da = (l(ac + 1.0) - l(ac - 1.0)) / 2.0;
        let (mut a_est, mut e_est, mut w_est, mut lam) = (vec![], vec![], vec![], vec![]);
        for i in 0..p.len() {
            let (pos, vel) = (p.position_km(i), p.velocity_km_s(i));
            let r = pos.norm();
            let vr = pos.dot(vel) / r;
            let theta = pos.y.atan2(pos.x);
            let ecos = 1.0 - r / ac;
            let esin = vr / (ac * fc.kappa_rad_s);
            let m = esin.atan2(ecos);
            let lam_i = theta - 2.0 * (fc.n_rad_s / fc.kappa_rad_s) * esin;
            // r v_theta = a² n (1 - e² cos² M) to first order.
            let ang_mom = pos.cross(vel).z / (1.0 - ecos * ecos);
            a_est.push(ac + (ang_mom - l(ac)) / dl_da);
            e_est.push(ecos.hypot(esin));
            w_est.push(wrap(lam_i - m));
            lam.push(wrap(lam_i) / (2.0 * std::f64::consts::PI));
            assert_eq!(pos.z, 0.0);
        }
        let n = f64::from(N);
        eprintln!(
            "core: mean a {:.3} km, std a {:.3} km, mean e {:.6}, KS lambda {:.5}",
            mean(&a_est),
            std(&a_est),
            mean(&e_est),
            ks_uniform(lam.clone())
        );
        assert!((mean(&a_est) - ac).abs() < 5.0 * sigma / n.sqrt(), "mean a");
        assert!((std(&a_est) / sigma - 1.0).abs() < 0.05, "sigma a");
        assert!((mean(&e_est) / e_cfg - 1.0).abs() < 0.01, "e");
        for &w in &w_est {
            let d = (w - apse + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
                - std::f64::consts::PI;
            assert!(d.abs() < 0.2, "apse {w} {d}");
        }
        let (sx, sy) = (
            mean(&w_est.iter().map(|w| w.cos()).collect::<Vec<_>>()),
            mean(&w_est.iter().map(|w| w.sin()).collect::<Vec<_>>()),
        );
        assert!((sy.atan2(sx) - apse).abs() < 0.01, "mean apse");
        assert!(ks_uniform(lam) < ks_limit(), "mean longitude");
    }

    #[test]
    fn clump_is_a_gaussian_cloud_with_the_circular_velocity() {
        let f = field();
        let (a, long, sigma) = (140_220.0, 30.0_f64, 10.0);
        let p = generate_particles(
            &f,
            &Distribution::Clump {
                semi_major_axis_km: a,
                longitude_deg: long,
                sigma_km: sigma,
            },
            N,
            SEED,
        )
        .unwrap();
        let (s, c) = long.to_radians().sin_cos();
        let v = circular_speed(&f, a);
        let n = f64::from(N);
        let comps = [
            p.x_km.iter().map(|x| x - a * c).collect::<Vec<_>>(),
            p.y_km.iter().map(|y| y - a * s).collect::<Vec<_>>(),
            p.z_km.clone(),
        ];
        for d in &comps {
            assert!(mean(d).abs() < 5.0 * sigma / n.sqrt());
            assert!((std(d) / sigma - 1.0).abs() < 0.04);
        }
        for i in 0..p.len() {
            assert_eq!(p.velocity_km_s(i), Vec3::new(-v * s, v * c, 0.0));
        }
    }

    fn moonlet(velocity: MoonletVelocity) -> Distribution {
        Distribution::Moonlet {
            semi_major_axis_km: 140_222.43,
            longitude_deg: 200.0,
            packing_radius_km: 20.0,
            velocity,
        }
    }

    const MOONLET_VELOCITIES: [MoonletVelocity; 3] = [
        MoonletVelocity::Synchronous,
        MoonletVelocity::Centre,
        MoonletVelocity::Circular,
    ];

    #[test]
    fn moonlet_is_uniform_in_its_sphere() {
        let f = field();
        let (a, r_max) = (140_222.43, 20.0);
        let (s, c) = 200.0_f64.to_radians().sin_cos();
        let p = generate_particles(&f, &moonlet(MoonletVelocity::Synchronous), N, SEED).unwrap();
        let n = f64::from(N);
        let d: Vec<[f64; 3]> = (0..p.len())
            .map(|i| [p.x_km[i] - a * c, p.y_km[i] - a * s, p.z_km[i]])
            .collect();
        let r: Vec<f64> = d
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .collect();
        assert!(r.iter().all(|&x| x <= r_max * (1.0 + 1e-12)));
        // Uniform in volume: (r / R)^3 is uniform in [0, 1].
        let u: Vec<f64> = r.iter().map(|x| (x / r_max).powi(3)).collect();
        assert!(ks_uniform(u) < ks_limit());
        // Isotropic: each component has mean 0 and sd R / sqrt(5).
        for k in 0..3 {
            let comp: Vec<f64> = d.iter().map(|v| v[k]).collect();
            let sd = r_max / 5.0_f64.sqrt();
            assert!(mean(&comp).abs() < 5.0 * sd / n.sqrt(), "axis {k}");
            assert!((std(&comp) / sd - 1.0).abs() < 0.03, "axis {k}");
        }
        // The same draws for every velocity mode: same positions.
        for v in MOONLET_VELOCITIES {
            let q = generate_particles(&f, &moonlet(v), 100, SEED).unwrap();
            for i in 0..100 {
                assert_eq!(q.position_km(i), p.position_km(i));
            }
        }
    }

    #[test]
    fn moonlet_velocities_follow_their_mode() {
        let f = field();
        let a = 140_222.43;
        let (s, c) = 200.0_f64.to_radians().sin_cos();
        let n_c = frequencies(&f, a).n_rad_s;
        let v_c = circular_speed(&f, a);
        for mode in MOONLET_VELOCITIES {
            let p = generate_particles(&f, &moonlet(mode), 1000, SEED).unwrap();
            for i in 0..p.len() {
                let (x, v) = (p.position_km(i), p.velocity_km_s(i));
                assert_eq!(v.z, 0.0);
                match mode {
                    MoonletVelocity::Synchronous => {
                        assert_eq!(v, Vec3::new(-n_c * x.y, n_c * x.x, 0.0));
                    }
                    MoonletVelocity::Centre => {
                        assert_eq!(v, Vec3::new(-v_c * s, v_c * c, 0.0));
                    }
                    MoonletVelocity::Circular => {
                        let r_cyl = x.x.hypot(x.y);
                        let speed = v.x.hypot(v.y);
                        assert!((speed / circular_speed(&f, r_cyl) - 1.0).abs() < 1e-14);
                        assert!((x.x * v.x + x.y * v.y).abs() < 1e-12 * r_cyl * speed);
                    }
                }
            }
        }
        // The centre point of the synchronous mode has the circular speed.
        assert!((n_c * a / v_c - 1.0).abs() < 1e-14);
    }

    /// The guiding-centre offsets of ICR-008 (Hill's equations with the exact
    /// n and kappa): `a_g - a = k (n / kappa)^2 x0` with k = 4, 2 for the
    /// synchronous and centre modes, and `a_g - a = x0` for the circular mode.
    /// The guiding centre comes from the angular momentum h = x vy - y vx,
    /// linearised: `dh/da = a kappa^2 / (2 n)` for circular orbits.
    #[test]
    fn moonlet_guiding_centres_spread_as_hill_predicts() {
        let f = field();
        let a = 140_222.43;
        let fr = frequencies(&f, a);
        let (n, kappa) = (fr.n_rad_s, fr.kappa_rad_s);
        let h_c = a * circular_speed(&f, a);
        let dh_da = a * kappa * kappa / (2.0 * n);
        let q = (n / kappa).powi(2);
        assert!((q - 1.009).abs() < 0.001, "(n/kappa)^2 = {q}");
        for (mode, want) in [
            (MoonletVelocity::Synchronous, 4.0 * q),
            (MoonletVelocity::Centre, 2.0 * q),
            (MoonletVelocity::Circular, 1.0),
        ] {
            let p = generate_particles(&f, &moonlet(mode), N, SEED).unwrap();
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for i in 0..p.len() {
                let (x, v) = (p.position_km(i), p.velocity_km_s(i));
                let x0 = x.x.hypot(x.y) - a;
                let da_g = (x.x * v.y - x.y * v.x - h_c) / dh_da;
                sxy += x0 * da_g;
                sxx += x0 * x0;
            }
            let slope = sxy / sxx;
            assert!(
                (slope / want - 1.0).abs() < 1e-3,
                "{mode:?}: slope {slope}, want {want}"
            );
        }
    }

    #[test]
    fn same_seed_same_particles_and_each_particle_stands_alone() {
        let f = field();
        let a = generate_particles(&f, &core(), 100, SEED).unwrap();
        let b = generate_particles(&f, &core(), 100, SEED).unwrap();
        let c = generate_particles(&f, &core(), 100, SEED + 1).unwrap();
        let short = generate_particles(&f, &core(), 10, SEED).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        for i in 0..10 {
            assert_eq!(short.position_km(i), a.position_km(i));
            assert_eq!(short.velocity_km_s(i), a.velocity_km_s(i));
        }
        assert_eq!(initial_particle(&f, &core(), SEED, 7).0, a.position_km(7));
    }
}
