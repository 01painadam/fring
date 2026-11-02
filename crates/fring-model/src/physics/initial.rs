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
//! All particles have `z = 0` and `vz = 0`, except the clump, which has a
//! Gaussian z offset: the distributions have no inclination field yet.
//!
//! | Distribution | a | e, apse | mean longitude |
//! |---|---|---|---|
//! | `Core` | Gaussian (mean, sigma) | config e, config apse | uniform |
//! | `Band` | uniform in [min, max] | 0 | uniform |
//! | `Clump` | see below | | |
//!
//! `Core`: the epicycle phase is `M = lambda - apse` (the pericentre is where
//! the guiding centre is at the apse longitude).
//!
//! `Clump`: three Gaussian offsets (standard deviation `sigma_km` each, in
//! the inertial x, y, z directions) about the point on the circle of radius
//! `a` at the given inertial longitude. Every particle gets the circular
//! velocity of that centre point (no velocity dispersion).

use super::circular::{circular_speed, frequencies};
use super::gravity::SaturnField;
use crate::config::Distribution;
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
