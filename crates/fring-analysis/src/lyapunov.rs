//! Lyapunov exponent of the moons Prometheus and Pandora (Saturn J2 to J6
//! plus the two moons) with the Rust IAS15 integrator.
//!
//! One pair is a base trajectory and a shadow trajectory in the 12D phase
//! space (positions in km; velocities scaled by 1/n so that the distance
//! `d = sqrt(|dx|^2 + |dv/n|^2)` is in km). Every `renorm_years` the shadow is
//! moved back to the distance `d0` along the same direction (Benettin), and
//! `l_j = ln(d_j / d0)` is stored. MEGNO uses the discrete form of
//! `docs/adr/0008-chaos-indicator.md`.
//!
//! Each pair has its own base trajectory: the nominal start state plus a
//! Gaussian position kick (the Philox stream [`crate::ANALYSIS_STREAM`], counter
//! index = pair index). Directions on one base trajectory would align with
//! the same unstable vector after a few Lyapunov times, so one shadow per
//! base trajectory gives independent samples.

use fring_model::physics::{Ias15, MoonInit, MoonSystem};
use fring_model::rng::{Counter, Philox, box_muller};
use fring_model::{Error, Result, SaturnParams};

/// Seconds in a Julian year.
pub const YEAR_S: f64 = 365.25 * 86_400.0;

/// Settings of a run.
#[derive(Clone, Copy, Debug)]
pub struct LyapunovConfig {
    /// Length of each run in years.
    pub years: f64,
    /// Renormalisation interval in years.
    pub renorm_years: f64,
    /// Initial and renormalised distance d0 in km.
    pub d0_km: f64,
    /// Standard deviation of the position kick of the base trajectory, km.
    pub kick_km: f64,
    /// IAS15 accuracy parameter.
    pub epsilon: f64,
    /// Run seed.
    pub seed: u64,
}

/// The series of one pair, one entry per renormalisation.
#[derive(Clone, Debug, Default)]
pub struct Pair {
    /// Time in years at the end of each interval.
    pub t_years: Vec<f64>,
    /// `L = sum l_j` (dimensionless).
    pub log_growth: Vec<f64>,
    /// Mean MEGNO (dimensionless).
    pub megno_mean: Vec<f64>,
    /// IAS15 steps of the base trajectory.
    pub steps: u64,
}

fn gaussians(rng: &Philox, index: u32, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n + 1);
    let mut draw = 0u16;
    while out.len() < n {
        let [u1, u2] = rng.uniform_f64(Counter {
            index,
            step: 0,
            stream: crate::ANALYSIS_STREAM,
            draw,
        });
        let (a, b) = box_muller(u1, u2);
        out.push(a);
        out.push(b);
        draw += 1;
    }
    out.truncate(n);
    out
}

/// Runs pair number `index`.
pub fn run_pair(
    saturn: &SaturnParams,
    moons: &[MoonInit],
    cfg: &LyapunovConfig,
    index: u32,
) -> Result<Pair> {
    let nm = moons.len();
    let dim = 3 * nm;
    let sys = MoonSystem::new(saturn, moons.iter().map(|m| m.gm_km3_s2).collect());
    let mut force = |x: &[f64], a: &mut [f64]| sys.accelerations(x, a);
    let rng = Philox::from_seed(cfg.seed);
    let g = gaussians(&rng, index, dim + 2 * dim);

    let mut x0 = Vec::with_capacity(dim);
    let mut v0 = Vec::with_capacity(dim);
    for m in moons {
        let (p, v) = (m.state.position_km, m.state.velocity_km_s);
        x0.extend([p.x, p.y, p.z]);
        v0.extend([v.x, v.y, v.z]);
    }
    for (k, x) in x0.iter_mut().enumerate() {
        *x += cfg.kick_km * g[k];
    }
    // Velocity scale: the mean motion of the first moon.
    let r0 = (x0[0] * x0[0] + x0[1] * x0[1] + x0[2] * x0[2]).sqrt();
    let n_scale = (saturn.gm_km3_s2 / (r0 * r0 * r0)).sqrt();

    // Shadow direction in the scaled 12D space.
    let dir = &g[dim..];
    let norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut xs: Vec<f64> = (0..dim)
        .map(|k| x0[k] + cfg.d0_km * dir[k] / norm)
        .collect();
    let mut vs: Vec<f64> = (0..dim)
        .map(|k| v0[k] + n_scale * cfg.d0_km * dir[dim + k] / norm)
        .collect();

    let dt0 = 0.01 * (r0 * r0 * r0 / saturn.gm_km3_s2).sqrt();
    let mut base = Ias15::new(cfg.epsilon, 0.0, &x0, &v0, dt0)?;
    let mut t_prev = 0.0;
    let n_int = (cfg.years / cfg.renorm_years).round() as usize;
    let dt_int = cfg.renorm_years * YEAR_S;

    let (mut xb, mut vb) = (vec![0.0; dim], vec![0.0; dim]);
    let (mut xe, mut ve) = (vec![0.0; dim], vec![0.0; dim]);
    let (mut l_sum, mut s_sum, mut y_sum) = (0.0, 0.0, 0.0);
    let mut out = Pair::default();
    for j in 1..=n_int {
        let t_end = j as f64 * dt_int;
        while base.time_s() < t_end {
            base.step(&mut force)?;
        }
        base.dense_output(t_end, &mut xb, &mut vb);
        // A restart needs a small first step: with the base step (1,500 s on a
        // 52,000 s orbit) the first step of a fresh IAS15 is wrong by about 1e-4.
        let mut shadow = Ias15::new(cfg.epsilon, t_prev, &xs, &vs, dt0)?;
        while shadow.time_s() < t_end {
            shadow.step(&mut force)?;
        }
        shadow.dense_output(t_end, &mut xe, &mut ve);
        let mut d2 = 0.0;
        for k in 0..dim {
            d2 += (xe[k] - xb[k]).powi(2) + ((ve[k] - vb[k]) / n_scale).powi(2);
        }
        let d = d2.sqrt();
        if !(d.is_finite() && d > 0.0) {
            return Err(Error::Backend(format!("pair {index}: separation {d}")));
        }
        let l = (d / cfg.d0_km).ln();
        let jf = j as f64;
        l_sum += l;
        s_sum += (jf - 0.5) * l;
        y_sum += 2.0 * s_sum / jf;
        out.t_years.push(t_end / YEAR_S);
        out.log_growth.push(l_sum);
        out.megno_mean.push(y_sum / jf);
        let f = cfg.d0_km / d;
        for k in 0..dim {
            xs[k] = xb[k] + (xe[k] - xb[k]) * f;
            vs[k] = vb[k] + (ve[k] - vb[k]) * f;
        }
        t_prev = t_end;
    }
    out.steps = base.steps();
    Ok(out)
}

/// Runs pairs `0..n_pairs` on `threads` threads; the result is in pair order.
pub fn run_all(
    saturn: &SaturnParams,
    moons: &[MoonInit],
    cfg: &LyapunovConfig,
    n_pairs: u32,
    threads: usize,
) -> Result<Vec<Pair>> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};
    let next = AtomicU32::new(0);
    let results: Mutex<Vec<Option<Result<Pair>>>> =
        Mutex::new((0..n_pairs).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n_pairs {
                        break;
                    }
                    let r = run_pair(saturn, moons, cfg, i);
                    results.lock().expect("lock")[i as usize] = Some(r);
                }
            });
        }
    });
    results
        .into_inner()
        .expect("lock")
        .into_iter()
        .map(|r| r.expect("pair done"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fring_model::physics::MoonInit;
    use fring_model::{BodyState, Vec3};

    fn two_moons() -> (SaturnParams, Vec<MoonInit>) {
        let saturn = SaturnParams {
            gm_km3_s2: 3.7931206234e7,
            reference_radius_km: 60330.0,
            j2: 0.0,
            j4: 0.0,
            j6: 0.0,
        };
        let mk = |name: &str, r: f64, phase: f64| {
            let v = (saturn.gm_km3_s2 / r).sqrt();
            MoonInit {
                name: name.into(),
                gm_km3_s2: 0.0,
                state: BodyState {
                    position_km: Vec3::new(r * phase.cos(), r * phase.sin(), 0.0),
                    velocity_km_s: Vec3::new(-v * phase.sin(), v * phase.cos(), 0.0),
                },
            }
        };
        (
            saturn,
            vec![mk("a", 139_400.0, 0.0), mk("b", 141_700.0, 2.0)],
        )
    }

    /// Massless "moons" on Kepler orbits: no chaos. The log growth is the
    /// Kepler shear only, so MEGNO tends to 2 (slowly: the first interval adds about ln(shear) / j) and does not grow like lambda t.
    #[test]
    fn kepler_problem_is_regular() {
        let (saturn, moons) = two_moons();
        let cfg = LyapunovConfig {
            years: 25.0,
            renorm_years: 0.25,
            d0_km: 1.0e-3,
            kick_km: 1.0,
            epsilon: 1e-9,
            seed: 7,
        };
        let p = run_pair(&saturn, &moons, &cfg, 0).unwrap();
        assert_eq!(p.t_years.len(), 100);
        let y = *p.megno_mean.last().unwrap();
        assert!(y > 1.5 && y < 3.0, "mean MEGNO {y}");
    }
}
