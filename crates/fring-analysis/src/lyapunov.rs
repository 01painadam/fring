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

/// Step in km of the central differences for the Jacobian.
const JV_STEP_KM: f64 = 1.0;

/// As [`run_pair`], but with the tangent vector instead of a shadow orbit.
///
/// The state is the base trajectory `x` (6 positions) and the tangent
/// vector `u` (6 position offsets). The force depends on positions only, so
/// `u'' = J(x) u`, with the Jacobian `J` from central differences along the
/// six axes (`h = JV_STEP_KM`; relative error about 1e-10). (A difference
/// along `u` itself was tried first: it is not smooth where `u` passes through
/// zero, and IAS15 collapsed to steps of 1e-7 s.) The two integrations are one IAS15 system of
/// dimension 12. Nothing here differences two nearby orbits, so there is no
/// rounding noise to be amplified by the Kepler shear. The tangent vector is
/// renormalised to `d0_km` every `renorm_years` and IAS15 restarts (a restart
/// changes the base orbit by about 1e-12 relative, which does not matter for
/// the exponent). The scaled velocity part (`v / n`) is part of the norm as in
/// [`run_pair`].
pub fn run_pair_tangent(
    saturn: &SaturnParams,
    moons: &[MoonInit],
    cfg: &LyapunovConfig,
    index: u32,
) -> Result<Pair> {
    let nm = moons.len();
    let dim = 3 * nm;
    let sys = MoonSystem::new(saturn, moons.iter().map(|m| m.gm_km3_s2).collect());
    let rng = Philox::from_seed(cfg.seed);
    let g = gaussians(&rng, index, 3 * dim);

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
    let r0 = (x0[0] * x0[0] + x0[1] * x0[1] + x0[2] * x0[2]).sqrt();
    let n_scale = (saturn.gm_km3_s2 / (r0 * r0 * r0)).sqrt();
    let dir = &g[dim..];
    let norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();

    // State of the 2*dim system: [x, u] and [v, w].
    let mut xs: Vec<f64> = x0.clone();
    xs.extend((0..dim).map(|k| cfg.d0_km * dir[k] / norm));
    let mut vs: Vec<f64> = v0.clone();
    vs.extend((0..dim).map(|k| n_scale * cfg.d0_km * dir[dim + k] / norm));

    let (mut xp, mut ap, mut am) = (vec![0.0; dim], vec![0.0; dim], vec![0.0; dim]);
    let mut jac = vec![0.0; dim * dim];
    let mut force = |y: &[f64], a: &mut [f64]| {
        let (x, u) = y.split_at(dim);
        sys.accelerations(x, &mut a[..dim]);
        // Jacobian column by column (central differences along the axes), so that
        // the force on u is exactly linear in u and smooth in time.
        for k in 0..dim {
            xp.copy_from_slice(x);
            xp[k] = x[k] + JV_STEP_KM;
            sys.accelerations(&xp, &mut ap);
            xp[k] = x[k] - JV_STEP_KM;
            sys.accelerations(&xp, &mut am);
            for i in 0..dim {
                jac[i * dim + k] = (ap[i] - am[i]) / (2.0 * JV_STEP_KM);
            }
        }
        for i in 0..dim {
            a[dim + i] = (0..dim).map(|k| jac[i * dim + k] * u[k]).sum();
        }
    };

    let dt0 = 0.01 * (r0 * r0 * r0 / saturn.gm_km3_s2).sqrt();
    let n_int = (cfg.years / cfg.renorm_years).round() as usize;
    let dt_int = cfg.renorm_years * YEAR_S;
    let mut t_prev = 0.0;
    let mut ias = Ias15::new(cfg.epsilon, 0.0, &xs, &vs, dt0)?;
    let (mut y, mut w) = (vec![0.0; 2 * dim], vec![0.0; 2 * dim]);
    let (mut l_sum, mut s_sum, mut y_sum) = (0.0, 0.0, 0.0);
    let mut out = Pair::default();
    for j in 1..=n_int {
        let t_end = j as f64 * dt_int;
        while ias.time_s() < t_end {
            ias.step(&mut force)?;
        }
        out.steps += ias.steps();
        ias.dense_output(t_end, &mut y, &mut w);
        let mut d2 = 0.0;
        for k in 0..dim {
            d2 += y[dim + k].powi(2) + (w[dim + k] / n_scale).powi(2);
        }
        let d = d2.sqrt();
        if !(d.is_finite() && d > 0.0) {
            return Err(Error::Backend(format!("pair {index}: tangent norm {d}")));
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
            y[dim + k] *= f;
            w[dim + k] *= f;
        }
        ias = Ias15::new(cfg.epsilon, t_end, &y, &w, dt0)?;
        t_prev = t_end;
    }
    let _ = t_prev;
    Ok(out)
}

/// The two methods of this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Shadow orbit ([`run_pair`]).
    Shadow,
    /// Tangent vector ([`run_pair_tangent`]).
    Tangent,
}

/// Runs pairs `0..n_pairs` on `threads` threads; the result is in pair order.
pub fn run_all(
    saturn: &SaturnParams,
    moons: &[MoonInit],
    cfg: &LyapunovConfig,
    n_pairs: u32,
    threads: usize,
    method: Method,
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
                    let t0 = std::time::Instant::now();
                    let r = match method {
                        Method::Shadow => run_pair(saturn, moons, cfg, i),
                        Method::Tangent => run_pair_tangent(saturn, moons, cfg, i),
                    };
                    if let Ok(p) = &r {
                        eprintln!(
                            "pair {i}: {:.1} s, {} steps, L({} yr) = {:.2}",
                            t0.elapsed().as_secs_f64(),
                            p.steps,
                            p.t_years.last().copied().unwrap_or(0.0),
                            p.log_growth.last().copied().unwrap_or(0.0)
                        );
                    }
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

/// The GR03 rate (task A4-ANALYSIS): least-squares fit of
/// `L(t) = lambda t + b ln t + a` to the points with `lo <= t <= hi` and
/// `t > 0` (years). The `b ln t` term is the Kepler shear (GR03a sec 2.2: the
/// no-chaos curve is "constant + (log t)/t" for `L/t`). Returns
/// `(lambda, b)` in 1/yr and dimensionless, or `None` with fewer than 4 points.
#[allow(clippy::needless_range_loop)] // indices of a 3x3 matrix
pub fn gr_fit(t: &[f64], l: &[f64], lo: f64, hi: f64) -> Option<(f64, f64)> {
    let mut m = [[0.0f64; 4]; 3];
    let mut n = 0;
    for (&ti, &li) in t.iter().zip(l) {
        if ti <= 0.0 || ti < lo || ti > hi {
            continue;
        }
        let x = [ti, ti.ln(), 1.0];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += x[i] * x[j];
            }
            m[i][3] += x[i] * li;
        }
        n += 1;
    }
    if n < 4 {
        return None;
    }
    // Gaussian elimination with partial pivoting.
    for c in 0..3 {
        let piv = (c..3).max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs()))?;
        m.swap(c, piv);
        if m[c][c].abs() < 1e-300 {
            return None;
        }
        for r in 0..3 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..4 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    Some((m[0][3] / m[0][0], m[1][3] / m[1][1]))
}

/// The GR03 anchored rate: `c` of the curve `c + (ln t)/t` that passes through
/// the last point of `L/t`, so `c = L(T)/T - ln(T)/T` (1/yr).
pub fn gr_anchor(t: &[f64], l: &[f64]) -> f64 {
    let (tt, ll) = (t[t.len() - 1], l[l.len() - 1]);
    ll / tt - tt.ln() / tt
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

    /// The tangent method on the Kepler problem: MEGNO tends to 2 as well.
    #[test]
    fn tangent_kepler_is_regular() {
        let (saturn, moons) = two_moons();
        let cfg = LyapunovConfig {
            years: 25.0,
            renorm_years: 0.25,
            d0_km: 1.0,
            kick_km: 1.0,
            epsilon: 1e-9,
            seed: 7,
        };
        let p = run_pair_tangent(&saturn, &moons, &cfg, 0).unwrap();
        let y = *p.megno_mean.last().unwrap();
        assert!(y > 1.5 && y < 3.0, "mean MEGNO {y}");
    }

    /// The GR03 fit recovers lambda from a synthetic L = lambda t + b ln t + a.
    #[test]
    fn gr_fit_recovers_rate() {
        let t: Vec<f64> = (1..=400).map(|k| 0.25 * f64::from(k)).collect();
        let l: Vec<f64> = t.iter().map(|x| 0.3 * x + 2.0 * x.ln() + 1.0).collect();
        let (lam, b) = gr_fit(&t, &l, 0.0, 100.0).unwrap();
        assert!(
            (lam - 0.3).abs() < 1e-9 && (b - 2.0).abs() < 1e-7,
            "{lam} {b}"
        );
        let (lam1, _) = gr_fit(&t, &l, 1.0, 100.0).unwrap();
        assert!((lam1 - 0.3).abs() < 1e-9);
        // Pure shear: L = ln(t) + 1 gives lambda = 0 from the fit.
        let l0: Vec<f64> = t.iter().map(|x| x.ln() + 1.0).collect();
        let (lam0, b0) = gr_fit(&t, &l0, 0.0, 100.0).unwrap();
        assert!(lam0.abs() < 1e-9 && (b0 - 1.0).abs() < 1e-7);
        // The anchored rate of L = c t + ln t is c.
        let la: Vec<f64> = t.iter().map(|x| 0.3 * x + x.ln()).collect();
        assert!((gr_anchor(&t, &la) - 0.3).abs() < 1e-12);
        assert!(gr_fit(&t[..3], &l[..3], 0.0, 100.0).is_none());
    }
}
