//! Apse longitude of the ring from frame files.
//!
//! For particles that start apse-aligned with the same e, the radius as a
//! function of the longitude is `rho = c0 - A cos(phi - varpi)` at all
//! times (each particle's epicycle phase and mean longitude advance at kappa
//! and n; the combination shows the ring pattern turning at `varpi_dot`). A
//! linear least-squares fit of `rho` against `cos phi` and `sin phi` over all
//! particles gives `varpi` and `e = A / c0` at each frame. Frame files hold
//! Cartesian positions in the rotating frame (f32, so `rho` is good to about
//! 0.01 km), so `rho = hypot(x, y)`, `phi = atan2(y, x)` and the inertial apse
//! longitude is `varpi_rot + n0 t`.

use fring_io::RunDir;
use fring_model::{Error, Result};

use crate::stats::{jackknife_slope_se, mean, ols};

/// Seconds per day.
pub const DAY_S: f64 = 86_400.0;

/// The apse longitude and eccentricity against time.
#[derive(Clone, Debug, Default)]
pub struct Series {
    /// Time in days.
    pub t_days: Vec<f64>,
    /// Inertial apse longitude in degrees, unwrapped.
    pub varpi_deg: Vec<f64>,
    /// Eccentricity `A / c0`.
    pub e: Vec<f64>,
}

/// Fit of `rho = c0 + c1 cos(phi) + c2 sin(phi)`. Returns `(c0, c1, c2)`.
pub fn fit_ring(rho: &[f64], phi: &[f64]) -> (f64, f64, f64) {
    // Normal equations 3x3, solved by Cramer's rule.
    let n = rho.len() as f64;
    let (mut sc, mut ss, mut scc, mut sss, mut scs) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sr, mut src, mut srs) = (0.0, 0.0, 0.0);
    for (&r, &p) in rho.iter().zip(phi) {
        let (s, c) = p.sin_cos();
        sc += c;
        ss += s;
        scc += c * c;
        sss += s * s;
        scs += c * s;
        sr += r;
        src += r * c;
        srs += r * s;
    }
    let m = [[n, sc, ss], [sc, scc, scs], [ss, scs, sss]];
    let b = [sr, src, srs];
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&m);
    let mut out = [0.0; 3];
    for (col, o) in out.iter_mut().enumerate() {
        let mut mm = m;
        for row in 0..3 {
            mm[row][col] = b[row];
        }
        *o = det(&mm) / d;
    }
    (out[0], out[1], out[2])
}

/// Reads all frames of a run. `parity` keeps only particles with
/// `id % 2 == parity` (for a split-half check).
pub fn series(run: &RunDir, parity: Option<u32>) -> Result<Series> {
    let mut out = Series::default();
    let mut prev: Option<f64> = None;
    let mut offset = 0.0;
    for idx in run.list_frames()? {
        let f = run.read_frame_file(idx)?;
        let (mut rho, mut phi) = (Vec::new(), Vec::new());
        for i in 0..f.len() {
            if parity.is_some_and(|p| f.id[i] % 2 != p) {
                continue;
            }
            let (x, y) = (f64::from(f.x_km[i]), f64::from(f.y_km[i]));
            rho.push(x.hypot(y));
            phi.push(y.atan2(x));
        }
        if rho.len() < 10 {
            return Err(Error::Input("apse: fewer than 10 particles".into()));
        }
        let (c0, c1, c2) = fit_ring(&rho, &phi);
        let raw = (-c2).atan2(-c1) + f.n0_rad_s * (f.time_s - f.t_ref_s);
        let mut v = raw.rem_euclid(2.0 * std::f64::consts::PI) + offset;
        if let Some(p) = prev {
            // Unwrap: the step between frames is far below pi.
            while v - p > std::f64::consts::PI {
                v -= 2.0 * std::f64::consts::PI;
                offset -= 2.0 * std::f64::consts::PI;
            }
            while v - p < -std::f64::consts::PI {
                v += 2.0 * std::f64::consts::PI;
                offset += 2.0 * std::f64::consts::PI;
            }
        }
        prev = Some(v);
        out.t_days.push(f.time_s / DAY_S);
        out.varpi_deg.push(v.to_degrees());
        out.e.push(c1.hypot(c2) / c0);
    }
    Ok(out)
}

/// Result of the linear fit of the apse longitude.
#[derive(Clone, Debug)]
pub struct Fit {
    /// Slope in degrees per day.
    pub rate_deg_per_day: f64,
    /// Delete-one-block jackknife standard error (10 blocks), deg/day.
    pub se_jackknife: f64,
    /// Plain least-squares standard error (white noise), deg/day.
    pub se_ols: f64,
    /// RMS of the residual about the line, degrees.
    pub rms_resid_deg: f64,
    /// Mean eccentricity.
    pub e_mean: f64,
    /// Modulus of the mean complex e-vector in the co-rotating frame of the
    /// fit (the free eccentricity).
    pub e_free: f64,
    /// RMS of the deviation of the e-vector from its mean (the forced and
    /// short-period part).
    pub e_dev_rms: f64,
    /// Slopes over 5 consecutive windows, deg/day.
    pub window_rates: Vec<f64>,
}

/// Fits a [`Series`].
pub fn fit(s: &Series) -> Fit {
    let (slope, icpt) = ols(&s.t_days, &s.varpi_deg);
    let resid: Vec<f64> = s
        .t_days
        .iter()
        .zip(&s.varpi_deg)
        .map(|(t, v)| v - (icpt + slope * t))
        .collect();
    let n = resid.len() as f64;
    let rms = (resid.iter().map(|r| r * r).sum::<f64>() / n).sqrt();
    let tm = mean(&s.t_days);
    let sxx: f64 = s.t_days.iter().map(|t| (t - tm) * (t - tm)).sum();
    let se_ols = rms / sxx.sqrt();
    // e-vector in the frame that turns with the fitted line.
    let (mut zx, mut zy) = (Vec::new(), Vec::new());
    for (r, e) in resid.iter().zip(&s.e) {
        zx.push(e * r.to_radians().cos());
        zy.push(e * r.to_radians().sin());
    }
    let (mx, my) = (mean(&zx), mean(&zy));
    let dev = (zx
        .iter()
        .zip(&zy)
        .map(|(x, y)| (x - mx).powi(2) + (y - my).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let nw = s.t_days.len() / 5;
    let window_rates = (0..5)
        .map(|w| {
            ols(
                &s.t_days[w * nw..(w + 1) * nw],
                &s.varpi_deg[w * nw..(w + 1) * nw],
            )
            .0
        })
        .collect();
    Fit {
        rate_deg_per_day: slope,
        se_jackknife: jackknife_slope_se(&s.t_days, &s.varpi_deg, 10),
        se_ols,
        rms_resid_deg: rms,
        e_mean: mean(&s.e),
        e_free: mx.hypot(my),
        e_dev_rms: dev,
        window_rates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_fit_recovers_the_pattern() {
        let (c0, a, varpi) = (140_222.0, 330.0, 1.2_f64);
        let phi: Vec<f64> = (0..500).map(|i| i as f64 * 0.0125663706).collect();
        let rho: Vec<f64> = phi.iter().map(|p| c0 - a * (p - varpi).cos()).collect();
        let (f0, c1, c2) = fit_ring(&rho, &phi);
        assert!((f0 - c0).abs() < 1e-6);
        assert!((c1.hypot(c2) - a).abs() < 1e-6);
        assert!(((-c2).atan2(-c1) - varpi).abs() < 1e-9);
    }
}
