//! Small statistics helpers.

/// Least-squares line `y = intercept + slope x`. Returns `(slope, intercept)`.
///
/// # Panics
/// If the slices differ in length or have fewer than 2 points.
pub fn ols(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert!(x.len() == y.len() && x.len() >= 2);
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxx += (a - mx) * (a - mx);
        sxy += (a - mx) * (b - my);
    }
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}

/// Mean of a slice (NaN if empty).
pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Sample standard deviation (n - 1).
pub fn std(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

/// The `q`-quantile (0 to 1) by linear interpolation. Sorts a copy.
pub fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return f64::NAN;
    }
    s.sort_by(f64::total_cmp);
    let pos = q * (s.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

/// Delete-one-block jackknife standard error of the OLS slope: the data are
/// cut into `blocks` contiguous blocks; each block is left out in turn.
pub fn jackknife_slope_se(x: &[f64], y: &[f64], blocks: usize) -> f64 {
    let n = x.len();
    let mut slopes = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let (lo, hi) = (b * n / blocks, (b + 1) * n / blocks);
        let xs: Vec<f64> = (0..n)
            .filter(|i| *i < lo || *i >= hi)
            .map(|i| x[i])
            .collect();
        let ys: Vec<f64> = (0..n)
            .filter(|i| *i < lo || *i >= hi)
            .map(|i| y[i])
            .collect();
        slopes.push(ols(&xs, &ys).0);
    }
    let m = mean(&slopes);
    let k = blocks as f64;
    ((k - 1.0) / k * slopes.iter().map(|s| (s - m) * (s - m)).sum::<f64>()).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_recovers_a_line() {
        let x: Vec<f64> = (0..50).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|v| 3.0 + 0.25 * v).collect();
        let (s, c) = ols(&x, &y);
        assert!((s - 0.25).abs() < 1e-12 && (c - 3.0).abs() < 1e-10);
        assert!(jackknife_slope_se(&x, &y, 5) < 1e-12);
    }

    #[test]
    fn quantile_of_a_ramp() {
        let v: Vec<f64> = (0..=100).map(f64::from).collect();
        assert!((quantile(&v, 0.5) - 50.0).abs() < 1e-12);
        assert!((quantile(&v, 0.1) - 10.0).abs() < 1e-12);
    }
}

/// Complementary error function (Chebyshev fit of Numerical Recipes, relative
/// error below 1.2e-7).
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// Two-sided p value of a standard normal `z`.
pub fn p_two_sided(z: f64) -> f64 {
    erfc(z.abs() / std::f64::consts::SQRT_2)
}

/// Two-proportion z test (pooled variance) of `k1` of `n1` against `k2` of
/// `n2`. Returns `(difference p1 - p2, sd of the difference, z, p)`.
/// The sd is the unpooled counting sd, `sqrt(p1 q1 / n1 + p2 q2 / n2)`.
///
/// # Panics
/// If `n1` or `n2` is 0.
pub fn two_proportion(k1: usize, n1: usize, k2: usize, n2: usize) -> (f64, f64, f64, f64) {
    assert!(n1 > 0 && n2 > 0);
    let (p1, p2) = (k1 as f64 / n1 as f64, k2 as f64 / n2 as f64);
    let pool = (k1 + k2) as f64 / (n1 + n2) as f64;
    let se_pool = (pool * (1.0 - pool) * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
    let sd = (p1 * (1.0 - p1) / n1 as f64 + p2 * (1.0 - p2) / n2 as f64).sqrt();
    let z = if se_pool > 0.0 {
        (p1 - p2) / se_pool
    } else {
        0.0
    };
    (p1 - p2, sd, z, p_two_sided(z))
}

/// Two-sample Kolmogorov-Smirnov test. Returns `(D, p)` with the asymptotic
/// p value (Kolmogorov distribution, effective n = n1 n2 / (n1 + n2)).
pub fn ks_two_sample(a: &[f64], b: &[f64]) -> (f64, f64) {
    let mut x: Vec<f64> = a.iter().copied().filter(|v| v.is_finite()).collect();
    let mut y: Vec<f64> = b.iter().copied().filter(|v| v.is_finite()).collect();
    x.sort_by(f64::total_cmp);
    y.sort_by(f64::total_cmp);
    let (n1, n2) = (x.len() as f64, y.len() as f64);
    let (mut i, mut j, mut d) = (0usize, 0usize, 0.0f64);
    while i < x.len() && j < y.len() {
        let v = x[i].min(y[j]);
        while i < x.len() && x[i] <= v {
            i += 1;
        }
        while j < y.len() && y[j] <= v {
            j += 1;
        }
        d = d.max((i as f64 / n1 - j as f64 / n2).abs());
    }
    let ne = n1 * n2 / (n1 + n2);
    let lam = (ne.sqrt() + 0.12 + 0.11 / ne.sqrt()) * d;
    let mut p = 0.0;
    for k in 1..=100 {
        let kf = k as f64;
        let term = 2.0 * (-2.0 * kf * kf * lam * lam).exp();
        p += if k % 2 == 1 { term } else { -term };
    }
    (d, p.clamp(0.0, 1.0))
}

/// Holm step-down adjusted p values (same order as the input).
pub fn holm(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&i, &j| p[i].total_cmp(&p[j]));
    let mut out = vec![0.0; m];
    let mut run = 0.0f64;
    for (rank, &i) in idx.iter().enumerate() {
        run = run.max(((m - rank) as f64 * p[i]).min(1.0));
        out[i] = run;
    }
    out
}

/// Bootstrap 16 % to 84 % interval (one sigma) of the median of `v`, with
/// `n_boot` resamples. The draws come from the seeded Philox stream
/// [`crate::ANALYSIS_STREAM`]. Returns `(median, lo, hi)`.
pub fn bootstrap_median_ci(v: &[f64], n_boot: usize, seed: u64) -> (f64, f64, f64) {
    use fring_model::rng::{Counter, Philox};
    let rng = Philox::from_seed(seed);
    let n = v.len();
    let mut meds = Vec::with_capacity(n_boot);
    let mut buf = vec![0.0; n];
    for b in 0..n_boot {
        for (j, slot) in buf.iter_mut().enumerate() {
            let u = rng.uniform_f64(Counter {
                index: b as u32,
                step: (j / 2) as u64,
                stream: crate::ANALYSIS_STREAM,
                draw: 0,
            })[j % 2];
            *slot = v[((u * n as f64) as usize).min(n - 1)];
        }
        meds.push(quantile(&buf, 0.5));
    }
    (
        quantile(v, 0.5),
        quantile(&meds, 0.1587),
        quantile(&meds, 0.8413),
    )
}

#[cfg(test)]
mod test_stat_tests {
    use super::*;

    #[test]
    fn normal_tail() {
        assert!((p_two_sided(1.959_964) - 0.05).abs() < 1e-6);
        assert!((p_two_sided(0.0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn proportions() {
        // 40 % against 50 %, 5,000 each: sd = sqrt(.24/5000 + .25/5000) = 0.0099.
        let (d, sd, z, p) = two_proportion(2000, 5000, 2500, 5000);
        assert!((d + 0.1).abs() < 1e-12);
        assert!((sd - 0.0099).abs() < 1e-4);
        assert!(z < -9.0 && p < 1e-15);
        let (_, _, z0, p0) = two_proportion(2000, 5000, 2000, 5000);
        assert!(z0 == 0.0 && (p0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ks_and_holm() {
        let a: Vec<f64> = (0..2000).map(|i| i as f64 / 2000.0).collect();
        let b: Vec<f64> = (0..2000).map(|i| (i as f64 + 0.5) / 2000.0).collect();
        let (d, p) = ks_two_sample(&a, &b);
        assert!(d < 0.001 && p > 0.99, "{d} {p}");
        let c: Vec<f64> = a.iter().map(|x| x + 0.2).collect();
        let (d2, p2) = ks_two_sample(&a, &c);
        assert!((d2 - 0.2).abs() < 0.001 && p2 < 1e-10);
        let h = holm(&[0.01, 0.04, 0.03]);
        assert!((h[0] - 0.03).abs() < 1e-12 && (h[1] - 0.06).abs() < 1e-12);
        assert!((h[2] - 0.06).abs() < 1e-12);
    }

    #[test]
    fn bootstrap_covers_the_median() {
        let v: Vec<f64> = (0..1001).map(f64::from).collect();
        let (m, lo, hi) = bootstrap_median_ci(&v, 200, 5);
        assert_eq!(m, 500.0);
        assert!(lo < m && m < hi && hi - lo < 60.0, "{lo} {hi}");
        assert_eq!(bootstrap_median_ci(&v, 200, 5), (m, lo, hi));
    }
}
