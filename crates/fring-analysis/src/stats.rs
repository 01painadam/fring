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
