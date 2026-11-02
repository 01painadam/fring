//! Statistical comparison of two sets of chaos-map particles (A2-ANALYSIS):
//! the chaotic fraction (mean MEGNO >= 5), the median Lyapunov estimate
//! lambda = 2 <Y> / t, and the distribution of ln <Y>. Chaotic orbits are
//! never compared one by one, only as distributions (AGENTS.md).

use crate::map::{Class, Point};
use crate::stats::{bootstrap_median_ci, ks_two_sample, two_proportion};

/// Fewest valid particles in a set for a comparison (M17).
pub const MIN_COUNT: usize = 1000;

/// One comparison of set A with set B.
#[derive(Clone, Debug)]
pub struct Row {
    /// Valid particles in A and B.
    pub n: (usize, usize),
    /// Chaotic fractions of A and B.
    pub frac: (f64, f64),
    /// Difference A - B, its counting sd, z and two-sided p.
    pub dfrac: (f64, f64, f64, f64),
    /// Median lambda in 1/yr with the bootstrap one-sigma interval, for A and B.
    pub lam_a: (f64, f64, f64),
    /// Same for B.
    pub lam_b: (f64, f64, f64),
    /// Difference of the medians and its sigma (the two interval half widths added in quadrature).
    pub dlam: (f64, f64),
    /// KS statistic and p value of ln <Y> (A against B).
    pub ks: (f64, f64),
}

fn valid(p: &[Point]) -> Vec<&Point> {
    p.iter().filter(|q| q.class != Class::Invalid).collect()
}

/// Compares `a` with `b` for a run of `years`. Returns `None` if either set
/// has fewer than `min_n` valid particles.
pub fn compare(a: &[Point], b: &[Point], years: f64, min_n: usize, seed: u64) -> Option<Row> {
    let (va, vb) = (valid(a), valid(b));
    if va.len() < min_n || vb.len() < min_n {
        return None;
    }
    let ch = |v: &[&Point]| v.iter().filter(|q| q.class == Class::Chaotic).count();
    let (ka, kb) = (ch(&va), ch(&vb));
    let (d, sd, z, p) = two_proportion(ka, va.len(), kb, vb.len());
    let lam = |v: &[&Point]| v.iter().map(|q| 2.0 * q.megno / years).collect::<Vec<_>>();
    let (la, lb) = (lam(&va), lam(&vb));
    let ca = bootstrap_median_ci(&la, 200, seed);
    let cb = bootstrap_median_ci(&lb, 200, seed + 1);
    let sa = 0.5 * (ca.2 - ca.1);
    let sb = 0.5 * (cb.2 - cb.1);
    let ln = |v: &[&Point]| {
        v.iter()
            .map(|q| q.megno.max(1e-12).ln())
            .collect::<Vec<_>>()
    };
    Some(Row {
        n: (va.len(), vb.len()),
        frac: (ka as f64 / va.len() as f64, kb as f64 / vb.len() as f64),
        dfrac: (d, sd, z, p),
        lam_a: ca,
        lam_b: cb,
        dlam: (ca.0 - cb.0, (sa * sa + sb * sb).sqrt()),
        ks: ks_two_sample(&ln(&va), &ln(&vb)),
    })
}

/// The header of [`Row::csv`], with `extra` leading columns.
pub fn header(extra: &str) -> String {
    format!(
        "{extra}n_a,n_b,frac_a,frac_b,dfrac,sd_counting,z,p,lam_a,lam_a_lo,lam_a_hi,lam_b,lam_b_lo,lam_b_hi,dlam,dlam_sigma,ks_d,ks_p"
    )
}

impl Row {
    /// One CSV line (no leading columns).
    pub fn csv(&self) -> String {
        format!(
            "{},{},{:.5},{:.5},{:.5},{:.5},{:.3},{:.3e},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3e}",
            self.n.0,
            self.n.1,
            self.frac.0,
            self.frac.1,
            self.dfrac.0,
            self.dfrac.1,
            self.dfrac.2,
            self.dfrac.3,
            self.lam_a.0,
            self.lam_a.1,
            self.lam_a.2,
            self.lam_b.0,
            self.lam_b.1,
            self.lam_b.2,
            self.dlam.0,
            self.dlam.1,
            self.ks.0,
            self.ks.1
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::classify;

    fn pts(n: usize, chaotic: usize, y_reg: f64, y_ch: f64) -> Vec<Point> {
        (0..n)
            .map(|i| {
                let y = if i < chaotic {
                    y_ch + (i % 7) as f64 * 0.1
                } else {
                    y_reg + (i % 5) as f64 * 0.05
                };
                Point {
                    id: i as u32,
                    r_km: 0.0,
                    a_km: 0.0,
                    lon_deg: 0.0,
                    megno: y,
                    ftle_per_yr: 0.0,
                    class: classify(y, 0),
                }
            })
            .collect()
    }

    #[test]
    fn same_sets_agree_and_shifted_sets_differ() {
        let a = pts(4000, 1600, 2.0, 9.0);
        let same = compare(&a, &a, 10.0, MIN_COUNT, 1).unwrap();
        assert!(same.dfrac.2.abs() < 1e-9 && same.dlam.0 == 0.0 && same.ks.1 > 0.99);
        let b = pts(4000, 2000, 2.0, 9.0);
        let r = compare(&a, &b, 10.0, MIN_COUNT, 1).unwrap();
        assert!((r.dfrac.0 + 0.1).abs() < 1e-12 && r.dfrac.2 < -8.0);
        assert!(r.ks.1 < 1e-6);
        assert!(r.n.0 >= MIN_COUNT);
    }

    #[test]
    fn too_few_particles_give_none() {
        let a = pts(500, 100, 2.0, 9.0);
        assert!(compare(&a, &a, 10.0, MIN_COUNT, 1).is_none());
    }
}
