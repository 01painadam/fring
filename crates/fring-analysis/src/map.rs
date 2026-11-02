//! The chaos map: bins `chaos.csv` (ICR-003, ADR 0008) by the initial
//! radius and the initial longitude relative to Prometheus.
//!
//! `chaos.csv` holds a Keplerian a computed with the point mass only. For a
//! circular orbit in Saturn's field (J2 to J6) that is about 640 km larger
//! than the orbital radius, so the map uses the radius `|r|` of the initial
//! state instead. The initial states are rebuilt from `config.toml` of the run
//! (counter-based RNG, so the particles are the same).

use fring_io::{ChaosRow, RunDir};
use fring_model::params::saturn_from_toml_str;
use fring_model::physics::{SaturnField, initial_particle};
use fring_model::{Error, Result, RunConfig};
use std::path::Path;

use crate::stats::quantile;

/// Class thresholds on the mean MEGNO (A0 decision, ADR 0008 and A1-ANALYSIS).
pub const BOUNDED_MAX: f64 = 1.5;
/// Upper edge of the regular class.
pub const REGULAR_MAX: f64 = 2.5;
/// Lower edge of the chaotic class.
pub const CHAOTIC_MIN: f64 = 5.0;

/// Class of a particle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Mean MEGNO below 1.5: the separation stays bounded.
    Bounded,
    /// 1.5 to 2.5.
    Regular,
    /// 2.5 to 5.
    Uncertain,
    /// 5 or more.
    Chaotic,
    /// The run skipped an interval (flag bit 0) or the value is not finite.
    Invalid,
}

impl Class {
    /// Index 0 to 4 in the order of the variants.
    pub fn index(self) -> usize {
        self as usize
    }
    /// Short name.
    pub fn name(self) -> &'static str {
        ["bounded", "regular", "uncertain", "chaotic", "invalid"][self.index()]
    }
}

/// Class from the mean MEGNO and the flags.
pub fn classify(megno_mean: f64, flags: u32) -> Class {
    if flags != 0 || !megno_mean.is_finite() {
        Class::Invalid
    } else if megno_mean < BOUNDED_MAX {
        Class::Bounded
    } else if megno_mean <= REGULAR_MAX {
        Class::Regular
    } else if megno_mean < CHAOTIC_MIN {
        Class::Uncertain
    } else {
        Class::Chaotic
    }
}

/// One particle of a map.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    /// Particle id.
    pub id: u32,
    /// Initial radius, km.
    pub r_km: f64,
    /// Initial longitude minus the longitude of Prometheus, degrees in [-180, 180).
    pub lon_deg: f64,
    /// Mean MEGNO.
    pub megno: f64,
    /// FTLE in 1/yr.
    pub ftle_per_yr: f64,
    /// Class.
    pub class: Class,
}

/// Reads a run directory into points.
pub fn read_points(run: &Path) -> Result<Vec<Point>> {
    let dir = RunDir::new(run);
    let rows: Vec<ChaosRow> = dir.read_chaos_file()?;
    let cfg_text = String::from_utf8(dir.read_config_copy()?)
        .map_err(|_| Error::Io("config.toml is not UTF-8".into()))?;
    let cfg = RunConfig::from_toml_str(&cfg_text)?;
    let params = Path::new(&cfg.run.params_dir).join("saturn.toml");
    let text = std::fs::read_to_string(&params)
        .map_err(|e| Error::Io(format!("{}: {e}", params.display())))?;
    let field = SaturnField::<f64>::from_params(&saturn_from_toml_str(&text)?);
    Ok(rows
        .iter()
        .map(|r| {
            let (p, _) = initial_particle(&field, &cfg.particles.distribution, cfg.run.seed, r.id);
            Point {
                id: r.id,
                r_km: p.norm(),
                lon_deg: r.lambda_rel_prometheus_deg,
                megno: r.megno_mean,
                ftle_per_yr: r.ftle_per_day * 365.25,
                class: classify(r.megno_mean, r.flags),
            }
        })
        .collect())
}

/// Writes the 2D map: one row per (radius bin, longitude bin).
pub fn write_map_csv(
    out: &mut impl std::io::Write,
    pts: &[Point],
    r_lo: f64,
    r_hi: f64,
    r_bin: f64,
    lon_bin: f64,
) -> std::io::Result<()> {
    let nr = ((r_hi - r_lo) / r_bin).ceil() as usize;
    let nl = (360.0 / lon_bin).ceil() as usize;
    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); nr * nl];
    for (k, p) in pts.iter().enumerate() {
        if p.class == Class::Invalid || p.r_km < r_lo || p.r_km >= r_hi {
            continue;
        }
        let ir = ((p.r_km - r_lo) / r_bin) as usize;
        let il = (((p.lon_deg + 180.0) / lon_bin) as usize).min(nl - 1);
        cells[ir * nl + il].push(k);
    }
    writeln!(
        out,
        "r_lo_km,r_hi_km,lon_lo_deg,lon_hi_deg,n,median_megno,frac_bounded,frac_regular,frac_uncertain,frac_chaotic,median_ftle_per_yr"
    )?;
    for ir in 0..nr {
        for il in 0..nl {
            let c = &cells[ir * nl + il];
            if c.is_empty() {
                continue;
            }
            let y: Vec<f64> = c.iter().map(|&k| pts[k].megno).collect();
            let f: Vec<f64> = c.iter().map(|&k| pts[k].ftle_per_yr).collect();
            let frac = |cl: Class| {
                c.iter().filter(|&&k| pts[k].class == cl).count() as f64 / c.len() as f64
            };
            writeln!(
                out,
                "{:.3},{:.3},{:.3},{:.3},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.5}",
                r_lo + ir as f64 * r_bin,
                r_lo + (ir + 1) as f64 * r_bin,
                -180.0 + il as f64 * lon_bin,
                -180.0 + (il + 1) as f64 * lon_bin,
                c.len(),
                quantile(&y, 0.5),
                frac(Class::Bounded),
                frac(Class::Regular),
                frac(Class::Uncertain),
                frac(Class::Chaotic),
                quantile(&f, 0.5)
            )?;
        }
    }
    Ok(())
}

/// Writes the 1D profile in radius: class fractions per bin.
pub fn write_profile_csv(
    out: &mut impl std::io::Write,
    pts: &[Point],
    r_lo: f64,
    r_hi: f64,
    r_bin: f64,
) -> std::io::Result<()> {
    let nr = ((r_hi - r_lo) / r_bin).ceil() as usize;
    let mut counts = vec![[0usize; 5]; nr];
    let mut ys: Vec<Vec<f64>> = vec![Vec::new(); nr];
    for p in pts {
        if p.r_km < r_lo || p.r_km >= r_hi {
            continue;
        }
        let ir = ((p.r_km - r_lo) / r_bin) as usize;
        counts[ir][p.class.index()] += 1;
        if p.class != Class::Invalid {
            ys[ir].push(p.megno);
        }
    }
    writeln!(
        out,
        "r_lo_km,r_hi_km,n,frac_bounded,frac_regular,frac_uncertain,frac_chaotic,median_megno"
    )?;
    for (ir, c) in counts.iter().enumerate() {
        let n: usize = c[..4].iter().sum();
        if n == 0 {
            continue;
        }
        let f = |i: usize| c[i] as f64 / n as f64;
        writeln!(
            out,
            "{:.3},{:.3},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
            r_lo + ir as f64 * r_bin,
            r_lo + (ir + 1) as f64 * r_bin,
            n,
            f(0),
            f(1),
            f(2),
            f(3),
            quantile(&ys[ir], 0.5)
        )?;
    }
    Ok(())
}

/// Histogram of the mean MEGNO (bin width `w`, from `lo` to `hi`, values
/// outside go to the end bins).
pub fn write_megno_hist(
    out: &mut impl std::io::Write,
    pts: &[Point],
    lo: f64,
    hi: f64,
    w: f64,
) -> std::io::Result<()> {
    let nb = ((hi - lo) / w).ceil() as usize;
    let mut h = vec![0usize; nb];
    for p in pts.iter().filter(|p| p.class != Class::Invalid) {
        let b = (((p.megno - lo) / w).floor().max(0.0) as usize).min(nb - 1);
        h[b] += 1;
    }
    writeln!(out, "megno_lo,megno_hi,count")?;
    for (b, c) in h.iter().enumerate() {
        writeln!(
            out,
            "{:.3},{:.3},{}",
            lo + b as f64 * w,
            lo + (b + 1) as f64 * w,
            c
        )?;
    }
    Ok(())
}

/// Cross-tabulation of the classes of two runs with the same particles.
pub fn cross_tab(a: &[Point], b: &[Point]) -> [[usize; 5]; 5] {
    let mut m = [[0usize; 5]; 5];
    for (p, q) in a.iter().zip(b) {
        assert_eq!(p.id, q.id);
        m[p.class.index()][q.class.index()] += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert_eq!(classify(1.0, 0), Class::Bounded);
        assert_eq!(classify(2.0, 0), Class::Regular);
        assert_eq!(classify(3.0, 0), Class::Uncertain);
        assert_eq!(classify(8.0, 0), Class::Chaotic);
        assert_eq!(classify(8.0, 1), Class::Invalid);
        assert_eq!(classify(f64::NAN, 0), Class::Invalid);
    }

    fn pt(id: u32, r: f64, lon: f64, y: f64) -> Point {
        Point {
            id,
            r_km: r,
            lon_deg: lon,
            megno: y,
            ftle_per_yr: 0.1,
            class: classify(y, 0),
        }
    }

    #[test]
    fn bins_and_fractions() {
        let pts = vec![
            pt(0, 100.5, -179.0, 2.0),
            pt(1, 100.6, -178.0, 9.0),
            pt(2, 108.0, 5.0, 1.0),
        ];
        let mut buf = Vec::new();
        write_map_csv(&mut buf, &pts, 100.0, 110.0, 5.0, 10.0).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "{text}");
        assert!(lines[1].starts_with("100.000,105.000,-180.000,-170.000,2,"));
        assert!(lines[1].contains(",0.0000,0.5000,0.0000,0.5000,"));
        let tab = cross_tab(&pts, &pts);
        assert_eq!(tab[0][0] + tab[1][1] + tab[3][3], 3);
    }
}
