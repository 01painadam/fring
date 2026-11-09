//! A3-ANALYSIS: particles that were flagged and frozen (ICR-004) and the
//! per-window tables of a map that has such particles.
//!
//! A lost particle has bit 1 (value 2) in the `flags` of `chaos.csv`. Its
//! mean MEGNO and FTLE stop at the loss, so it is never used for a rate.
//! The reason and the step of the loss are in `drops.csv` (written by
//! `fring-run`); until its reader exists in `fring-io`, [`lost_from_flags`]
//! gives the list from the flags alone (reason 1, step unknown).

use crate::map::{Class, Point};
use crate::stats::{bootstrap_median_ci, quantile};
use fring_io::{ChaosRow, DropRow, RunDir};
use std::collections::HashMap;
use std::path::Path;

/// Flag bit 1 of `chaos.csv`: the particle (or its shadow) was lost.
pub const FLAG_LOST: u32 = 2;
/// Reason 1 (`fring_model::DROP_UNBOUND`).
pub const REASON_UNBOUND: u8 = 1;

/// One lost particle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lost {
    /// Particle id.
    pub id: u32,
    /// 1 unbound, 2 inside Prometheus, 3 inside Pandora.
    pub reason: u8,
    /// Global step of the loss, `u64::MAX` if unknown.
    pub drop_step: u64,
}

/// The lost list from the flags of `chaos.csv` alone: reason 1 and an unknown
/// step. Right for a run without the radius rule.
pub fn lost_from_flags(rows: &[ChaosRow]) -> Vec<Lost> {
    rows.iter()
        .filter(|r| r.flags & FLAG_LOST != 0)
        .map(|r| Lost {
            id: r.id,
            reason: REASON_UNBOUND,
            drop_step: u64::MAX,
        })
        .collect()
}

/// The lost list from `drops.csv` rows (reason and step are exact).
pub fn lost_from_drops(rows: &[DropRow]) -> Vec<Lost> {
    rows.iter()
        .map(|r| Lost {
            id: r.id,
            reason: r.reason,
            drop_step: r.drop_step,
        })
        .collect()
}

/// Ids in `a` and `b` as sorted lists differ? Returns a message if so.
pub fn check_same_ids(from_drops: &[Lost], from_flags: &[Lost]) -> Result<(), String> {
    let ids = |v: &[Lost]| {
        let mut x: Vec<u32> = v.iter().map(|l| l.id).collect();
        x.sort_unstable();
        x
    };
    let (a, b) = (ids(from_drops), ids(from_flags));
    if a == b {
        Ok(())
    } else {
        Err(format!(
            "drops.csv lists {} particles, flag bit 1 of chaos.csv marks {}; the sets differ",
            a.len(),
            b.len()
        ))
    }
}

/// The lost list of a run directory. With `drops.csv` the list and the
/// reasons come from it, and flag bit 1 of `chaos.csv` is a cross-check (the
/// two sets must be equal, else an error). Without `drops.csv` (a run with no
/// `[loss]`), the flags give the list with reason 1.
pub fn read_lost(run: &Path) -> Result<Vec<Lost>, Box<dyn std::error::Error>> {
    let dir = RunDir::new(run);
    let flags = lost_from_flags(&dir.read_chaos_file()?);
    if run.join("drops.csv").exists() {
        let drops = lost_from_drops(&dir.read_drops_file()?);
        check_same_ids(&drops, &flags)?;
        Ok(drops)
    } else {
        Ok(flags)
    }
}

/// One radius window of the table.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowRow {
    /// Window edges, km (initial orbital radius, [`Point::a_km`]).
    pub lo_km: f64,
    /// Upper edge, km.
    pub hi_km: f64,
    /// Particles in the window.
    pub n: usize,
    /// Lost as unbound.
    pub unbound: usize,
    /// Lost inside Prometheus.
    pub prometheus: usize,
    /// Lost inside Pandora.
    pub pandora: usize,
    /// Flag bit 0 only (skipped interval), not lost.
    pub invalid: usize,
    /// Live and chaotic.
    pub chaotic: usize,
    /// Live with a valid class (the survivors).
    pub live: usize,
    /// Median lambda of the survivors, 1/yr: 2 median(MEGNO) / years.
    pub lambda_per_yr: f64,
    /// One sigma of lambda (half of the 16 to 84 % bootstrap interval of the
    /// median MEGNO, 200 resamples, seed 1), 1/yr.
    pub lambda_sigma_per_yr: f64,
}

impl WindowRow {
    /// Chaotic fraction among the survivors.
    pub fn frac_survivors(&self) -> f64 {
        self.chaotic as f64 / self.live as f64
    }
    /// Chaotic fraction with every lost particle counted as chaotic (an
    /// upper bound: a particle that a moon scatters or absorbs).
    pub fn frac_bound(&self) -> f64 {
        let lost = self.unbound + self.prometheus + self.pandora;
        (self.chaotic + lost) as f64 / (self.live + lost) as f64
    }
}

/// Table over `windows` (lo, hi pairs, may overlap). A particle is in a window
/// if `lo <= a_km < hi`. `years` is the run length.
pub fn window_table(
    pts: &[Point],
    lost: &[Lost],
    windows: &[(f64, f64)],
    years: f64,
) -> Vec<WindowRow> {
    let by_id: HashMap<u32, Lost> = lost.iter().map(|l| (l.id, *l)).collect();
    windows
        .iter()
        .map(|&(lo, hi)| {
            let mut w = WindowRow {
                lo_km: lo,
                hi_km: hi,
                n: 0,
                unbound: 0,
                prometheus: 0,
                pandora: 0,
                invalid: 0,
                chaotic: 0,
                live: 0,
                lambda_per_yr: f64::NAN,
                lambda_sigma_per_yr: f64::NAN,
            };
            let mut y = Vec::new();
            for p in pts.iter().filter(|p| p.a_km >= lo && p.a_km < hi) {
                w.n += 1;
                match by_id.get(&p.id).map(|l| l.reason) {
                    Some(REASON_UNBOUND) => w.unbound += 1,
                    Some(2) => w.prometheus += 1,
                    Some(3) => w.pandora += 1,
                    Some(_) => w.unbound += 1,
                    None if p.class == Class::Invalid => w.invalid += 1,
                    None => {
                        w.live += 1;
                        w.chaotic += usize::from(p.class == Class::Chaotic);
                        y.push(p.megno);
                    }
                }
            }
            if !y.is_empty() {
                w.lambda_per_yr = 2.0 * quantile(&y, 0.5) / years;
                let (_, lo, hi) = bootstrap_median_ci(&y, 200, 1);
                w.lambda_sigma_per_yr = (hi - lo) / years;
            }
            w
        })
        .collect()
}

/// The windows of `docs/results/a1-analysis.md`, plus finer windows at the
/// inner edge where the losses are.
pub fn default_windows() -> Vec<(f64, f64)> {
    vec![
        (139_700.0, 139_950.0),
        (139_950.0, 140_150.0),
        (140_197.0, 140_247.0),
        (140_150.0, 140_700.0),
        (140_700.0, 140_900.0),
        (140_900.0, 141_100.0),
        (139_700.0, 141_100.0),
    ]
}

/// CSV header of [`write_window_csv`].
pub const WINDOW_HEADER: &str = "lo_km,hi_km,n,live,unbound,prometheus,pandora,invalid,chaotic,\
frac_chaotic_survivors,frac_chaotic_bound,lambda_per_yr,lambda_sigma_per_yr";

/// Writes the window table.
pub fn write_window_csv(out: &mut impl std::io::Write, rows: &[WindowRow]) -> std::io::Result<()> {
    writeln!(out, "{WINDOW_HEADER}")?;
    for w in rows {
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{:.5},{:.5},{:.4},{:.4}",
            w.lo_km,
            w.hi_km,
            w.n,
            w.live,
            w.unbound,
            w.prometheus,
            w.pandora,
            w.invalid,
            w.chaotic,
            w.frac_survivors(),
            w.frac_bound(),
            w.lambda_per_yr,
            w.lambda_sigma_per_yr
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::classify;

    fn pt(id: u32, a: f64, y: f64, flags: u32) -> Point {
        Point {
            id,
            r_km: a,
            a_km: a,
            lon_deg: 0.0,
            megno: y,
            ftle_per_yr: 0.0,
            class: classify(y, flags),
        }
    }

    fn row(id: u32, flags: u32) -> ChaosRow {
        ChaosRow {
            id,
            a0_km: 0.0,
            lambda_rel_prometheus_deg: 0.0,
            ftle_per_day: 0.0,
            log_growth: 0.0,
            megno_mean: 9.0,
            flags,
        }
    }

    #[test]
    fn flags_make_the_lost_list() {
        let l = lost_from_flags(&[row(0, 0), row(1, 2), row(2, 1), row(3, 3)]);
        assert_eq!(l.iter().map(|x| x.id).collect::<Vec<_>>(), vec![1, 3]);
        assert!(l.iter().all(|x| x.reason == 1 && x.drop_step == u64::MAX));
    }

    fn drop_row(id: u32, reason: u8) -> DropRow {
        DropRow {
            id,
            reason,
            drop_step: 7,
            time_s: 0.0,
            position_km: [0.0; 3],
            velocity_km_s: [0.0; 3],
        }
    }

    #[test]
    fn drops_give_reasons_and_the_cross_check_compares_ids() {
        let d = lost_from_drops(&[drop_row(5, 2), drop_row(1, 3)]);
        assert_eq!((d[0].reason, d[1].reason, d[0].drop_step), (2, 3, 7));
        let f = lost_from_flags(&[row(1, 2), row(5, 3)]);
        assert!(check_same_ids(&d, &f).is_ok());
        let f2 = lost_from_flags(&[row(1, 2)]);
        assert!(check_same_ids(&d, &f2).is_err());
    }

    #[test]
    fn lost_particles_are_counted_but_not_classified() {
        // Window 100 to 200: ids 0..=5. 0,1 chaotic; 2 regular; 3 lost
        // (Prometheus, MEGNO 99 must not enter lambda); 4 lost (unbound);
        // 5 skipped interval. Id 6 is outside the window.
        let pts = vec![
            pt(0, 110.0, 10.0, 0),
            pt(1, 120.0, 12.0, 0),
            pt(2, 130.0, 2.0, 0),
            pt(3, 140.0, 99.0, 2),
            pt(4, 150.0, 99.0, 2),
            pt(5, 160.0, 7.0, 1),
            pt(6, 250.0, 7.0, 0),
        ];
        let lost = vec![
            Lost {
                id: 3,
                reason: 2,
                drop_step: 5,
            },
            Lost {
                id: 4,
                reason: 1,
                drop_step: 9,
            },
        ];
        let t = window_table(&pts, &lost, &[(100.0, 200.0)], 10.0);
        let w = &t[0];
        assert_eq!((w.n, w.live, w.chaotic), (6, 3, 2));
        assert_eq!(
            (w.unbound, w.prometheus, w.pandora, w.invalid),
            (1, 1, 0, 1)
        );
        // median of 10, 12, 2 is 10: lambda = 2 * 10 / 10.
        assert!((w.lambda_per_yr - 2.0).abs() < 1e-12);
        assert!((w.frac_survivors() - 2.0 / 3.0).abs() < 1e-12);
        assert!((w.frac_bound() - 4.0 / 5.0).abs() < 1e-12);
    }

    #[test]
    fn csv_has_one_row_per_window() {
        let t = window_table(&[pt(0, 1.0, 9.0, 0)], &[], &[(0.0, 2.0), (2.0, 3.0)], 1.0);
        let mut b = Vec::new();
        write_window_csv(&mut b, &t).unwrap();
        let s = String::from_utf8(b).unwrap();
        assert_eq!(s.lines().count(), 3);
        assert!(s.starts_with(WINDOW_HEADER));
    }
}
