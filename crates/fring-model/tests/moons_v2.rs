//! Task C3 against V2: the Rust moon integrator against the REBOUND reference
//! `validation/reference/v2_moons_6h.npy` (ADR 0004 model, IAS15 epsilon 1e-9,
//! 20 Julian years, a row every 6 h).
//!
//! The moons are mildly chaotic (`validation/reports/v2.md`), so a position
//! comparison is only meaningful up to the first epoch where the two runs
//! differ by 1 km. After that epoch, statistics are compared.
//!
//! The 1-year comparison is fast. The 20-year comparison is `#[ignore]`:
//! `cargo test -p fring-model --release --test moons_v2 -- --ignored --nocapture`.

#![allow(clippy::needless_range_loop)]

use fring_model::params::{moons_from_toml_str, saturn_from_toml_str};
use fring_model::physics::{build_ephemeris, interpolate_moons};
use fring_model::{MoonEphemeris, Vec3};

const SATURN_TOML: &str = include_str!("../../../params/saturn.toml");
const MOONS_TOML: &str = include_str!("../../../params/moons.toml");
const NPY_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../validation/reference/v2_moons_6h.npy"
);
const ROW_S: f64 = 21_600.0;
const YEAR_S: f64 = 365.25 * 86_400.0;

/// Reads a little-endian float64 C-order `.npy` file (format 1.0 or 2.0).
fn read_npy(path: &str) -> (usize, usize, Vec<f64>) {
    let b = std::fs::read(path).expect("reference file");
    assert_eq!(&b[..6], b"\x93NUMPY");
    let (hlen, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[start..start + hlen]).unwrap();
    assert!(header.contains("'<f8'") && header.contains("'fortran_order': False"));
    let shape = header.split("'shape': (").nth(1).unwrap();
    let shape = shape.split(')').next().unwrap();
    let dims: Vec<usize> = shape
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (rows, cols) = (dims[0], dims[1]);
    let (chunks, rest) = b[start + hlen..].as_chunks::<8>();
    assert!(rest.is_empty());
    let data: Vec<f64> = chunks.iter().map(|c| f64::from_le_bytes(*c)).collect();
    assert_eq!(data.len(), rows * cols);
    (rows, cols, data)
}

fn build(duration_s: f64) -> MoonEphemeris {
    let saturn = saturn_from_toml_str(SATURN_TOML).unwrap();
    let moons = moons_from_toml_str(MOONS_TOML).unwrap();
    build_ephemeris(&saturn, &moons, duration_s, 1800.0).unwrap()
}

/// Position differences (km) between the table and the reference at each row
/// `0..rows`, for Prometheus and Pandora.
fn differences(eph: &MoonEphemeris, data: &[f64], rows: usize) -> Vec<[f64; 2]> {
    (0..rows)
        .map(|r| {
            let row = &data[r * 13..(r + 1) * 13];
            let m = interpolate_moons(eph, row[0]).unwrap();
            let mut d = [0.0; 2];
            for k in 0..2 {
                let o = 1 + 6 * k;
                let p = Vec3::new(row[o], row[o + 1], row[o + 2]);
                d[k] = (m.moons[k].position_km - p).norm();
            }
            d
        })
        .collect()
}

#[test]
fn one_year_agrees_with_rebound() {
    let (rows, _, data) = read_npy(NPY_PATH);
    let n = 1461; // 365 days of 6 h rows
    assert!(rows > n);
    let eph = build(n as f64 * ROW_S);
    eph.validate().unwrap();
    let d = differences(&eph, &data, n);
    let max = d.iter().flatten().cloned().fold(0.0, f64::max);
    println!("max |dr| over the first year: {max:e} km");
    assert!(max < 1.0e-4, "{max:e} km");
}

/// Unwrapped azimuth (rad) of a series of (x, y).
fn unwrap(angles: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut out = Vec::new();
    let mut prev = None;
    let mut offset = 0.0;
    for a in angles {
        if let Some(p) = prev {
            let mut d: f64 = a - p;
            while d > std::f64::consts::PI {
                d -= 2.0 * std::f64::consts::PI;
                offset -= 2.0 * std::f64::consts::PI;
            }
            while d < -std::f64::consts::PI {
                d += 2.0 * std::f64::consts::PI;
                offset += 2.0 * std::f64::consts::PI;
            }
        }
        prev = Some(a);
        out.push(a + offset);
    }
    out
}

/// Slope and intercept of the least-squares line through (t, y).
fn fit(t: &[f64], y: &[f64]) -> (f64, f64) {
    let n = t.len() as f64;
    let (mt, my) = (t.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = t.iter().zip(y).map(|(a, b)| (a - mt) * (b - my)).sum();
    let sxx: f64 = t.iter().map(|a| (a - mt) * (a - mt)).sum();
    (sxy / sxx, my - sxy / sxx * mt)
}

/// A series of states (t_s, [x,y,z,vx,vy,vz] for each of two moons).
type Series = Vec<(f64, [[f64; 6]; 2])>;

fn series_from_rows(data: &[f64], rows: usize) -> Series {
    (0..rows)
        .map(|r| {
            let row = &data[r * 13..(r + 1) * 13];
            let mut s = [[0.0; 6]; 2];
            for k in 0..2 {
                s[k].copy_from_slice(&row[1 + 6 * k..7 + 6 * k]);
            }
            (row[0], s)
        })
        .collect()
}

fn series_from_table(eph: &MoonEphemeris, rows: usize) -> Series {
    (0..rows)
        .map(|r| {
            let t = r as f64 * ROW_S;
            let m = interpolate_moons(eph, t).unwrap();
            let mut s = [[0.0; 6]; 2];
            for k in 0..2 {
                let (p, v) = (m.moons[k].position_km, m.moons[k].velocity_km_s);
                s[k] = [p.x, p.y, p.z, v.x, v.y, v.z];
            }
            (t, s)
        })
        .collect()
}

struct Statistics {
    mean_motion_deg_day: [f64; 2],
    synodic_days: f64,
    conjunctions: usize,
    antialignment_yr: f64,
}

/// Statistics that do not depend on the exact phase of a chaotic orbit.
///
/// Mean motion: slope of the unwrapped azimuth. Synodic period: time per
/// conjunction (the azimuth difference crossing multiples of 2 pi). Apse
/// anti-alignment: 360 degrees divided by the growth rate of the difference of
/// the longitudes of pericentre, from the eccentricity vectors (two-body, GM of
/// Saturn) averaged over 4 days (16 rows).
fn statistics(s: &Series, gm: f64) -> Statistics {
    let t: Vec<f64> = s.iter().map(|r| r.0).collect();
    let lam: Vec<Vec<f64>> = (0..2)
        .map(|k| unwrap(s.iter().map(|r| r.1[k][1].atan2(r.1[k][0]))))
        .collect();
    let mut mm = [0.0; 2];
    for k in 0..2 {
        mm[k] = fit(&t, &lam[k]).0.to_degrees() * 86_400.0;
    }
    let diff: Vec<f64> = (0..s.len()).map(|i| lam[1][i] - lam[0][i]).collect();
    let tau = 2.0 * std::f64::consts::PI;
    let conj = ((diff[diff.len() - 1] - diff[0]).abs() / tau).floor() as usize;
    let synodic =
        (t[t.len() - 1] - t[0]) / 86_400.0 / ((diff[diff.len() - 1] - diff[0]).abs() / tau);
    // Averaged eccentricity vectors.
    let window = 16;
    let evec = |k: usize, i: usize| {
        let p = Vec3::new(s[i].1[k][0], s[i].1[k][1], s[i].1[k][2]);
        let v = Vec3::new(s[i].1[k][3], s[i].1[k][4], s[i].1[k][5]);
        let r = p.norm();
        p * (v.norm_squared() / gm - 1.0 / r) - v * (p.dot(v) / gm)
    };
    let mut tt = Vec::new();
    let mut dvarpi = Vec::new();
    let mut i = 0;
    while i + window <= s.len() {
        let mut e = [Vec3::zero(), Vec3::zero()];
        for k in 0..2 {
            for j in i..i + window {
                e[k] += evec(k, j);
            }
        }
        let varpi = [e[0].y.atan2(e[0].x), e[1].y.atan2(e[1].x)];
        tt.push(t[i + window / 2]);
        dvarpi.push(varpi[1] - varpi[0]);
        i += window;
    }
    let dvarpi = unwrap(dvarpi.into_iter());
    let slope = fit(&tt, &dvarpi).0; // rad/s
    Statistics {
        mean_motion_deg_day: mm,
        synodic_days: synodic,
        conjunctions: conj,
        antialignment_yr: tau / slope.abs() / YEAR_S,
    }
}

#[test]
#[ignore = "20 years: about 1 s in release, a minute in debug"]
fn twenty_years_against_rebound() {
    let (rows, _, data) = read_npy(NPY_PATH);
    assert_eq!(rows, 29_221);
    let t0 = std::time::Instant::now();
    let eph = build((rows - 1) as f64 * ROW_S);
    println!("build 20 yr: {:.2} s", t0.elapsed().as_secs_f64());
    eph.validate().unwrap();
    let d = differences(&eph, &data, rows);
    let max = [0, 1].map(|k| d.iter().map(|x| x[k]).fold(0.0, f64::max));
    println!(
        "max |dr| over 20 yr: Prometheus {:e} km, Pandora {:e} km",
        max[0], max[1]
    );
    for (yr, label) in [
        (1.0, "1"),
        (5.0, "5"),
        (10.0, "10"),
        (15.0, "15"),
        (20.0, "20"),
    ] {
        let i = ((yr * YEAR_S / ROW_S) as usize).min(rows - 1);
        println!("  |dr| at {label} yr: {:e} / {:e} km", d[i][0], d[i][1]);
    }
    let first = d.iter().position(|x| x[0] >= 1.0 || x[1] >= 1.0);
    match first {
        Some(i) => println!(
            "first epoch with |dr| >= 1 km: {:.3} yr",
            i as f64 * ROW_S / YEAR_S
        ),
        None => println!("1 km epoch: not reached in 20 yr"),
    }
    let saturn = saturn_from_toml_str(SATURN_TOML).unwrap();
    let rust = statistics(&series_from_table(&eph, rows), saturn.gm_km3_s2);
    let reference = statistics(&series_from_rows(&data, rows), saturn.gm_km3_s2);
    println!(
        "mean motion deg/day Rust {:?} REBOUND {:?}",
        rust.mean_motion_deg_day, reference.mean_motion_deg_day
    );
    println!(
        "synodic days Rust {:.4} REBOUND {:.4} ({} / {} conjunctions)",
        rust.synodic_days, reference.synodic_days, rust.conjunctions, reference.conjunctions
    );
    println!(
        "anti-alignment yr Rust {:.4} REBOUND {:.4}",
        rust.antialignment_yr, reference.antialignment_yr
    );
    // Statistics within 1 %.
    for k in 0..2 {
        let rel = (rust.mean_motion_deg_day[k] / reference.mean_motion_deg_day[k] - 1.0).abs();
        assert!(rel < 0.01, "mean motion {k}: {rel:e}");
    }
    assert!((rust.synodic_days / reference.synodic_days - 1.0).abs() < 0.01);
    assert!((rust.antialignment_yr / reference.antialignment_yr - 1.0).abs() < 0.01);
    // Positions: agreement below 1 km until the 1 km epoch is, by definition,
    // true; the real test is that the epoch is late. Require at least 5 yr.
    if let Some(i) = first {
        assert!(i as f64 * ROW_S / YEAR_S > 5.0);
    }
}
