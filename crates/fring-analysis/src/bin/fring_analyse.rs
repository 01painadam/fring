//! `fring-analyse`: the A1-ANALYSIS tools. Each subcommand reads run
//! directories and writes small CSV files; `python/` makes the figures.
//!
//! ```text
//! fring-analyse map <run> <out_dir> <tag> [r_lo r_hi r_bin lon_bin]
//! fring-analyse windows <run> <out.csv> <years>
//!   per radius window: survivors, lost by reason (flag bit 1), chaotic fraction
//!   of the survivors and with the lost counted as chaotic, lambda (A3-ANALYSIS)
//! fring-analyse compare <run_a> <run_b>
//! fring-analyse closest <config.toml> <out.csv> [moons r_lo r_hi n_max chaos_run]
//! fring-analyse apse <out_dir> <run>...
//! fring-analyse moons <run>
//! fring-analyse laplace <run_with_both_moons>
//! fring-analyse lyapunov <out_dir> <n_pairs> <years> [threads d0_km renorm_years kick_km seed tag method(tangent|shadow) moons_file L]
//!   moons_file (index 10): a file with the layout of params/moons.toml, e.g.
//!     params/variants/gr03b-moons.toml; "" or absent = params/moons.toml. A given
//!     file also prints the two GM values.
//!   L (index 11): also write lyapunov_L<tag>.csv (L(t) of every run),
//!     lyapunov_Lmean<tag>.csv, lyapunov_gr<tag>.csv and print the GR03 rates.
//!     With L, fewer than 5 runs are allowed (one long trajectory).
//! ```

use fring_analysis::apse::{self, DAY_S};
use fring_analysis::laplace::{radius_from_mean_motion, secular_apse_rate_rad_s};
use fring_analysis::lyapunov::{self, LyapunovConfig};
use fring_analysis::map::{self, Class};
use fring_analysis::stats::{self, mean, ols, quantile, std};
use fring_analysis::{closest, setup};
use fring_io::RunDir;
use fring_model::physics::{SaturnField, frequencies};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("map") => cmd_map(&args[1..]),
        Some("windows") => cmd_windows(&args[1..]),
        Some("compare") => cmd_compare(&args[1..]),
        Some("closest") => cmd_closest(&args[1..]),
        Some("apse") => cmd_apse(&args[1..]),
        Some("moons") => cmd_moons(&args[1..]),
        Some("laplace") => cmd_laplace(&args[1..]),
        Some("lyapunov") => cmd_lyapunov(&args[1..]),
        Some("closest-join") => cmd_closest_join(&args[1..]),
        Some("ecc-compare") => cmd_ecc_compare(&args[1..]),
        Some("edge-compare") => cmd_edge_compare(&args[1..]),
        Some("ecc-configs") => cmd_ecc_configs(&args[1..]),
        Some("pair-compare") => cmd_pair_compare(&args[1..]),
        Some("closest-bins") => cmd_closest_bins(&args[1..]),
        _ => Err("usage: fring-analyse map|compare|closest|apse|moons|laplace|lyapunov ...".into()),
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn num(args: &[String], i: usize, default: f64) -> f64 {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn create(dir: &str, name: &str) -> Res<BufWriter<File>> {
    std::fs::create_dir_all(dir)?;
    Ok(BufWriter::new(File::create(Path::new(dir).join(name))?))
}

fn class_counts(pts: &[map::Point]) -> [usize; 5] {
    let mut c = [0usize; 5];
    for p in pts {
        c[p.class.index()] += 1;
    }
    c
}

fn print_fractions(label: &str, pts: &[map::Point]) {
    let c = class_counts(pts);
    let n: usize = c[..4].iter().sum();
    println!(
        "{label}: n={} invalid={} bounded={:.4} regular={:.4} uncertain={:.4} chaotic={:.4}",
        n,
        c[4],
        c[0] as f64 / n as f64,
        c[1] as f64 / n as f64,
        c[2] as f64 / n as f64,
        c[3] as f64 / n as f64
    );
}

fn cmd_map(a: &[String]) -> Res<()> {
    let (run, out, tag) = (&a[0], &a[1], &a[2]);
    let (r_lo, r_hi) = (num(a, 3, 139_700.0), num(a, 4, 141_100.0));
    let (r_bin, lon_bin) = (num(a, 5, 7.0), num(a, 6, 10.0));
    let pts = map::read_points(Path::new(run))?;
    map::write_map_csv(
        &mut create(out, &format!("{tag}_map.csv"))?,
        &pts,
        r_lo,
        r_hi,
        r_bin,
        lon_bin,
    )?;
    map::write_profile_csv(
        &mut create(out, &format!("{tag}_profile.csv"))?,
        &pts,
        r_lo,
        r_hi,
        num(a, 7, r_bin),
    )?;
    map::write_megno_hist(
        &mut create(out, &format!("{tag}_hist.csv"))?,
        &pts,
        -2.0,
        20.0,
        0.25,
    )?;
    print_fractions(tag, &pts);
    Ok(())
}

fn cmd_windows(a: &[String]) -> Res<()> {
    let years: f64 = a
        .get(2)
        .and_then(|s| s.parse().ok())
        .ok_or("usage: windows <run> <out.csv> <years>")?;
    let pts = map::read_points(Path::new(&a[0]))?;
    let lost = fring_analysis::loss::read_lost(Path::new(&a[0]))?;
    let t = fring_analysis::loss::window_table(
        &pts,
        &lost,
        &fring_analysis::loss::default_windows(),
        years,
    );
    let mut f = BufWriter::new(File::create(&a[1])?);
    fring_analysis::loss::write_window_csv(&mut f, &t)?;
    println!("windows: {} particles, {} lost", pts.len(), lost.len());
    Ok(())
}

fn cmd_compare(a: &[String]) -> Res<()> {
    let p = map::read_points(Path::new(&a[0]))?;
    let q = map::read_points(Path::new(&a[1]))?;
    let m = map::cross_tab(&p, &q);
    let names = ["bounded", "regular", "uncertain", "chaotic", "invalid"];
    println!("rows: {} ; columns: {}", a[0], a[1]);
    println!("class,{}", names.join(","));
    for (i, row) in m.iter().enumerate() {
        println!(
            "{},{}",
            names[i],
            row.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    Ok(())
}

/// `closest <config> <out.csv> [moons=prometheus] [r_lo r_hi n_max] [chaos_run]`
/// (`-` keeps a default). `moons` is a comma list. With `r_lo r_hi n_max` only
/// the first `n_max` particles with an initial radius in the window are
/// followed. Writes `<out>.csv` (one `min_<moon>_km` column per moon) and, per
/// moon, `<out>_<moon>_cum.csv` and `<out>_<moon>_hist.csv`.
fn cmd_closest(a: &[String]) -> Res<()> {
    let s = setup::load(Path::new(&a[0]))?;
    let moons: Vec<&str> = a
        .get(2)
        .filter(|m| m.as_str() != "-")
        .map_or("prometheus", String::as_str)
        .split(',')
        .collect();
    let init = if a.len() > 5 && a[3] != "-" {
        closest::select_window(
            &s.init,
            num(a, 3, 0.0),
            num(a, 4, f64::MAX),
            num(a, 5, 1e9) as usize,
        )
    } else {
        s.init.clone()
    };
    eprintln!(
        "closest: {} particles, moons {moons:?}",
        init.particles.len()
    );
    let res = closest::run(&s, &init, &moons, |l| eprintln!("{l}"))?;
    let moons_toml =
        std::fs::read_to_string(Path::new(&s.config.run.params_dir).join("moons.toml"))?;
    let classes = if let Some(dir) = a.get(6).filter(|d| d.as_str() != "-") {
        let pts = map::read_points(Path::new(dir))?;
        Some(
            pts.iter()
                .map(|p| (p.id, *p))
                .collect::<std::collections::HashMap<_, _>>(),
        )
    } else {
        None
    };
    let n = res.id.len() as f64;
    let mut w = BufWriter::new(File::create(&a[1])?);
    write!(w, "id,r0_km")?;
    for m in &moons {
        write!(w, ",min_{m}_km,min_half_{m}_km")?;
    }
    writeln!(w, ",class")?;
    let mut matched = 0usize;
    for i in 0..res.id.len() {
        write!(w, "{},{:.4}", res.id[i], res.r0_km[i])?;
        for q in 0..moons.len() {
            write!(w, ",{:.3},{:.3}", res.min_km[q][i], res.min_half_km[q][i])?;
        }
        let cl = classes
            .as_ref()
            .and_then(|c| c.get(&res.id[i]))
            .filter(|p| (p.r_km - res.r0_km[i]).abs() < 1e-6)
            .map_or("", |p| {
                matched += 1;
                p.class.name()
            });
        writeln!(w, ",{cl}")?;
    }
    w.flush()?;
    let stem = Path::new(&a[1]).with_extension("");
    for (q, m) in moons.iter().enumerate() {
        let track = s
            .system
            .moons
            .tracks
            .iter()
            .find(|t| t.name == *m)
            .ok_or("moon not found")?;
        let rm = mean(
            &track
                .samples
                .iter()
                .map(|b| b.position_km.norm())
                .collect::<Vec<_>>(),
        );
        let hill = rm * (track.gm_km3_s2 / (3.0 * s.saturn.gm_km3_s2)).cbrt();
        let rad = closest::moon_radius_km(&moons_toml, m).ok_or("no radius in params")?;
        println!("{m}: mean radius {rm:.1} km, Hill radius {hill:.2} km, body radius {rad} km");
        let min = &res.min_km[q];
        let mut tw = BufWriter::new(File::create(format!("{}_{m}_cum.csv", stem.display()))?);
        writeln!(
            tw,
            "hill_radii,km,n_below,frac_below,n_bounded,n_regular,n_uncertain,n_chaotic,n_unmatched"
        )?;
        let mut ks = vec![rad / hill];
        ks.extend([
            0.5, 1.0, 2.0, 3.0, 4.1, 5.0, 6.3, 7.3, 10.0, 20.0, 50.0, 100.0,
        ]);
        for k in ks {
            let mut by = [0usize; 5];
            let mut tot = 0usize;
            for (i, d) in min.iter().enumerate() {
                if *d < k * hill {
                    tot += 1;
                    let cl = classes
                        .as_ref()
                        .and_then(|c| c.get(&res.id[i]))
                        .filter(|p| (p.r_km - res.r0_km[i]).abs() < 1e-6)
                        .map_or(4, |p| p.class.index());
                    by[cl] += 1;
                }
            }
            writeln!(
                tw,
                "{k:.4},{:.1},{tot},{:.5},{},{},{},{},{}",
                k * hill,
                tot as f64 / n,
                by[0],
                by[1],
                by[2],
                by[3],
                by[4]
            )?;
        }
        tw.flush()?;
        let mut hw = BufWriter::new(File::create(format!("{}_{m}_hist.csv", stem.display()))?);
        writeln!(hw, "hill_lo,hill_hi,count")?;
        let edges: Vec<f64> = (0..=40)
            .map(|k| 0.5 * 10f64.powf(k as f64 * 0.075))
            .collect();
        for w2 in edges.windows(2) {
            let c = min
                .iter()
                .filter(|d| **d / hill >= w2[0] && **d / hill < w2[1])
                .count();
            writeln!(hw, "{:.4},{:.4},{c}", w2[0], w2[1])?;
        }
        hw.flush()?;
        println!(
            "{m}: hits (< {rad} km) {}; min distance quantiles 0/1/5/50 %: {:.1} {:.1} {:.1} {:.1} km; classes matched {matched}",
            min.iter().filter(|d| **d < rad).count(),
            quantile(min, 0.0),
            quantile(min, 0.01),
            quantile(min, 0.05),
            quantile(min, 0.5)
        );
    }
    Ok(())
}

/// `closest-bins <closest.csv> <out.csv> [bin_km hill_km min_column body_km]`: closest approach against the initial radius.
/// `ecc-configs <out_dir> <years> <tag> [count=5000] [step_div=1] [bins=all] [set=std]`:
/// writes the configs of the eccentric map (22 bins of 25 km from 140,150 to
/// 140,700 km). Set `std`: each bin as an `ecc` file (e and apse from
/// `params/ring.toml`) and a `circ` control (e = 0) with the same seed; for 10
/// years also the apse + 120 deg variants and the seed repeats of three bins
/// (inner 0, core 2, outer 21). Set `ecc` or `phase120` (A5-ANALYSIS): only the
/// ecc run, or only the apse + 120 deg run, of the bins in the comma list
/// `bins` (`all` is 0 to 21). `step_div` divides the step and multiplies the
/// renorm interval. `<out_dir>/list_<tag>.txt` names the runs, one per line.
fn cmd_ecc_configs(a: &[String]) -> Res<()> {
    let (dir, years, tag) = (&a[0], num(a, 1, 10.0), &a[2]);
    let count = num(a, 3, 5000.0) as u32;
    let step_div = num(a, 4, 1.0) as u32;
    let bins: Vec<usize> = match a.get(5).map(String::as_str) {
        None | Some("all") => (0..22).collect(),
        Some(l) => l.split(',').map(str::parse).collect::<Result<_, _>>()?,
    };
    let set = a.get(6).map_or("std", String::as_str);
    let ring = std::fs::read_to_string("params/ring.toml")?;
    let e = setup::toml_number(&ring, "eccentricity").ok_or("no eccentricity")?;
    let w = setup::toml_number(&ring, "apse_longitude_deg").ok_or("no apse")?;
    std::fs::create_dir_all(dir)?;
    let jobs = setup::ecc_jobs(years, tag, step_div.max(1), &bins, set, e, w)?;
    for j in &jobs {
        let lo = 140_150.0 + 25.0 * j.bin as f64;
        let text = setup::ecc_config(
            &j.name,
            j.seed,
            j.step_s,
            j.renorm_steps,
            years,
            (lo, lo + 25.0),
            count,
            j.ecc,
            j.apse_deg,
        );
        std::fs::write(Path::new(dir).join(format!("{}.toml", j.name)), text)?;
    }
    let names: Vec<&str> = jobs.iter().map(|j| j.name.as_str()).collect();
    std::fs::write(
        Path::new(dir).join(format!("list_{tag}.txt")),
        names.join("\n") + "\n",
    )?;
    println!("{} configs in {dir}", jobs.len());
    Ok(())
}

/// `pair-compare <years> <out.csv> <label>=<A1,A2,..>:<B1,B2,..> ...`: for each
/// group, joins the points of the run directories of A and of B and writes one
/// row of the comparison (the columns of `edge_step.csv`, A first). Used by
/// A5-ANALYSIS for the step test (A = h/4, B = h) and the apse phase (A = +120
/// deg, B = base).
fn cmd_pair_compare(a: &[String]) -> Res<()> {
    use fring_analysis::compare::header;
    let years = num(a, 0, 10.0);
    let mut w = BufWriter::new(File::create(&a[1])?);
    writeln!(w, "{}", header("window,a_lo_km,a_hi_km,"))?;
    let read = |list: &str| -> Res<Vec<map::Point>> {
        let mut v = Vec::new();
        for d in list.split(',') {
            v.extend(map::read_points(Path::new(d))?);
        }
        Ok(v)
    };
    for g in &a[2..] {
        let (label, rest) = g.split_once('=').ok_or("group needs label=A:B")?;
        let (ra, rb) = rest.split_once(':').ok_or("group needs label=A:B")?;
        let (pa, pb) = (read(ra)?, read(rb)?);
        let lo = pa
            .iter()
            .chain(&pb)
            .map(|p| p.a_km)
            .fold(f64::MAX, f64::min);
        let hi = pa
            .iter()
            .chain(&pb)
            .map(|p| p.a_km)
            .fold(f64::MIN, f64::max);
        let r =
            map_compare(&pa, &pb, years, 13).ok_or(format!("group {label}: too few particles"))?;
        writeln!(w, "{label},{lo:.0},{hi:.0},{}", r.csv())?;
    }
    w.flush()?;
    Ok(())
}

fn window(p: &[map::Point], lo: f64, hi: f64) -> Vec<map::Point> {
    p.iter()
        .filter(|q| q.a_km >= lo && q.a_km < hi)
        .copied()
        .collect()
}

/// `ecc-compare <runs_dir> <tag> <years> <out_dir>`: per bin the eccentric run
/// `<tag>-ecc-bNN` against its e = 0 control `<tag>-circ-bNN` (same particles
/// by id), with Holm-adjusted p over the bins that completed; pooled windows;
/// and the seed repeat and apse + 120 deg runs against the base run. A bin
/// whose run directory has no `chaos.csv` is written with status `missing`.
fn cmd_ecc_compare(a: &[String]) -> Res<()> {
    use fring_analysis::compare::{MIN_COUNT, header};
    let (runs, tag, years, out) = (&a[0], &a[1], num(a, 2, 10.0), &a[3]);
    let read = |name: &str| map::read_points(&Path::new(runs).join(name)).ok();
    let mut w = create(out, &format!("ecc_bins_{tag}.csv"))?;
    writeln!(w, "{},holm_p,status", header("a_lo_km,a_hi_km,"))?;
    let mut rows = Vec::new();
    let (mut all_e, mut all_c) = (Vec::new(), Vec::new());
    for bin in 0..22usize {
        let (lo, hi) = (140_150.0 + 25.0 * bin as f64, 140_175.0 + 25.0 * bin as f64);
        let e = read(&format!("{tag}-ecc-b{bin:02}"));
        let c = read(&format!("{tag}-circ-b{bin:02}"));
        let row = match (&e, &c) {
            (Some(e), Some(c)) => map_compare(e, c, years, 100 + bin as u64),
            _ => None,
        };
        if let (Some(e), Some(c)) = (e, c) {
            all_e.extend(e);
            all_c.extend(c);
        }
        rows.push((lo, hi, row));
    }
    let ps: Vec<f64> = rows
        .iter()
        .filter_map(|r| r.2.as_ref().map(|x| x.dfrac.3))
        .collect();
    let adj = stats::holm(&ps);
    let mut k = 0;
    for (lo, hi, row) in &rows {
        match row {
            Some(r) => {
                writeln!(w, "{lo:.0},{hi:.0},{},{:.3e},ok", r.csv(), adj[k])?;
                k += 1;
            }
            None => writeln!(w, "{lo:.0},{hi:.0},{},,missing", ",".repeat(17))?,
        }
    }
    w.flush()?;
    let mut ww = create(out, &format!("ecc_windows_{tag}.csv"))?;
    writeln!(ww, "{}", header("window,a_lo_km,a_hi_km,"))?;
    for (name, lo, hi) in [
        ("core", 140_197.0, 140_247.0),
        ("inner", 140_150.0, 140_425.0),
        ("outer", 140_425.0, 140_700.0),
        ("quiet_zone", 140_150.0, 140_700.0),
    ] {
        match map_compare(&window(&all_e, lo, hi), &window(&all_c, lo, hi), years, 7) {
            Some(r) => writeln!(ww, "{name},{lo:.0},{hi:.0},{}", r.csv())?,
            None => writeln!(ww, "{name},{lo:.0},{hi:.0},{},", ",".repeat(17))?,
        }
    }
    ww.flush()?;
    let mut rw = create(out, &format!("ecc_repeats_{tag}.csv"))?;
    writeln!(rw, "{}", header("kind,bin,"))?;
    for bin in [0usize, 2, 21] {
        let base = read(&format!("{tag}-ecc-b{bin:02}"));
        for kind in ["repeat", "phase120"] {
            let other = read(&format!("{tag}-ecc-{kind}-b{bin:02}"));
            let row = match (&other, &base) {
                (Some(o), Some(b)) => fring_analysis::compare::compare(o, b, years, MIN_COUNT, 9),
                _ => None,
            };
            match row {
                Some(r) => writeln!(rw, "{kind},{bin},{}", r.csv())?,
                None => writeln!(rw, "{kind},{bin},{},", ",".repeat(17))?,
            }
        }
    }
    rw.flush()?;
    println!("wrote ecc_bins/windows/repeats_{tag}.csv in {out}");
    Ok(())
}

/// Compares two sets of the same ids in order: ecc against its control.
fn map_compare(
    e: &[map::Point],
    c: &[map::Point],
    years: f64,
    seed: u64,
) -> Option<fring_analysis::compare::Row> {
    fring_analysis::compare::compare(e, c, years, fring_analysis::compare::MIN_COUNT, seed)
}

/// `edge-compare <run_h> <run_h4> <years> <out.csv>`: the outer-edge window at
/// the step h against the step h/4 (same particles). A = h4, B = h.
fn cmd_edge_compare(a: &[String]) -> Res<()> {
    use fring_analysis::compare::header;
    let h = map::read_points(Path::new(&a[0]))?;
    let h4 = map::read_points(Path::new(&a[1]))?;
    let years = num(a, 2, 10.0);
    let mut w = BufWriter::new(File::create(&a[3])?);
    writeln!(w, "{}", header("window,a_lo_km,a_hi_km,"))?;
    for (name, lo, hi) in [
        ("whole", 140_850.0, 141_100.0),
        ("w1", 140_850.0, 140_900.0),
        ("w2", 140_900.0, 141_000.0),
        ("w3", 141_000.0, 141_100.0),
    ] {
        let r = map_compare(&window(&h4, lo, hi), &window(&h, lo, hi), years, 11)
            .ok_or("too few particles in a window")?;
        writeln!(w, "{name},{lo:.0},{hi:.0},{}", r.csv())?;
    }
    w.flush()?;
    Ok(())
}

/// `closest-join <closest.csv> <chaos_run> <moon> <hill_km> <body_km> <out_dir>`:
/// joins the classes of a chaos run to an existing `closest` CSV by particle id
/// (the CPU and GPU trajectories differ after a Lyapunov time, so this is a
/// statistical relation, not a trajectory match). Fails if fewer than all rows
/// match or if `|r0|` disagrees. Writes `<out_dir>/closest_<moon>_classes.csv`:
/// for each limit, the particles below it by class.
fn cmd_closest_join(a: &[String]) -> Res<()> {
    let text = std::fs::read_to_string(&a[0])?;
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().ok_or("empty csv")?.split(',').collect();
    let col = head
        .iter()
        .position(|h| *h == format!("min_{}_km", a[2]))
        .ok_or("no such moon column")?;
    let (hill, body): (f64, f64) = (a[3].parse()?, a[4].parse()?);
    let by_id: std::collections::HashMap<u32, map::Point> = map::read_points(Path::new(&a[1]))?
        .into_iter()
        .map(|p| (p.id, p))
        .collect();
    let rows: Vec<(u32, f64, f64)> = lines
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            Ok((f[0].parse()?, f[1].parse()?, f[col].parse()?))
        })
        .collect::<Res<_>>()?;
    let mut matched = 0usize;
    let limits = [
        ("hit".to_string(), body),
        ("1_hill".to_string(), hill),
        ("4.1_hill".to_string(), 4.1 * hill),
        ("6.3_hill".to_string(), 6.3 * hill),
        ("all".to_string(), f64::INFINITY),
    ];
    let mut counts = vec![[0usize; 5]; limits.len()];
    for (id, r0, d) in &rows {
        let p = by_id
            .get(id)
            .ok_or(format!("id {id} is not in the chaos run"))?;
        if (p.r_km - r0).abs() > 1e-3 {
            return Err(format!("id {id}: r0 {r0} differs from the chaos run {}", p.r_km).into());
        }
        matched += 1;
        for (k, (_, lim)) in limits.iter().enumerate() {
            if d < lim {
                counts[k][p.class.index()] += 1;
            }
        }
    }
    assert_eq!(
        matched,
        rows.len(),
        "classes matched {matched} of {}",
        rows.len()
    );
    if matched == 0 {
        return Err("no particle matched".into());
    }
    let mut w = create(&a[5], &format!("closest_{}_classes.csv", a[2]))?;
    writeln!(
        w,
        "limit,km,n_bounded,n_regular,n_uncertain,n_chaotic,n_invalid,n_below"
    )?;
    for (k, (name, lim)) in limits.iter().enumerate() {
        let c = counts[k];
        writeln!(
            w,
            "{name},{lim:.2},{},{},{},{},{},{}",
            c[0],
            c[1],
            c[2],
            c[3],
            c[4],
            c.iter().sum::<usize>()
        )?;
    }
    w.flush()?;
    println!("classes matched {matched} of {}", rows.len());
    Ok(())
}

fn cmd_closest_bins(a: &[String]) -> Res<()> {
    let text = std::fs::read_to_string(&a[0])?;
    let bin = num(a, 2, 50.0);
    // Hill radius and body radius in km come from the `closest` printout.
    let hill = num(a, 3, 63.4);
    let col = num(a, 4, 2.0) as usize;
    let body = num(a, 5, 43.09);
    let mut rows: Vec<(f64, f64)> = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        rows.push((f[1].parse()?, f[col].parse()?));
    }
    let r0 = rows.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
    let r1 = rows.iter().map(|r| r.0).fold(0.0, f64::max);
    let nb = ((r1 - r0) / bin).ceil() as usize + 1;
    let mut bins: Vec<Vec<f64>> = vec![Vec::new(); nb];
    for (r, d) in &rows {
        bins[((r - r0) / bin) as usize].push(*d);
    }
    let mut w = BufWriter::new(File::create(&a[1])?);
    writeln!(
        w,
        "r_lo_km,r_hi_km,n,frac_hit,frac_below_1_hill,frac_below_3_hill,frac_below_4p1_hill,frac_below_6p3_hill,frac_below_7p3_hill,median_min_km"
    )?;
    for (i, b) in bins.iter().enumerate() {
        if b.is_empty() {
            continue;
        }
        let f = |k: f64| b.iter().filter(|d| **d < k * hill).count() as f64 / b.len() as f64;
        let hit = b.iter().filter(|d| **d < body).count() as f64 / b.len() as f64;
        writeln!(
            w,
            "{:.1},{:.1},{},{hit:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
            r0 + i as f64 * bin,
            r0 + (i + 1) as f64 * bin,
            b.len(),
            f(1.0),
            f(3.0),
            f(4.1),
            f(6.3),
            f(7.3),
            quantile(b, 0.5)
        )?;
    }
    Ok(())
}

fn cmd_apse(a: &[String]) -> Res<()> {
    let out = &a[0];
    let mut w = create(out, "apse_fits.csv")?;
    writeln!(
        w,
        "run,rate_deg_per_day,se_jackknife,se_ols,rms_resid_deg,e_mean,e_free,e_dev_rms,rate_even,rate_odd,w1,w2,w3,w4,w5"
    )?;
    for run in &a[1..] {
        let dir = RunDir::new(run);
        let s = apse::series(&dir, None)?;
        let f = apse::fit(&s);
        let r_even = apse::fit(&apse::series(&dir, Some(0))?).rate_deg_per_day;
        let r_odd = apse::fit(&apse::series(&dir, Some(1))?).rate_deg_per_day;
        let name = Path::new(run)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(run);
        writeln!(
            w,
            "{name},{:.7},{:.7},{:.7},{:.5},{:.7},{:.7},{:.7},{:.7},{:.7},{}",
            f.rate_deg_per_day,
            f.se_jackknife,
            f.se_ols,
            f.rms_resid_deg,
            f.e_mean,
            f.e_free,
            f.e_dev_rms,
            r_even,
            r_odd,
            f.window_rates
                .iter()
                .map(|r| format!("{r:.7}"))
                .collect::<Vec<_>>()
                .join(",")
        )?;
        println!(
            "{name}: rate {:.6} +- {:.6} (jackknife) +- {:.6} (ols) deg/d; even {:.6} odd {:.6}; rms resid {:.4} deg; e mean {:.6}, e free {:.6}, e deviation rms {:.2e}; windows {:?}",
            f.rate_deg_per_day,
            f.se_jackknife,
            f.se_ols,
            r_even,
            r_odd,
            f.rms_resid_deg,
            f.e_mean,
            f.e_free,
            f.e_dev_rms,
            f.window_rates
                .iter()
                .map(|r| (r * 1e6).round() / 1e6)
                .collect::<Vec<_>>()
        );
        // Time series, thinned to about 600 points.
        let mut ws = create(out, &format!("apse_series_{name}.csv"))?;
        writeln!(ws, "t_years,varpi_deg,residual_deg,e")?;
        let (slope, icpt) = ols(&s.t_days, &s.varpi_deg);
        let stride = (s.t_days.len() / 600).max(1);
        for i in (0..s.t_days.len()).step_by(stride) {
            writeln!(
                ws,
                "{:.4},{:.5},{:.5},{:.7}",
                s.t_days[i] / 365.25,
                s.varpi_deg[i],
                s.varpi_deg[i] - (icpt + slope * s.t_days[i]),
                s.e[i]
            )?;
        }
    }
    Ok(())
}

/// Per moon: mean radius, extremes, mean motion from `moons.bin`.
struct MoonGeom {
    name: String,
    r_mean: f64,
    r_min: f64,
    r_max: f64,
    n_rad_s: f64,
}

fn moon_geometry(run: &RunDir) -> Res<Vec<MoonGeom>> {
    // The ephemeris (every 1,800 s, about 30 samples per orbit) is rebuilt from the
    // config of the run; `moons.bin` is sampled once per 6 days, which aliases the orbit.
    let s = setup::load(&run.config_path())?;
    let mut out = Vec::new();
    for t in &s.system.moons.tracks {
        let mut r = Vec::new();
        let mut ang = 0.0f64;
        let mut prev: Option<f64> = None;
        for b in &t.samples {
            let p = b.position_km;
            r.push(p.norm());
            let th = p.y.atan2(p.x);
            if let Some(pv) = prev {
                let mut d = th - pv;
                while d < 0.0 {
                    d += 2.0 * std::f64::consts::PI;
                }
                ang += d;
            }
            prev = Some(th);
        }
        let span = (t.samples.len() - 1) as f64 * s.system.moons.interval_s;
        out.push(MoonGeom {
            name: t.name.clone(),
            r_mean: mean(&r),
            r_min: r.iter().copied().fold(f64::INFINITY, f64::min),
            r_max: r.iter().copied().fold(0.0, f64::max),
            n_rad_s: ang / span,
        });
    }
    Ok(out)
}

fn cmd_moons(a: &[String]) -> Res<()> {
    let run = RunDir::new(&a[0]);
    let cfg = fring_model::RunConfig::from_toml_str(&String::from_utf8(run.read_config_copy()?)?)?;
    let (saturn, all) = setup::load_params(Path::new(&cfg.run.params_dir))?;
    let field = SaturnField::<f64>::from_params(&saturn);
    println!(
        "name, mean r, rmin, rmax, e_eff=(rmax-rmin)/(rmax+rmin), n [deg/d], guiding radius from n, Hill radius"
    );
    let mut csv = match a.get(1) {
        Some(out) => {
            let mut w = create(out, "moon_geometry.csv")?;
            writeln!(
                w,
                "name,r_mean_km,r_min_km,r_max_km,e_eff,n_deg_per_day,a_guiding_km,hill_km"
            )?;
            Some(w)
        }
        None => None,
    };
    for g in moon_geometry(&run)? {
        let gm = all
            .iter()
            .find(|m| m.name == g.name)
            .ok_or("moon")?
            .gm_km3_s2;
        let a_gc = radius_from_mean_motion(&field, g.n_rad_s);
        if let Some(w) = csv.as_mut() {
            writeln!(
                w,
                "{},{:.2},{:.2},{:.2},{:.6},{:.5},{:.2},{:.3}",
                g.name,
                g.r_mean,
                g.r_min,
                g.r_max,
                (g.r_max - g.r_min) / (g.r_max + g.r_min),
                g.n_rad_s.to_degrees() * DAY_S,
                a_gc,
                a_gc * (gm / (3.0 * saturn.gm_km3_s2)).cbrt()
            )?;
        }
        println!(
            "{} {:.1} {:.1} {:.1} e={:.5} n={:.4} a_gc={:.1} hill={:.2}",
            g.name,
            g.r_mean,
            g.r_min,
            g.r_max,
            (g.r_max - g.r_min) / (g.r_max + g.r_min),
            g.n_rad_s.to_degrees() * DAY_S,
            a_gc,
            a_gc * (gm / (3.0 * saturn.gm_km3_s2)).cbrt()
        );
    }
    Ok(())
}

fn cmd_laplace(a: &[String]) -> Res<()> {
    let run = RunDir::new(&a[0]);
    let cfg = fring_model::RunConfig::from_toml_str(&String::from_utf8(run.read_config_copy()?)?)?;
    let (saturn, all) = setup::load_params(Path::new(&cfg.run.params_dir))?;
    let field = SaturnField::<f64>::from_params(&saturn);
    let a_core = 140_222.43;
    let fr = frequencies(&field, a_core);
    let to_deg_day = |r: f64| r.to_degrees() * DAY_S;
    println!(
        "core a={a_core}: n={:.5} deg/d, kappa={:.5} deg/d, n-kappa (Saturn only, J2 to J6) = {:.5} deg/d",
        to_deg_day(fr.n_rad_s),
        to_deg_day(fr.kappa_rad_s),
        to_deg_day(fr.n_rad_s - fr.kappa_rad_s)
    );
    let mut total = 0.0;
    let mut csv = match a.get(1) {
        Some(out) => {
            let mut w = create(out, "laplace_secular.csv")?;
            writeln!(w, "name,a_guiding_km,alpha,mu,rate_deg_per_day")?;
            writeln!(
                w,
                "saturn_only,,,,{:.6}",
                to_deg_day(fr.n_rad_s - fr.kappa_rad_s)
            )?;
            Some(w)
        }
        None => None,
    };
    for g in moon_geometry(&run)? {
        let gm = all
            .iter()
            .find(|m| m.name == g.name)
            .ok_or("moon")?
            .gm_km3_s2;
        let a_gc = radius_from_mean_motion(&field, g.n_rad_s);
        let mu = gm / saturn.gm_km3_s2;
        let rate = secular_apse_rate_rad_s(fr.n_rad_s, mu, a_core, a_gc, 4_000_000);
        total += rate;
        if let Some(w) = csv.as_mut() {
            writeln!(
                w,
                "{},{a_gc:.1},{:.5},{mu:.4e},{:.6}",
                g.name,
                a_core.min(a_gc) / a_core.max(a_gc),
                to_deg_day(rate)
            )?;
        }
        let alpha = a_core.min(a_gc) / a_core.max(a_gc);
        println!(
            "{}: a_gc={a_gc:.1} km, alpha={alpha:.5}, mu={mu:.3e}, secular rate = {:.6} deg/d",
            g.name,
            to_deg_day(rate)
        );
    }
    if let Some(w) = csv.as_mut() {
        writeln!(w, "sum_of_moons,,,,{:.6}", to_deg_day(total))?;
    }
    println!("sum of moons = {:.6} deg/d", to_deg_day(total));
    Ok(())
}

/// Mean, standard deviation and error of the mean of `v`, as text.
fn mean_se(v: &[f64]) -> String {
    if v.len() < 2 {
        return format!(
            "n {} mean {:.4}",
            v.len(),
            v.first().copied().unwrap_or(f64::NAN)
        );
    }
    format!(
        "n {} mean {:.4} std {:.4} se {:.4}",
        v.len(),
        mean(v),
        std(v),
        std(v) / (v.len() as f64).sqrt()
    )
}

/// The GR03 analysis (argument `L`): writes L(t) of every run, the mean L(t),
/// and the per-run rates; prints the rates for all runs and for the chaotic group.
#[allow(clippy::needless_range_loop)] // t and L are indexed together
fn gr_report(out: &str, tag: &str, pairs: &[lyapunov::Pair]) -> Res<()> {
    let t = &pairs[0].t_years;
    let nt = t.len();
    let years = t[nt - 1];
    let ia = t.iter().position(|x| *x >= 30.0).unwrap_or(0);
    let nan = |o: Option<(f64, f64)>| o.map_or(f64::NAN, |x| x.0);
    let mut wl = create(out, &format!("lyapunov_L{tag}.csv"))?;
    writeln!(wl, "pair,t_years,L")?;
    let mut wg = create(out, &format!("lyapunov_gr{tag}.csv"))?;
    writeln!(
        wg,
        "pair,slope_30_end,gr_fit_0_end,gr_fit_1_end,gr_anchor,chaotic"
    )?;
    let (mut slope, mut main, mut sens, mut anc, mut chaos) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (k, p) in pairs.iter().enumerate() {
        for i in 0..nt {
            writeln!(wl, "{k},{:.3},{:.5}", t[i], p.log_growth[i])?;
        }
        let sl = ols(&t[ia..], &p.log_growth[ia..]).0;
        let (m, s1) = (
            nan(lyapunov::gr_fit(t, &p.log_growth, 0.0, years)),
            nan(lyapunov::gr_fit(t, &p.log_growth, 1.0, years)),
        );
        let an = lyapunov::gr_anchor(t, &p.log_growth);
        let c = sl > 0.1;
        writeln!(wg, "{k},{sl:.5},{m:.5},{s1:.5},{an:.5},{}", u8::from(c))?;
        if years > 100.0 {
            println!(
                "  run {k}: GR fit 0 to 100 yr {:.4}, 0 to {years} yr {m:.4}, 1 to {years} yr {s1:.4}",
                nan(lyapunov::gr_fit(t, &p.log_growth, 0.0, 100.0))
            );
        }
        slope.push(sl);
        main.push(m);
        sens.push(s1);
        anc.push(an);
        chaos.push(c);
    }
    let mut wm = create(out, &format!("lyapunov_Lmean{tag}.csv"))?;
    writeln!(wm, "t_years,L_mean_all,L_mean_chaotic")?;
    let nch = chaos.iter().filter(|c| **c).count();
    for i in 0..nt {
        let all: f64 = pairs.iter().map(|p| p.log_growth[i]).sum::<f64>() / pairs.len() as f64;
        let ch: f64 = pairs
            .iter()
            .zip(&chaos)
            .filter(|(_, c)| **c)
            .map(|(p, _)| p.log_growth[i])
            .sum::<f64>()
            / nch as f64;
        writeln!(wm, "{:.3},{all:.5},{ch:.5}", t[i])?;
    }
    println!(
        "GR03 analysis: {} runs, {nch} chaotic (slope over 30 to {years} yr above 0.1/yr), fraction {:.3}",
        pairs.len(),
        nch as f64 / pairs.len() as f64
    );
    let pick = |v: &[f64], want: Option<bool>| -> Vec<f64> {
        v.iter()
            .zip(&chaos)
            .filter(|(_, c)| want.is_none_or(|w| w == **c))
            .map(|(x, _)| *x)
            .collect()
    };
    for (name, v) in [
        ("slope of L over 30 to end", &slope),
        ("GR fit, t > 0 to end (main)", &main),
        ("GR fit, 1 yr to end (sensitivity)", &sens),
        ("GR anchor c = L(T)/T - ln(T)/T", &anc),
    ] {
        println!("  {name}:");
        println!("    all     {}", mean_se(&pick(v, None)));
        println!("    chaotic {}", mean_se(&pick(v, Some(true))));
        println!("    regular {}", mean_se(&pick(v, Some(false))));
    }
    Ok(())
}

fn cmd_lyapunov(a: &[String]) -> Res<()> {
    let out = &a[0];
    let n_pairs = num(a, 1, 100.0) as u32;
    let years = num(a, 2, 100.0);
    let threads = num(a, 3, 8.0) as usize;
    let (saturn, mut all) = setup::load_params(Path::new("params"))?;
    if let Some(f) = a.get(10).filter(|f| !f.is_empty()) {
        all = setup::load_moons_file(Path::new(f))?;
    }
    let moons: Vec<_> = all
        .into_iter()
        .filter(|m| m.name == "prometheus" || m.name == "pandora")
        .collect();
    if let Some(f) = a.get(10).filter(|f| !f.is_empty()) {
        let gm = |n: &str| {
            moons
                .iter()
                .find(|m| m.name == n)
                .map_or(f64::NAN, |m| m.gm_km3_s2)
        };
        println!(
            "moons: {f}; GM prometheus = {} km3/s2, GM pandora = {} km3/s2",
            gm("prometheus"),
            gm("pandora")
        );
    }
    let cfg = LyapunovConfig {
        years,
        renorm_years: num(a, 5, 0.25),
        d0_km: num(a, 4, 1.0e-3),
        kick_km: num(a, 6, 2.0),
        epsilon: 1e-9,
        seed: num(a, 7, 20_261_105.0) as u64,
    };
    let tag = a
        .get(8)
        .filter(|t| !t.is_empty())
        .map_or(String::new(), |t| format!("_{t}"));
    let t0 = std::time::Instant::now();
    let method = if a.get(9).map(String::as_str) == Some("shadow") {
        lyapunov::Method::Shadow
    } else {
        lyapunov::Method::Tangent
    };
    let pairs = lyapunov::run_all(&saturn, &moons, &cfg, n_pairs, threads, method)?;
    println!(
        "{n_pairs} pairs x {years} yr on {threads} threads: {:.1} s; base steps per pair {}",
        t0.elapsed().as_secs_f64(),
        pairs[0].steps
    );
    if a.get(11).map(String::as_str) == Some("L") {
        gr_report(out, &tag, &pairs)?;
    }
    if n_pairs < 5 {
        return Ok(());
    }
    let nt = pairs[0].t_years.len();
    let t = &pairs[0].t_years;
    let lam = |p: &lyapunov::Pair, i: usize| p.log_growth[i] / p.t_years[i];
    // Curves: FTLE(T), slope over [T/2... no: slope of L over [t_a, T] with t_a = 30 yr for T >= 40.
    let mut w = create(out, &format!("lyapunov_curve{tag}.csv"))?;
    writeln!(
        w,
        "t_years,ftle_mean,ftle_std,slope_mean,slope_std,megno_rate_mean,megno_rate_std"
    )?;
    let t_a = 30.0f64.min(years * 0.3);
    for i in 0..nt {
        let f: Vec<f64> = pairs.iter().map(|p| lam(p, i)).collect();
        let (mut sl, mut mr) = (Vec::new(), Vec::new());
        if t[i] > t_a + 5.0 {
            let ia = t.iter().position(|x| *x >= t_a).unwrap_or(0);
            for p in &pairs {
                sl.push(ols(&t[ia..=i], &p.log_growth[ia..=i]).0);
                let ybar_t: Vec<f64> = (ia..=i).map(|k| p.megno_mean[k]).collect();
                mr.push(2.0 * ols(&t[ia..=i], &ybar_t).0);
            }
        }
        let (sm, ss) = if sl.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            (mean(&sl), std(&sl))
        };
        let (mm, ms) = if mr.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            (mean(&mr), std(&mr))
        };
        writeln!(
            w,
            "{:.3},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}",
            t[i],
            mean(&f),
            std(&f),
            sm,
            ss,
            mm,
            ms
        )?;
    }
    // Final numbers over [t_a, years].
    let ia = t.iter().position(|x| *x >= 30.0).unwrap_or(0);
    let slopes: Vec<f64> = pairs
        .iter()
        .map(|p| ols(&t[ia..], &p.log_growth[ia..]).0)
        .collect();
    let mrates: Vec<f64> = pairs
        .iter()
        .map(|p| 2.0 * ols(&t[ia..], &p.megno_mean[ia..]).0)
        .collect();
    let ftle: Vec<f64> = pairs.iter().map(|p| lam(p, nt - 1)).collect();
    let mut wp = create(out, &format!("lyapunov_pairs{tag}.csv"))?;
    writeln!(
        wp,
        "pair,ftle_end_per_yr,slope_30_end_per_yr,megno_rate_30_end_per_yr,slope_30_60,slope_60_end"
    )?;
    let window = |p: &lyapunov::Pair, lo: f64, hi: f64| {
        let ia = t.iter().position(|x| *x >= lo).unwrap_or(0);
        let ib = t.iter().position(|x| *x >= hi).unwrap_or(nt - 1);
        ols(&t[ia..=ib], &p.log_growth[ia..=ib]).0
    };
    let s1: Vec<f64> = pairs.iter().map(|p| window(p, 30.0, 60.0)).collect();
    let s2: Vec<f64> = pairs.iter().map(|p| window(p, 60.0, years)).collect();
    for k in 0..pairs.len() {
        writeln!(
            wp,
            "{k},{:.5},{:.5},{:.5},{:.5},{:.5}",
            ftle[k], slopes[k], mrates[k], s1[k], s2[k]
        )?;
    }
    // Two populations: Kepler shear alone gives a slope of about 1/t (0.02 to 0.03 per
    // year at 30 to 100 years); chaotic trajectories give 0.1 or more.
    let thr = 0.1;
    let chaotic: Vec<f64> = slopes.iter().copied().filter(|s| *s > thr).collect();
    let regular: Vec<f64> = slopes.iter().copied().filter(|s| *s <= thr).collect();
    println!(
        "threshold {thr}/yr on the slope over 30 to {years} yr: {} chaotic, {} regular",
        chaotic.len(),
        regular.len()
    );
    if chaotic.len() > 1 {
        println!(
            "  chaotic: mean {:.4} std {:.4} (se {:.4}) median {:.4} min {:.4} max {:.4}; Lyapunov time {:.1} yr",
            mean(&chaotic),
            std(&chaotic),
            std(&chaotic) / (chaotic.len() as f64).sqrt(),
            quantile(&chaotic, 0.5),
            quantile(&chaotic, 0.0),
            quantile(&chaotic, 1.0),
            1.0 / mean(&chaotic)
        );
    }
    if regular.len() > 1 {
        println!(
            "  regular: mean {:.4} std {:.4} (1/t at 65 yr = {:.4})",
            mean(&regular),
            std(&regular),
            1.0 / 65.0
        );
    }
    let switch = (0..pairs.len())
        .filter(|&k| (s1[k] > thr) != (s2[k] > thr))
        .count();
    println!("  pairs that change class between years 30-60 and 60-{years}: {switch}");
    let n = pairs.len() as f64;
    println!(
        "plain FTLE at {} yr: mean {:.4} std {:.4} (se {:.4}) per yr",
        t[nt - 1],
        mean(&ftle),
        std(&ftle),
        std(&ftle) / n.sqrt()
    );
    println!(
        "slope of L over 30 to {} yr: mean {:.4} std {:.4} (se {:.4}) per yr; median {:.4}",
        t[nt - 1],
        mean(&slopes),
        std(&slopes),
        std(&slopes) / n.sqrt(),
        quantile(&slopes, 0.5)
    );
    println!(
        "2 d<Y>/dt over 30 to {} yr: mean {:.4} std {:.4} (se {:.4}) per yr",
        t[nt - 1],
        mean(&mrates),
        std(&mrates),
        std(&mrates) / n.sqrt()
    );
    for (lo, hi) in [(10.0, 30.0), (30.0, 60.0), (60.0, 100.0)] {
        if hi > years {
            continue;
        }
        let (ia, ib) = (
            t.iter().position(|x| *x >= lo).unwrap_or(0),
            t.iter().position(|x| *x >= hi).unwrap_or(nt - 1),
        );
        let s: Vec<f64> = pairs
            .iter()
            .map(|p| ols(&t[ia..=ib], &p.log_growth[ia..=ib]).0)
            .collect();
        println!(
            "slope of L over {lo} to {hi} yr: mean {:.4} std {:.4}",
            mean(&s),
            std(&s)
        );
    }
    let _ = Class::Bounded;
    Ok(())
}
