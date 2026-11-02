//! `fring-analyse`: the A1-ANALYSIS tools. Each subcommand reads run
//! directories and writes small CSV files; `python/` makes the figures.
//!
//! ```text
//! fring-analyse map <run> <out_dir> <tag> [r_lo r_hi r_bin lon_bin]
//! fring-analyse compare <run_a> <run_b>
//! fring-analyse closest <config.toml> <out.csv> [<chaos_run>]
//! fring-analyse apse <out_dir> <run>...
//! fring-analyse moons <run>
//! fring-analyse laplace <run_with_both_moons>
//! fring-analyse lyapunov <out_dir> <n_pairs> <years> [threads d0_km renorm_years kick_km seed tag method(tangent|shadow)]
//! ```

use fring_analysis::apse::{self, DAY_S};
use fring_analysis::laplace::{radius_from_mean_motion, secular_apse_rate_rad_s};
use fring_analysis::lyapunov::{self, LyapunovConfig};
use fring_analysis::map::{self, Class};
use fring_analysis::stats::{mean, ols, quantile, std};
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
        Some("compare") => cmd_compare(&args[1..]),
        Some("closest") => cmd_closest(&args[1..]),
        Some("apse") => cmd_apse(&args[1..]),
        Some("moons") => cmd_moons(&args[1..]),
        Some("laplace") => cmd_laplace(&args[1..]),
        Some("lyapunov") => cmd_lyapunov(&args[1..]),
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

fn cmd_closest(a: &[String]) -> Res<()> {
    let s = setup::load(Path::new(&a[0]))?;
    let res = closest::run(&s, "prometheus", |l| eprintln!("{l}"))?;
    // Hill radius of Prometheus from the mean radius of its track.
    let track = s
        .system
        .moons
        .tracks
        .iter()
        .find(|t| t.name == "prometheus")
        .ok_or("no prometheus")?;
    let rm = mean(
        &track
            .samples
            .iter()
            .map(|b| b.position_km.norm())
            .collect::<Vec<_>>(),
    );
    let hill = rm * (track.gm_km3_s2 / (3.0 * s.saturn.gm_km3_s2)).cbrt();
    println!("prometheus mean radius {rm:.1} km, Hill radius {hill:.2} km");
    let classes = if let Some(dir) = a.get(2) {
        Some(map::read_points(Path::new(dir))?)
    } else {
        None
    };
    let mut w = BufWriter::new(File::create(&a[1])?);
    writeln!(w, "id,r0_km,min_km,min_half_km,class")?;
    let mut matched = 0usize;
    for i in 0..res.id.len() {
        let cl = classes.as_ref().and_then(|c| c.get(i)).map_or("", |p| {
            if (p.r_km - res.r0_km[i]).abs() < 1e-6 {
                matched += 1;
                p.class.name()
            } else {
                ""
            }
        });
        writeln!(
            w,
            "{},{:.4},{:.3},{:.3},{}",
            res.id[i], res.r0_km[i], res.min_km[i], res.min_half_km[i], cl
        )?;
    }
    w.flush()?;
    let n = res.id.len() as f64;
    for k in [1.0, 2.0, 3.0, 5.0, 7.3, 10.0] {
        let c = res.min_km.iter().filter(|d| **d < k * hill).count();
        println!(
            "closer than {k:>4} Hill radii ({:>6.1} km): {c} ({:.3} %)",
            k * hill,
            100.0 * c as f64 / n
        );
    }
    // Cumulative table by class.
    let tpath = Path::new(&a[1]).with_extension("cum.csv");
    let mut tw = BufWriter::new(File::create(&tpath)?);
    writeln!(
        tw,
        "hill_radii,km,n_below,frac_below,n_bounded,n_regular,n_uncertain,n_chaotic,n_unmatched"
    )?;
    for k in [0.5, 1.0, 2.0, 3.0, 5.0, 7.3, 10.0, 20.0, 50.0, 100.0] {
        let mut by = [0usize; 5];
        let mut tot = 0usize;
        for i in 0..res.id.len() {
            if res.min_km[i] < k * hill {
                tot += 1;
                let cl = classes
                    .as_ref()
                    .and_then(|c| c.get(i))
                    .filter(|p| (p.r_km - res.r0_km[i]).abs() < 1e-6)
                    .map_or(4, |p| p.class.index());
                by[cl] += 1;
            }
        }
        writeln!(
            tw,
            "{k},{:.1},{tot},{:.5},{},{},{},{},{}",
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
    // Histogram of the minimum distance in Hill radii (log bins), for the figure.
    let mut hw = BufWriter::new(File::create(Path::new(&a[1]).with_extension("hist.csv"))?);
    writeln!(hw, "hill_lo,hill_hi,count")?;
    let edges: Vec<f64> = (0..=40)
        .map(|k| 0.5 * 10f64.powf(k as f64 * 0.075))
        .collect();
    for w2 in edges.windows(2) {
        let c = res
            .min_km
            .iter()
            .filter(|d| **d / hill >= w2[0] && **d / hill < w2[1])
            .count();
        writeln!(hw, "{:.4},{:.4},{c}", w2[0], w2[1])?;
    }
    hw.flush()?;
    let c = res.min_km.iter().filter(|d| **d < track_radius(&s)).count();
    println!("closer than the moon radius: {c}");
    println!(
        "min distance quantiles 0/1/5/50 %: {:.1} {:.1} {:.1} {:.1} km; classes matched {matched}",
        quantile(&res.min_km, 0.0),
        quantile(&res.min_km, 0.01),
        quantile(&res.min_km, 0.05),
        quantile(&res.min_km, 0.5)
    );
    Ok(())
}

fn track_radius(_s: &setup::Setup) -> f64 {
    43.09 // params/moons.toml radius_km of Prometheus
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
        let alpha = a_core.min(a_gc) / a_core.max(a_gc);
        println!(
            "{}: a_gc={a_gc:.1} km, alpha={alpha:.5}, mu={mu:.3e}, secular rate = {:.6} deg/d",
            g.name,
            to_deg_day(rate)
        );
    }
    println!("sum of moons = {:.6} deg/d", to_deg_day(total));
    Ok(())
}

fn cmd_lyapunov(a: &[String]) -> Res<()> {
    let out = &a[0];
    let n_pairs = num(a, 1, 100.0) as u32;
    let years = num(a, 2, 100.0);
    let threads = num(a, 3, 8.0) as usize;
    let (saturn, all) = setup::load_params(Path::new("params"))?;
    let moons: Vec<_> = all
        .into_iter()
        .filter(|m| m.name == "prometheus" || m.name == "pandora")
        .collect();
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
