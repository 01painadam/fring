//! ICR-004 test D: energy of sub-stepped Wisdom-Holman steps in the circular
//! restricted three-body problem (Saturn as a point mass, Prometheus on a
//! circular orbit, massless particles).
//!
//! The Jacobi constant per unit mass in the frame that turns with the moon is
//!
//! ```text
//! C_J = 1/2 |v_rot|^2 - 1/2 n_m^2 (x^2 + y^2) - GM_S/r1 - GM_m/r2
//!     = 1/2 |v|^2 - n_m (x vy - y vx) - GM_S/r1 - GM_m/r2    (inertial v)
//! ```
//!
//! and is exactly conserved in the circular restricted problem. The backend
//! works in Saturn-centred inertial coordinates and adds Saturn's reflex
//! (indirect) acceleration `-GM_m r_m / |r_m|^3` (ADR 0004). In the rotating
//! frame this is a static uniform force, so the quantity that the simulated
//! system conserves exactly is `C_J + g (r . m_hat)` with `g = GM_m / a_m^2`.
//! The test uses this quantity (its term is 7.7e-8 km² s⁻² times the cosine
//! of the longitude from the moon, which would swamp the numbers). Set
//! `E1_D_LITERAL=1` for the formula of ICR-004 without the term.
//!
//! Limits (ICR-004 amendment 2): (i) p95 of |dCJ| after the pass <= 1e-6
//! km² s⁻² for the sub-stepped WH and for today's WH (b >= 1 R_H); the ratio
//! of the two is printed, not a limit. (ii) rms at N = 100 <= 2.7e-6 km² s⁻²
//! for both, slope < 0.7. The reference must be below 1e-3 x the limit.
//!
//! Three integrators: the sub-stepped WH
//! (`[substep]`), today's WH (no `[substep]`) and WH with h/64 as the
//! reference (IAS15 is not available in Rust; ICR-004 allows it).
//!
//! Run in release: `cargo test -p fring-cpu --release --test energy_encounter
//! -- --ignored --nocapture`. Set `E1_D_QUICK=1` for a ten times smaller run.

use fring_cpu::CpuBackend;
use fring_model::{
    Backend, BodyState, InitialState, MoonEphemeris, MoonTrack, Particles, RunConfig, SaturnParams,
    System, Vec3,
};

const H: f64 = 360.0;
const YEAR_S: f64 = 365.25 * 86_400.0;
const EPH_DT_DEFAULT: f64 = 900.0;
const EPH_YEARS: f64 = 53.0;

struct Setup {
    gm_s: f64,
    gm_m: f64,
    a_m: f64,
    n_m: f64,
    r_hill: f64,
    sys: System,
}

fn setup() -> Setup {
    let eph_dt: f64 = std::env::var("E1_D_EPH_DT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(EPH_DT_DEFAULT);
    let sp = fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
        .unwrap();
    let moons =
        fring_model::params::moons_from_toml_str(include_str!("../../../params/moons.toml"))
            .unwrap();
    let prom = &moons[0];
    assert_eq!(prom.name, "prometheus");
    let (gm_s, gm_m) = (sp.gm_km3_s2, prom.gm_km3_s2);
    let a_m = prom.state.position_km.norm();
    let n_m = (gm_s / (a_m * a_m * a_m)).sqrt();
    let r_hill = a_m * (gm_m / (3.0 * gm_s)).cbrt();
    let n = (EPH_YEARS * YEAR_S / eph_dt) as usize + 2;
    let mut samples = Vec::with_capacity(n);
    let mut acc = Vec::with_capacity(n);
    for k in 0..n {
        let th = n_m * k as f64 * eph_dt;
        let (c, s) = (th.cos(), th.sin());
        samples.push(BodyState {
            position_km: Vec3::new(a_m * c, a_m * s, 0.0),
            velocity_km_s: Vec3::new(-a_m * n_m * s, a_m * n_m * c, 0.0),
        });
        acc.push(Vec3::new(-n_m * n_m * a_m * c, -n_m * n_m * a_m * s, 0.0));
    }
    let eph = MoonEphemeris {
        start_s: 0.0,
        interval_s: eph_dt,
        tracks: vec![MoonTrack {
            name: "prometheus".to_string(),
            gm_km3_s2: gm_m,
            samples,
            accelerations_km_s2: acc,
        }],
    };
    let saturn = SaturnParams {
        j2: 0.0,
        j4: 0.0,
        j6: 0.0,
        ..sp
    };
    Setup {
        gm_s,
        gm_m,
        a_m,
        n_m,
        r_hill,
        sys: System { saturn, moons: eph },
    }
}

fn config(step_s: f64, extra: &str) -> RunConfig {
    let text = include_str!("../../fring-model/tests/data/example-run.toml");
    let t = text
        .replace("step_s = 360.0", &format!("step_s = {step_s}"))
        .replace(
            "duration_years = 1.0",
            &format!("duration_years = {}", EPH_YEARS - 1.0),
        )
        .replace(r#"["prometheus", "pandora"]"#, r#"["prometheus"]"#);
    RunConfig::from_toml_str(&format!("{t}\n{extra}")).unwrap()
}

/// Jacobi constant per unit mass, km² s⁻², of particle `i` at time `t_s`.
fn jacobi(s: &Setup, p: &Particles, i: usize, t_s: f64) -> f64 {
    let (pos, vel) = (p.position_km(i), p.velocity_km_s(i));
    let th = s.n_m * t_s;
    let moon = Vec3::new(s.a_m * th.cos(), s.a_m * th.sin(), 0.0);
    let lz = pos.x * vel.y - pos.y * vel.x;
    let reflex = if std::env::var("E1_D_LITERAL").is_ok() {
        0.0
    } else {
        s.gm_m / (s.a_m * s.a_m) * pos.dot(moon) / s.a_m
    };
    0.5 * vel.norm_squared() - s.n_m * lz - s.gm_s / pos.norm() - s.gm_m / (pos - moon).norm()
        + reflex
}

fn make_backend(s: &Setup, kind: Kind, init: &InitialState) -> CpuBackend {
    let (step, extra) = match kind {
        Kind::Substep => (H, "[substep]\naccuracy = 0.1\nmax_level = 4\n"),
        Kind::Today => (H, ""),
        Kind::Reference => (H / 64.0, ""),
    };
    let mut b = CpuBackend::new(&config(step, extra), &s.sys, init).unwrap();
    b.set_moon_radii_km(&[43.09]).unwrap();
    b
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Substep,
    Today,
    Reference,
}

fn steps_per_h(kind: Kind) -> u64 {
    if kind == Kind::Reference { 64 } else { 1 }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn circular_particle(s: &Setup, a_km: f64, phase: f64, id: u32, p: &mut Particles) {
    let v = (s.gm_s / a_km).sqrt();
    let (c, sn) = (phase.cos(), phase.sin());
    p.push(
        id,
        Vec3::new(a_km * c, a_km * sn, 0.0),
        Vec3::new(-v * sn, v * c, 0.0),
    );
}

/// nearest-rank percentile of an already sorted slice
fn pct(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    sorted[((q * n as f64).ceil() as usize).clamp(1, n) - 1]
}

fn quick() -> bool {
    std::env::var("E1_D_QUICK").is_ok()
}

/// Moon distance of every particle at time `t_s`.
fn distances(s: &Setup, p: &Particles, t_s: f64) -> Vec<f64> {
    let th = s.n_m * t_s;
    let moon = Vec3::new(s.a_m * th.cos(), s.a_m * th.sin(), 0.0);
    (0..p.len())
        .map(|i| (p.position_km(i) - moon).norm())
        .collect()
}

/// Runs the reference (h/64) for `n_steps` steps of H and returns the closest
/// distance to the moon of every particle (sampled every 16 sub-steps, 90 s,
/// 4.5 km at 0.05 km/s) and the final state.
fn closest_distances(s: &Setup, init: &InitialState, n_steps: u64) -> (Vec<f64>, Particles) {
    let mut b = make_backend(s, Kind::Reference, init);
    let mut dmin = vec![f64::MAX; init.particles.len()];
    let sub = 16u64;
    for m in 1..=(n_steps * 64 / sub) {
        b.step(sub).unwrap();
        let t = m as f64 * sub as f64 * H / 64.0;
        let d = distances(s, &b.particles().unwrap().particles.clone(), t);
        for (m, d) in dmin.iter_mut().zip(d) {
            *m = m.min(d);
        }
    }
    (dmin, b.particles().unwrap().particles.clone())
}

/// Particles on circular orbits `b` km outside the moon's orbit, `ahead_rh`
/// Hill radii ahead of the moon (the moon overtakes them).
fn pass_particles(s: &Setup, bs_rh: &[f64], ahead_rh: &[f64]) -> InitialState {
    let mut p = Particles::with_capacity(bs_rh.len());
    for (i, (b, ah)) in bs_rh.iter().zip(ahead_rh).enumerate() {
        let a = s.a_m + b * s.r_hill;
        circular_particle(s, a, ah * s.r_hill / a, i as u32, &mut p);
    }
    InitialState {
        time_s: 0.0,
        particles: p,
    }
}

/// (i) One encounter per particle.
///
/// The strata are those of the measured closest distance to the moon (as in
/// E1-REF). Particles on circular orbits `b` outside the moon's orbit pass
/// it from 10 R_H upstream; the closest distance is a function of `b` (a
/// pilot run with the reference measures it), so `b` is chosen for targets
/// uniform in each stratum; the strata use the distance measured in the real
/// set.
#[test]
#[ignore = "release: about a minute; ICR-004 test D (i)"]
fn d_i_one_encounter_per_particle() {
    let s = setup();
    let per = if quick() { 50 } else { 500 };
    let strata = [(0.68, 1.0), (1.0, 2.0), (2.0, 4.1), (4.1, 6.3)];
    // End time: the slowest pass (b = 0.4 R_H) is 11.5 R_H past the moon.
    let u_min = 1.5 * s.n_m * 0.4 * s.r_hill;
    let n_steps = (1.3 * (2.0 * 11.5 * s.r_hill) / u_min / H).ceil() as u64;
    let t_end = n_steps as f64 * H;
    // Pilot: closest distance against b on the branch below the near-collision band.
    let n_pilot = 400;
    let pb: Vec<f64> = (0..n_pilot)
        .map(|i| 0.4 + (1.66 - 0.4) * i as f64 / (n_pilot - 1) as f64)
        .collect();
    let (pd, _) = closest_distances(&s, &pass_particles(&s, &pb, &vec![10.0; n_pilot]), n_steps);
    let monotone = pd.windows(2).filter(|w| w[1] > w[0]).count();
    eprintln!(
        "D(i) pilot: {n_pilot} values of b in [0.4, 1.66] R_H; closest distance {:.2} to {:.2} R_H; {monotone} increases",
        pd[n_pilot - 1] / s.r_hill,
        pd[0] / s.r_hill
    );
    // b for a target distance d (R_H): first pilot point where the distance falls to d.
    let b_for = |d_rh: f64| -> f64 {
        let d = d_rh * s.r_hill;
        let k = pd
            .iter()
            .position(|&x| x <= d)
            .unwrap_or(n_pilot - 1)
            .max(1);
        let f = (pd[k - 1] - d) / (pd[k - 1] - pd[k]).max(1e-12);
        pb[k - 1] + f.clamp(0.0, 1.0) * (pb[k] - pb[k - 1])
    };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut bs, mut ahead) = (Vec::new(), Vec::new());
    for &(lo, hi) in &strata {
        for _ in 0..(per * 12 / 10) {
            bs.push(b_for(lo + (hi - lo) * rng.next()));
            ahead.push(10.0 + 3.0 * (rng.next() - 0.5));
        }
    }
    let init = pass_particles(&s, &bs, &ahead);
    let c0: Vec<f64> = (0..init.particles.len())
        .map(|i| jacobi(&s, &init.particles, i, 0.0))
        .collect();
    let (dmin, _) = closest_distances(&s, &init, n_steps);
    // |dCJ| at the end and the largest |dCJ| during the pass (every step of H).
    let mut dc = Vec::new();
    let mut dc_max = Vec::new();
    for kind in [Kind::Substep, Kind::Today, Kind::Reference] {
        let t0 = std::time::Instant::now();
        let mut b = make_backend(&s, kind, &init);
        let mut max = vec![0.0f64; init.particles.len()];
        let mut last = Vec::new();
        for m in 1..=n_steps {
            b.step(steps_per_h(kind)).unwrap();
            let pv = b.particles().unwrap().particles.clone();
            last = (0..pv.len())
                .map(|i| (jacobi(&s, &pv, i, m as f64 * H) - c0[i]).abs())
                .collect();
            for (mx, d) in max.iter_mut().zip(&last) {
                *mx = mx.max(*d);
            }
        }
        dc.push(last);
        dc_max.push(max);
        eprintln!(
            "D(i) {kind:?}: {n_steps} steps of {H} s, {:.1} s",
            t0.elapsed().as_secs_f64()
        );
    }
    eprintln!(
        "D(i) R_H = {:.3} km, end time {t_end:.0} s ({n_steps} steps)",
        s.r_hill
    );
    let mut ok = true;
    for &(lo, hi) in &strata {
        // Particles whose measured closest distance is in the stratum, first `per`.
        let idx: Vec<usize> = (0..dmin.len())
            .filter(|&i| dmin[i] >= lo * s.r_hill && dmin[i] < hi * s.r_hill)
            .take(per)
            .collect();
        let sel = |v: &Vec<f64>| -> Vec<f64> {
            let mut x: Vec<f64> = idx.iter().map(|&i| v[i]).collect();
            x.sort_by(|a, b| a.partial_cmp(b).unwrap());
            x
        };
        let (a, t, r) = (sel(&dc[0]), sel(&dc[1]), sel(&dc[2]));
        let (pa, pt, pr) = (pct(&a, 0.95), pct(&t, 0.95), pct(&r, 0.95));
        let (ma, mt, mr) = (
            pct(&sel(&dc_max[0]), 0.95),
            pct(&sel(&dc_max[1]), 0.95),
            pct(&sel(&dc_max[2]), 0.95),
        );
        eprintln!(
            "D(i)   during the pass (information): p95 of max |dCJ| substep {ma:.3e}  today {mt:.3e}  ratio {:.3e}  reference {mr:.3e}",
            ma / mt
        );
        eprintln!(
            "D(i) stratum [{lo}, {hi}) R_H: n = {}, p95 |dCJ| substep {pa:.3e}  today {pt:.3e}  ratio {:.3e}  reference(h/64) {pr:.3e}  reference/substep {:.3e}  (median substep {:.3e}, today {:.3e})",
            idx.len(),
            pa / pt,
            pr / pa,
            pct(&a, 0.5),
            pct(&t, 0.5)
        );
        // ICR-004 amendment 2: the absolute limit holds for both schemes (today's
        // WH is the production setting); the ratio to today is printed above
        // and is not a limit; the reference must be below 1e-3 x the limit.
        if lo >= 1.0 {
            let pass = pa <= 1.0e-6 && pt <= 1.0e-6;
            let ref_ok = pr <= 1.0e-3 * 1.0e-6;
            eprintln!(
                "D(i)   p95 <= 1e-6 for substep and today: {pass}; reference below 1e-9 (1e-3 x the limit): {ref_ok}"
            );
            ok &= pass && ref_ok && idx.len() >= per * 9 / 10;
        }
    }
    assert!(
        ok,
        "a limit of ICR-004 test D (i) is not met: see the D(i) lines"
    );
}

/// (ii) Many encounters.
#[test]
#[ignore = "release: several minutes; ICR-004 test D (ii)"]
fn d_ii_many_encounters() {
    let s = setup();
    let n_p = if quick() { 10 } else { 100 };
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut p = Particles::with_capacity(n_p);
    let mut t_syn = Vec::new();
    for id in 0..n_p {
        let a = s.a_m + 300.0 + 300.0 * rng.next();
        circular_particle(&s, a, std::f64::consts::TAU * rng.next(), id as u32, &mut p);
        let n_i = (s.gm_s / (a * a * a)).sqrt();
        t_syn.push(std::f64::consts::TAU / (s.n_m - n_i));
    }
    let init = InitialState {
        time_s: 0.0,
        particles: p,
    };
    // Sample every 30 days up to the time at which the slowest particle has
    // had N = 100 encounters (about 52 years).
    let sample_steps = 7_200u64; // 30 days
    let t_max = 100.0 * t_syn.iter().cloned().fold(0.0, f64::max);
    let n_samples = (t_max / (sample_steps as f64 * H)).ceil() as usize;
    let n_samples =
        n_samples.min(((EPH_YEARS - 1.5) * YEAR_S / (sample_steps as f64 * H)) as usize);
    let c0: Vec<f64> = (0..n_p)
        .map(|i| jacobi(&s, &init.particles, i, 0.0))
        .collect();
    let n_ref_samples = ((20.0 * YEAR_S) / (sample_steps as f64 * H)) as usize;
    let mut series: Vec<Vec<Vec<f64>>> = Vec::new(); // [kind][sample][particle]
    for kind in [Kind::Substep, Kind::Today, Kind::Reference] {
        let t0 = std::time::Instant::now();
        let mut b = make_backend(&s, kind, &init);
        let mut rows = Vec::new();
        let limit = if kind == Kind::Reference {
            n_ref_samples.min(n_samples)
        } else {
            n_samples
        };
        for m in 1..=limit {
            b.step(sample_steps * steps_per_h(kind)).unwrap();
            let t = m as f64 * sample_steps as f64 * H;
            let pv = b.particles().unwrap().particles.clone();
            rows.push(
                (0..n_p)
                    .map(|i| (jacobi(&s, &pv, i, t) - c0[i]).abs())
                    .collect::<Vec<f64>>(),
            );
        }
        eprintln!(
            "D(ii) {kind:?}: {limit} samples of 30 days, {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        series.push(rows);
    }
    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
    let t_of = |m: usize| (m + 1) as f64 * sample_steps as f64 * H;
    let n_mean = |m: usize| t_syn.iter().map(|ts| t_of(m) / ts).sum::<f64>() / n_p as f64;
    // rms against the mean number of encounters, for the checkpoints.
    let mut fit_x = Vec::new();
    let mut fit_y = Vec::new();
    for m in (0..n_samples).step_by(((n_samples / 12).max(1)).min(n_samples)) {
        let nm = n_mean(m);
        let (ra, rt) = (rms(&series[0][m]), rms(&series[1][m]));
        let rr = if m < series[2].len() {
            Some(rms(&series[2][m]))
        } else {
            None
        };
        eprintln!(
            "D(ii) t = {:6.2} yr  N_mean = {nm:6.1}  rms |dCJ| substep {ra:.3e}  today {rt:.3e}  reference {}",
            t_of(m) / YEAR_S,
            rr.map_or("-".to_string(), |r| format!(
                "{r:.3e} (ratio to substep {:.2e})",
                r / ra
            ))
        );
        if nm >= 5.0 {
            fit_x.push(nm.ln());
            fit_y.push(ra.ln());
        }
    }
    // Least squares slope of ln rms against ln N.
    let k = fit_x.len() as f64;
    let (mx, my) = (fit_x.iter().sum::<f64>() / k, fit_y.iter().sum::<f64>() / k);
    let slope = fit_x
        .iter()
        .zip(&fit_y)
        .map(|(x, y)| (x - mx) * (y - my))
        .sum::<f64>()
        / fit_x.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
    // rms at N = 100: each particle at the sample nearest to its own N = 100.
    let at_100 = |ser: &Vec<Vec<f64>>| -> Vec<f64> {
        (0..n_p)
            .map(|i| {
                let m = ((100.0 * t_syn[i] / (sample_steps as f64 * H)).round() as usize)
                    .clamp(1, ser.len())
                    - 1;
                ser[m][i]
            })
            .collect()
    };
    let (a100, t100) = (rms(&at_100(&series[0])), rms(&at_100(&series[1])));
    // Reference over the checkpoints where it exists. ICR-004 amendment 2: its
    // rms must be below 1e-3 x the limit (2.7e-9); the ratio to the substep rms
    // is printed (the reference is at the rounding floor, about 1e-12).
    let ratio = (0..series[2].len())
        .map(|m| rms(&series[2][m]) / rms(&series[0][m]))
        .fold(0.0f64, f64::max);
    let ref_max = (0..series[2].len())
        .map(|m| rms(&series[2][m]))
        .fold(0.0f64, f64::max);
    eprintln!("D(ii) slope of ln rms against ln N (N_mean >= 5): {slope:.3}");
    eprintln!(
        "D(ii) rms |dCJ| at N = 100: substep {a100:.3e}  today {t100:.3e}  (limit 2.7e-6 for both)"
    );
    eprintln!(
        "D(ii) largest reference rms {ref_max:.3e} (limit 2.7e-9 = 1e-3 x the limit); \
         largest reference/substep ratio {ratio:.3e} (information)"
    );
    let ok = a100 <= 2.7e-6 && t100 <= 2.7e-6 && slope < 0.7 && ref_max <= 1.0e-3 * 2.7e-6;
    assert!(
        ok,
        "a limit of ICR-004 test D (ii) is not met: see the D(ii) lines"
    );
}
