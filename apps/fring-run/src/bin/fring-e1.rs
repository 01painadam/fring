//! `fring-e1`: runs the CPU backend on the E1-REF files and writes the WH
//! result files for `validation/scripts/e1_compare.py` (task E1-RUN).
//!
//! Owner: A5 (see `AGENTS.md`). The logic is in `fring_run::e1`.

use fring_run::RunError;
use fring_run::e1::{self, Variant};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "\
fring-e1: run the CPU backend on the E1-REF files (layer 1, layer 2)

USAGE:
    fring-e1 <layer1|layer2|all> [--refdir <dir>] [--params <dir>] [--out <dir>]
             [--variants plain,substep,loss] [--ids <n>] [--l2-step <s>] [--l2-stride <k>]
             [--l2-perturb-km <x> --l2-seed <k>]
             [--l2-div <m>] [--l2-trace-ids i,j,k --l2-trace-every <steps>] [--quiet]

Defaults: --refdir validation/reference, --params params, --out runs/e1,
--variants plain,substep. --ids n runs only the first n ids (tests).
--l2-step s sets the nominal layer 2 step (default 356.2537; the step is then
adjusted so that the end time is exact). Only for noise studies.
--l2-div m runs m times as many steps of h/m. --l2-trace-ids writes trace_l2.csv
(state and Prometheus position every --l2-trace-every steps of h, default 243 = 1 day).
--l2-stride k measures the closest distance from every k-th step only (default 1).
--l2-perturb-km x --l2-seed k add a random offset of length x km to each layer 2
start position (seeded Philox, keyed by k and the id); x = 0 changes nothing.
Files: <out>/<variant>/l1_wh.csv, l2_wh.csv, frozen_l1.csv, frozen_l2.csv;
<out>/moons_diff.csv. Compare with validation/scripts/e1_compare.py.
";

struct Opts {
    layers: (bool, bool),
    refdir: PathBuf,
    params: PathBuf,
    out: PathBuf,
    variants: Vec<Variant>,
    ids: Option<usize>,
    nominal_step_s: f64,
    l2_stride: u64,
    l2_div: u64,
    trace_ids: Vec<u32>,
    trace_every: u64,
    perturb_km: f64,
    seed: u64,
    quiet: bool,
}

fn parse() -> Result<Opts, String> {
    let mut a = std::env::args().skip(1);
    let layers = match a.next().as_deref() {
        Some("layer1") => (true, false),
        Some("layer2") => (false, true),
        Some("all") => (true, true),
        _ => return Err("first argument must be layer1, layer2 or all".into()),
    };
    let mut o = Opts {
        layers,
        refdir: "validation/reference".into(),
        params: "params".into(),
        out: "runs/e1".into(),
        variants: vec![Variant::Plain, Variant::Substep],
        ids: None,
        nominal_step_s: e1::NOMINAL_STEP_S,
        l2_stride: 1,
        l2_div: 1,
        trace_ids: Vec::new(),
        trace_every: 243,
        perturb_km: 0.0,
        seed: 0,
        quiet: false,
    };
    while let Some(f) = a.next() {
        let mut val = |name: &str| a.next().ok_or(format!("{name} needs a value"));
        match f.as_str() {
            "--refdir" => o.refdir = val("--refdir")?.into(),
            "--params" => o.params = val("--params")?.into(),
            "--out" => o.out = val("--out")?.into(),
            "--variants" => {
                o.variants = val("--variants")?
                    .split(',')
                    .map(Variant::parse)
                    .collect::<Result<_, _>>()?
            }
            "--ids" => o.ids = Some(val("--ids")?.parse().map_err(|e| format!("--ids: {e}"))?),
            "--l2-step" => {
                o.nominal_step_s = val("--l2-step")?
                    .parse()
                    .map_err(|e| format!("--l2-step: {e}"))?
            }
            "--l2-stride" => {
                o.l2_stride = val("--l2-stride")?
                    .parse()
                    .map_err(|e| format!("--l2-stride: {e}"))?
            }
            "--l2-perturb-km" => {
                o.perturb_km = val("--l2-perturb-km")?
                    .parse()
                    .map_err(|e| format!("--l2-perturb-km: {e}"))?
            }
            "--l2-seed" => {
                o.seed = val("--l2-seed")?
                    .parse()
                    .map_err(|e| format!("--l2-seed: {e}"))?
            }
            "--l2-div" => {
                o.l2_div = val("--l2-div")?
                    .parse()
                    .map_err(|e| format!("--l2-div: {e}"))?
            }
            "--l2-trace-ids" => {
                o.trace_ids = val("--l2-trace-ids")?
                    .split(',')
                    .map(|x| x.parse().map_err(|e| format!("--l2-trace-ids: {e}")))
                    .collect::<Result<_, _>>()?
            }
            "--l2-trace-every" => {
                o.trace_every = val("--l2-trace-every")?
                    .parse()
                    .map_err(|e| format!("--l2-trace-every: {e}"))?
            }
            "--quiet" | "-q" => o.quiet = true,
            _ => return Err(format!("unknown argument {f:?}")),
        }
    }
    Ok(o)
}

fn write(path: &std::path::Path, text: &str) -> Result<(), RunError> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| RunError::Run(format!("{}: {e}", d.display())))?;
    }
    std::fs::write(path, text).map_err(|e| RunError::Run(format!("{}: {e}", path.display())))
}

fn read(path: &std::path::Path) -> Result<String, RunError> {
    std::fs::read_to_string(path)
        .map_err(|e| RunError::Setup(format!("cannot read {}: {e}", path.display())))
}

fn go(o: &Opts) -> Result<(), RunError> {
    let say = |m: &str| {
        if !o.quiet {
            eprintln!("{m}");
        }
    };
    let (do1, do2) = o.layers;
    let mut l1 = Vec::new();
    if do1 {
        l1 = e1::parse_l1_states(&read(&o.refdir.join("e1_l1_states.csv"))?)?;
        if let Some(n) = o.ids {
            l1.truncate(n);
        }
    }
    let (mut l2, mut days) = (Default::default(), 0.0);
    if do2 {
        l2 = e1::parse_l2_states(&read(&o.refdir.join("e1_l2_states.csv"))?)?;
        if let Some(n) = o.ids {
            l2.id.truncate(n);
            l2.x_km.truncate(n);
            l2.y_km.truncate(n);
            l2.z_km.truncate(n);
            l2.vx_km_s.truncate(n);
            l2.vy_km_s.truncate(n);
            l2.vz_km_s.truncate(n);
        }
        e1::perturb_positions(&mut l2, o.perturb_km, o.seed);
        let j: serde_json_days::Days =
            serde_json_days::parse(&read(&o.refdir.join("e1_l2.json"))?)?;
        days = j.0;
    }
    // The moon-difference check reads all 36 epochs (t0 up to 350 d, plus 2 d).
    let mut cover: f64 = 352.0 * 86_400.0;
    for r in &l1 {
        cover = cover.max(r.t0_s + e1::WINDOW_S);
    }
    if do2 {
        cover = cover.max(days * 86_400.0);
    }
    let t = Instant::now();
    let setup = e1::build_setup(&o.params, cover + 3600.0)?;
    say(&format!(
        "ephemeris to {cover:.0} s in {:.1} s",
        t.elapsed().as_secs_f64()
    ));
    let moons_csv = read(&o.refdir.join("e1_moons_t0.csv"))?;
    let (text, max_r, max_v) = e1::moon_difference(&setup.system.moons, &moons_csv)?;
    write(&o.out.join("moons_diff.csv"), &text)?;
    say(&format!(
        "moon difference to e1_moons_t0.csv: max |dr| {max_r:.3e} km, max |dv| {max_v:.3e} km/s"
    ));
    for &v in &o.variants {
        let dir = o.out.join(v.name());
        if do1 {
            let t = Instant::now();
            let r = e1::layer1(
                &l1,
                &setup.system,
                v,
                e1::cpu_backend(setup.radii_km.clone()),
            )?;
            let head = format!(
                "E1-RUN layer 1, variant {}, CPU, {} steps of {:.6} s per encounter",
                v.name(),
                e1::L1_STEPS,
                r.step_s
            );
            write(&dir.join("l1_wh.csv"), &e1::format_l1(&r.finals, &head))?;
            write(&dir.join("frozen_l1.csv"), &e1::format_frozen(1, &r.frozen))?;
            say(&format!(
                "layer1 {}: {} ids, {} frozen, {:.1} s",
                v.name(),
                r.finals.len(),
                r.frozen.len(),
                t.elapsed().as_secs_f64()
            ));
        }
        if do2 {
            let t = Instant::now();
            let trace = (!o.trace_ids.is_empty()).then(|| e1::Trace {
                ids: o.trace_ids.clone(),
                every_steps: o.trace_every,
            });
            let r = e1::layer2(
                l2.clone(),
                days,
                o.nominal_step_s,
                o.l2_div,
                o.l2_stride,
                trace.as_ref(),
                &setup.system,
                v,
                e1::cpu_backend(setup.radii_km.clone()),
            )?;
            let head = format!(
                "E1-RUN layer 2, variant {}, CPU, {} steps of {:.6} s",
                v.name(),
                r.n_steps,
                r.step_s
            );
            write(&dir.join("l2_wh.csv"), &e1::format_l2(&r.rows, &head))?;
            if trace.is_some() {
                let mut tx = String::from(
                    "# id,t_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s,prom_x_km,prom_y_km,prom_z_km\n",
                );
                for (id, ts, st, mp) in &r.trace {
                    tx.push_str(&format!(
                        "{id},{ts:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e}\n",
                        st[0], st[1], st[2], st[3], st[4], st[5], mp[0], mp[1], mp[2]
                    ));
                }
                write(&dir.join("trace_l2.csv"), &tx)?;
            }
            write(&dir.join("frozen_l2.csv"), &e1::format_frozen(2, &r.frozen))?;
            let (c, chaotic) = e1::l2_summary(&r.rows);
            say(&format!(
                "layer2 {}: {} ids, {} frozen, {:.1} s; inside 1, 3, 6.3 R_H: {} {} {}; chaotic fraction {chaotic:.4}",
                v.name(),
                r.rows.len(),
                r.frozen.len(),
                t.elapsed().as_secs_f64(),
                c[0],
                c[1],
                c[2]
            ));
        }
    }
    Ok(())
}

/// `days` from `e1_l2.json` without a JSON dependency.
mod serde_json_days {
    use fring_run::RunError;
    pub struct Days(pub f64);
    pub fn parse(text: &str) -> Result<Days, RunError> {
        let key = "\"days\"";
        let i = text
            .find(key)
            .ok_or_else(|| RunError::Setup("e1_l2.json has no \"days\"".into()))?;
        let rest = text[i + key.len()..].trim_start_matches([' ', ':']);
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
            .unwrap_or(rest.len());
        rest[..end]
            .parse()
            .map(Days)
            .map_err(|e| RunError::Setup(format!("e1_l2.json days: {e}")))
    }
}

fn main() -> ExitCode {
    if std::env::args().skip(1).any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match parse() {
        Err(m) => {
            eprintln!("fring-e1: {m}\n\n{USAGE}");
            ExitCode::from(2)
        }
        Ok(o) => match go(&o) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("fring-e1: {e}");
                ExitCode::from(e.exit_code())
            }
        },
    }
}
