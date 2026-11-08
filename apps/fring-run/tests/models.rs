//! K2-CPU and K3-CPU acceptance in `fring-run`: a config with
//! `model = "stochastic"` or `"hard_sphere"` runs on the CPU and writes
//! non-zero collision columns to `stats.csv`. Small runs in debug builds.

use fring_io::RunDir;
use fring_run::cli::Args;
use fring_run::run;
use std::path::PathBuf;

/// 600 particles in a clump (sigma 6 km), 50 m radius, moons off, 250 steps.
/// The radius is far above real ring particles so that a few hundred steps
/// are enough (as in `fring-cpu/tests/collisions_hs.rs`).
fn config(collisions: &str) -> String {
    format!(
        "[run]\nname = \"models\"\nbackend = \"cpu\"\nseed = 20261108\nparams_dir = {:?}\n\
         [time]\nstep_s = 360.0\nduration_years = 0.00285\n\
         [frame]\nreference_semi_major_axis_km = 140220.0\n\
         [domain]\nkind = \"full_ring\"\n\
         [moons]\ninclude = []\n\
         [particles]\ncount = 600\n\
         [particles.distribution]\nkind = \"clump\"\nsemi_major_axis_km = 140200.0\n\
         longitude_deg = 10.0\nsigma_km = 6.0\n\
         {collisions}\n\
         [output]\nframe_every_steps = 1000\nstats_every_steps = 50\nvelocities = false\n",
        format!("{}/../../params", env!("CARGO_MANIFEST_DIR"))
    )
}

struct Sums {
    rows: usize,
    collisions: u64,
    loss: f64,
    candidates: u64,
    cells: u64,
}

fn run_model(name: &str, collisions: &str) -> Sums {
    let dir: PathBuf =
        std::env::temp_dir().join(format!("fring-run-models-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = dir.join("c.toml");
    std::fs::write(&cfg, config(collisions)).unwrap();
    let out = dir.join("out");
    let result = run(&Args {
        config: cfg,
        out: Some(out.clone()),
        quiet: true,
    });
    let rows = result.map(|_| RunDir::new(out).read_stats_file().unwrap());
    let _ = std::fs::remove_dir_all(&dir);
    let rows = rows.unwrap();
    let s = Sums {
        rows: rows.len(),
        collisions: rows.iter().map(|r| r.collisions_since_last).sum(),
        loss: rows.iter().map(|r| r.collision.energy_loss_km2_s2).sum(),
        candidates: rows.iter().map(|r| r.collision.candidates).sum(),
        cells: rows.iter().map(|r| r.collision.occupied_cells).sum(),
    };
    println!(
        "{name}: rows {}, collisions {}, loss {:.3e} km2/s2, candidates {}, occupied cells {}",
        s.rows, s.collisions, s.loss, s.candidates, s.cells
    );
    s
}

#[test]
fn stochastic_on_the_cpu_writes_collision_columns() {
    let s = run_model(
        "stochastic",
        "[collisions]\nmodel = \"stochastic\"\nparticle_radius_km = 0.05\nweight = 1.0\n\
         cell_radial_km = 10.0\ncell_azimuthal_km = 20.0\ncell_vertical_km = 10.0\n\
         [collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5",
    );
    assert!(s.rows >= 2);
    assert!(s.collisions > 0, "no collisions");
    assert!(s.loss > 0.0);
    assert!(s.candidates > 0);
    assert!(s.cells > 0);
}

#[test]
fn hard_sphere_on_the_cpu_writes_collision_columns() {
    let s = run_model(
        "hard_sphere",
        "[collisions]\nmodel = \"hard_sphere\"\nparticle_radius_km = 0.05\n\
         [collisions.restitution]\nlaw = \"constant\"\nepsilon = 0.5",
    );
    assert!(s.rows >= 2);
    assert!(s.collisions > 0, "no collisions");
    assert!(s.loss > 0.0);
    // Not a stochastic-model field.
    assert_eq!(s.candidates, 0);
}
