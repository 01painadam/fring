//! Closest approach of every particle to Prometheus, from a CPU run that
//! steps one time step at a time (a run of `fring-run` only writes frames
//! every few steps, which is too coarse: an encounter at 220 km moves about
//! 14 km per step).

use fring_cpu::CpuBackend;
use fring_model::{Backend, Error, Result};

use crate::setup::Setup;

/// Result for every particle, in id order.
#[derive(Clone, Debug, Default)]
pub struct Closest {
    /// Particle ids.
    pub id: Vec<u32>,
    /// Initial radius in km.
    pub r0_km: Vec<f64>,
    /// Minimum distance to the moon over the whole run, km.
    pub min_km: Vec<f64>,
    /// Minimum distance over the first half of the run, km.
    pub min_half_km: Vec<f64>,
    /// Steps done.
    pub steps: u64,
}

/// Steps the CPU backend and tracks the minimum distance to the moon called
/// `moon` (it must be in `[moons] include`). `log` gets a progress line about
/// every tenth of the run.
pub fn run(setup: &Setup, moon: &str, mut log: impl FnMut(&str)) -> Result<Closest> {
    let k = setup
        .system
        .moons
        .tracks
        .iter()
        .position(|t| t.name == moon)
        .ok_or_else(|| Error::Config(format!("moon {moon:?} is not in [moons] include")))?;
    let n_steps = setup.config.time.n_steps();
    let mut backend = CpuBackend::new(&setup.config, &setup.system, &setup.init)?;
    let n = setup.init.particles.len();
    let mut min = vec![f64::INFINITY; n];
    let mut min_half = vec![f64::INFINITY; n];
    let mut r0 = Vec::with_capacity(n);
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        r0.push(setup.init.particles.position_km(i).norm());
        ids.push(setup.init.particles.id[i]);
    }
    let start = std::time::Instant::now();
    for step in 1..=n_steps {
        backend.step(1)?;
        let m = backend.moons().moons[k].position_km;
        let view = backend.particles()?;
        let half = step <= n_steps / 2;
        for i in 0..n {
            let d = (view.particles.position_km(i) - m).norm();
            if d < min[i] {
                min[i] = d;
            }
            if half && d < min_half[i] {
                min_half[i] = d;
            }
        }
        if step % (n_steps / 10).max(1) == 0 {
            log(&format!(
                "closest: step {step} of {n_steps}, {:.0} s",
                start.elapsed().as_secs_f64()
            ));
        }
    }
    Ok(Closest {
        id: ids,
        r0_km: r0,
        min_km: min,
        min_half_km: min_half,
        steps: n_steps,
    })
}
