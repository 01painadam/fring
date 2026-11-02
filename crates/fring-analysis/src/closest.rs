//! Closest approach of every particle to one or more moons, from a CPU run
//! that steps one time step at a time (a run of `fring-run` only writes
//! frames every few steps, which is too coarse). The minimum is taken over
//! the straight segment between two step ends (linear interpolation of the
//! particle and of the moon), so that a fast pass is not missed.

use fring_cpu::CpuBackend;
use fring_model::{Backend, Error, InitialState, Result, Vec3};

use crate::setup::Setup;

/// Result for every particle, in the order of `id`.
#[derive(Clone, Debug, Default)]
pub struct Closest {
    /// Moon names, in the order of the rows of `min_km`.
    pub moons: Vec<String>,
    /// Particle ids.
    pub id: Vec<u32>,
    /// Initial radius in km.
    pub r0_km: Vec<f64>,
    /// Minimum distance to each moon over the whole run, km (`[moon][particle]`).
    pub min_km: Vec<Vec<f64>>,
    /// Minimum distance to each moon over the first half of the run, km.
    pub min_half_km: Vec<Vec<f64>>,
    /// Steps done.
    pub steps: u64,
}

/// Minimum of `|d0 + (d1 - d0) s|` for s in [0, 1] (km).
pub fn segment_min_km(d0: Vec3<f64>, d1: Vec3<f64>) -> f64 {
    let e = d1 - d0;
    let ee = e.dot(e);
    let s = if ee > 0.0 {
        (-d0.dot(e) / ee).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (d0 + e * s).norm()
}

/// Keeps the particles whose initial radius is in `[r_lo_km, r_hi_km)`, at most
/// `n_max` of them (the lowest ids). The ids are kept.
pub fn select_window(
    init: &InitialState,
    r_lo_km: f64,
    r_hi_km: f64,
    n_max: usize,
) -> InitialState {
    let mut p = fring_model::Particles::with_capacity(n_max);
    for i in 0..init.particles.len() {
        let pos = init.particles.position_km(i);
        let r = pos.norm();
        if r >= r_lo_km && r < r_hi_km && p.len() < n_max {
            p.push(init.particles.id[i], pos, init.particles.velocity_km_s(i));
        }
    }
    InitialState {
        time_s: init.time_s,
        particles: p,
    }
}

/// Steps the CPU backend and tracks the minimum distance to every moon in
/// `moons` (each must be in `[moons] include`). `init` is the start state
/// (all of `setup.init`, or a subset from [`select_window`]). `log` gets a
/// progress line about every tenth of the run.
pub fn run(
    setup: &Setup,
    init: &InitialState,
    moons: &[&str],
    mut log: impl FnMut(&str),
) -> Result<Closest> {
    let mut k = Vec::new();
    for name in moons {
        k.push(
            setup
                .system
                .moons
                .tracks
                .iter()
                .position(|t| t.name == *name)
                .ok_or_else(|| Error::Config(format!("moon {name:?} is not in [moons] include")))?,
        );
    }
    let n_steps = setup.config.time.n_steps();
    let n = init.particles.len();
    let mut config = setup.config.clone();
    config.particles.count = n as u32;
    let mut backend = CpuBackend::new(&config, &setup.system, init)?;
    let mut min = vec![vec![f64::INFINITY; n]; k.len()];
    let mut min_half = vec![vec![f64::INFINITY; n]; k.len()];
    let r0: Vec<f64> = (0..n)
        .map(|i| init.particles.position_km(i).norm())
        .collect();
    let ids = init.particles.id.clone();
    let mut prev_p: Vec<Vec3<f64>> = (0..n).map(|i| init.particles.position_km(i)).collect();
    let mut prev_m: Vec<Vec3<f64>> = k
        .iter()
        .map(|&j| backend.moons().moons[j].position_km)
        .collect();
    let start = std::time::Instant::now();
    for step in 1..=n_steps {
        backend.step(1)?;
        let half = step <= n_steps / 2;
        let m_now: Vec<Vec3<f64>> = k
            .iter()
            .map(|&j| backend.moons().moons[j].position_km)
            .collect();
        let view = backend.particles()?;
        for i in 0..n {
            let p1 = view.particles.position_km(i);
            for q in 0..k.len() {
                let d = segment_min_km(prev_p[i] - prev_m[q], p1 - m_now[q]);
                if d < min[q][i] {
                    min[q][i] = d;
                }
                if half && d < min_half[q][i] {
                    min_half[q][i] = d;
                }
            }
            prev_p[i] = p1;
        }
        prev_m = m_now;
        if step % (n_steps / 10).max(1) == 0 {
            log(&format!(
                "closest: step {step} of {n_steps}, {:.0} s",
                start.elapsed().as_secs_f64()
            ));
        }
    }
    Ok(Closest {
        moons: moons.iter().map(|s| (*s).to_string()).collect(),
        id: ids,
        r0_km: r0,
        min_km: min,
        min_half_km: min_half,
        steps: n_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_minimum_of_a_straight_pass() {
        // Passes the origin at 3 km (y = 3) between x = -10 and x = +10.
        let d = segment_min_km(Vec3::new(-10.0, 3.0, 0.0), Vec3::new(10.0, 3.0, 0.0));
        assert!((d - 3.0).abs() < 1e-12);
        // Ends before the closest point: the end point is the minimum.
        let d = segment_min_km(Vec3::new(-10.0, 3.0, 0.0), Vec3::new(-4.0, 3.0, 0.0));
        assert!((d - 5.0).abs() < 1e-12);
        // A zero-length segment.
        let p = Vec3::new(3.0, 4.0, 0.0);
        assert!((segment_min_km(p, p) - 5.0).abs() < 1e-12);
    }
}

/// Mean radius in km of the moon `name`, read from the text of
/// `params/moons.toml` (the `radius_km` line after the `name` line).
pub fn moon_radius_km(moons_toml: &str, name: &str) -> Option<f64> {
    let mut found = false;
    for line in moons_toml.lines() {
        let l = line.trim();
        if l.starts_with("name") {
            found = l.contains(&format!("\"{name}\""));
        } else if found && l.starts_with("radius_km") {
            return l.split('=').nth(1)?.split('#').next()?.trim().parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod radius_tests {
    use super::*;

    #[test]
    fn radii_from_params() {
        let t = include_str!("../../../params/moons.toml");
        assert_eq!(moon_radius_km(t, "pandora"), Some(40.63));
        assert_eq!(moon_radius_km(t, "prometheus"), Some(43.09));
        assert_eq!(moon_radius_km(t, "janus"), None);
    }
}
