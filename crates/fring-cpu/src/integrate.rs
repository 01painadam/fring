//! The particle loops: serial functions that the parallel backend and the tests share.

use fring_model::physics::{KickTable, SaturnField, dkd_step, wh_steps, wh_steps_moons};
use fring_model::{Error, Particles, Result, Vec3};

/// Advances every particle by `n_steps` Wisdom-Holman steps of `h_s` seconds
/// in Saturn's field (the baseline integrator, ADR 0003).
///
/// Each particle is independent, so the loop runs particle by particle (the
/// state stays in the CPU cache). Saturn's field only, no moons, serial; the
/// backend uses [`advance_chunk`] with rayon.
///
/// Returns `Error::Backend` if a particle has an unbound orbit (the Kepler
/// drift needs a bound orbit). Particles before it have been advanced; the
/// failing particle and the ones after it are unchanged.
pub fn step_particles(
    field: &SaturnField<f64>,
    p: &mut Particles,
    h_s: f64,
    n_steps: u64,
) -> Result<()> {
    for i in 0..p.len() {
        let mut pos = Vec3::new(p.x_km[i], p.y_km[i], p.z_km[i]);
        let mut vel = Vec3::new(p.vx_km_s[i], p.vy_km_s[i], p.vz_km_s[i]);
        if wh_steps(field, &mut pos, &mut vel, h_s, n_steps).is_none() {
            return Err(Error::Backend(format!(
                "particle {} (index {i}) has an unbound or invalid orbit",
                p.id[i]
            )));
        }
        store(p, i, pos, vel);
    }
    Ok(())
}

/// The plain drift-kick-drift leapfrog, kept as a reference (ADR 0003).
/// Same loop as [`step_particles`].
pub fn step_particles_leapfrog(
    field: &SaturnField<f64>,
    p: &mut Particles,
    h_s: f64,
    n_steps: u64,
) {
    for i in 0..p.len() {
        let mut pos = Vec3::new(p.x_km[i], p.y_km[i], p.z_km[i]);
        let mut vel = Vec3::new(p.vx_km_s[i], p.vy_km_s[i], p.vz_km_s[i]);
        for _ in 0..n_steps {
            dkd_step(field, &mut pos, &mut vel, h_s);
        }
        store(p, i, pos, vel);
    }
}

fn store(p: &mut Particles, i: usize, pos: Vec3<f64>, vel: Vec3<f64>) {
    p.x_km[i] = pos.x;
    p.y_km[i] = pos.y;
    p.z_km[i] = pos.z;
    p.vx_km_s[i] = vel.x;
    p.vy_km_s[i] = vel.y;
    p.vz_km_s[i] = vel.z;
}

/// Energy per unit mass (km² s⁻²) of particle `i`.
pub fn particle_energy(field: &SaturnField<f64>, p: &Particles, i: usize) -> f64 {
    field.specific_energy(p.position_km(i), p.velocity_km_s(i))
}

/// State of one particle: `[x, y, z, vx, vy, vz]` in km and km/s (inertial).
pub type State = [f64; 6];

/// Advances every particle of `chunk` by the steps of one block, serially.
/// With `Some(table)` the block is `table.len()` steps with the moon kicks;
/// with `None` it is `n_steps` steps in Saturn's field only.
///
/// This is the only code that integrates a particle, so the parallel backend
/// gives the same bits as a serial loop over this function, whatever the
/// number of threads or the chunk size.
///
/// Returns the index in `chunk` of the first particle with an unbound or
/// invalid orbit (that particle is unchanged), or `None` if all went well.
/// The particles after the failing one are advanced.
pub fn advance_chunk(
    field: &SaturnField<f64>,
    table: Option<&KickTable<f64>>,
    chunk: &mut [State],
    h_s: f64,
    n_steps: u64,
) -> Option<usize> {
    let mut first_failure = None;
    for (i, s) in chunk.iter_mut().enumerate() {
        let mut pos = Vec3::new(s[0], s[1], s[2]);
        let mut vel = Vec3::new(s[3], s[4], s[5]);
        let ok = match table {
            Some(t) => wh_steps_moons(field, t, &mut pos, &mut vel, h_s),
            None => wh_steps(field, &mut pos, &mut vel, h_s, n_steps),
        };
        if ok.is_some() {
            *s = [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z];
        } else if first_failure.is_none() {
            first_failure = Some(i);
        }
    }
    first_failure
}
