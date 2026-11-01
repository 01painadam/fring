//! The leapfrog loop over all particles (no moons yet; they come in C3/C4).

use fring_model::physics::{SaturnField, dkd_step};
use fring_model::{Particles, Vec3};

/// Advances every particle by `n_steps` drift-kick-drift steps of `h_s`
/// seconds in Saturn's field.
///
/// Each particle is independent, so the loop runs particle by particle (the
/// state stays in the CPU cache). C4 will run this loop with rayon.
pub fn step_particles(field: &SaturnField<f64>, p: &mut Particles, h_s: f64, n_steps: u64) {
    for i in 0..p.len() {
        let mut pos = Vec3::new(p.x_km[i], p.y_km[i], p.z_km[i]);
        let mut vel = Vec3::new(p.vx_km_s[i], p.vy_km_s[i], p.vz_km_s[i]);
        for _ in 0..n_steps {
            dkd_step(field, &mut pos, &mut vel, h_s);
        }
        p.x_km[i] = pos.x;
        p.y_km[i] = pos.y;
        p.z_km[i] = pos.z;
        p.vx_km_s[i] = vel.x;
        p.vy_km_s[i] = vel.y;
        p.vz_km_s[i] = vel.z;
    }
}

/// Energy per unit mass (km² s⁻²) of particle `i`.
pub fn particle_energy(field: &SaturnField<f64>, p: &Particles, i: usize) -> f64 {
    field.specific_energy(p.position_km(i), p.velocity_km_s(i))
}
