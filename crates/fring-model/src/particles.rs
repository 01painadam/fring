//! Ring particles as given to and returned by a [`Backend`](crate::Backend).
//!
//! Particles are stored as a structure of arrays (one array per component),
//! in the inertial frame, in f64. A GPU backend converts its f32 local
//! coordinates to this form when the host asks for the particles.

use crate::{Error, Result, Vec3};

/// Particle states as a structure of arrays. All arrays have the same length.
///
/// The particle with index `i` has id `id[i]`, position
/// `(x_km[i], y_km[i], z_km[i])` and velocity
/// `(vx_km_s[i], vy_km_s[i], vz_km_s[i])`, inertial frame. The id stays with a
/// particle when a backend reorders particles (for example, sorts them by
/// collision cell).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Particles {
    /// Particle id, unique in a run (0 to count − 1 at the start).
    pub id: Vec<u32>,
    /// x position in km.
    pub x_km: Vec<f64>,
    /// y position in km.
    pub y_km: Vec<f64>,
    /// z position in km.
    pub z_km: Vec<f64>,
    /// x velocity in km/s.
    pub vx_km_s: Vec<f64>,
    /// y velocity in km/s.
    pub vy_km_s: Vec<f64>,
    /// z velocity in km/s.
    pub vz_km_s: Vec<f64>,
}

impl Particles {
    /// Empty arrays with space for `n` particles.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            id: Vec::with_capacity(n),
            x_km: Vec::with_capacity(n),
            y_km: Vec::with_capacity(n),
            z_km: Vec::with_capacity(n),
            vx_km_s: Vec::with_capacity(n),
            vy_km_s: Vec::with_capacity(n),
            vz_km_s: Vec::with_capacity(n),
        }
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.id.len()
    }

    /// True if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    /// Adds one particle at the end.
    pub fn push(&mut self, id: u32, position_km: Vec3<f64>, velocity_km_s: Vec3<f64>) {
        self.id.push(id);
        self.x_km.push(position_km.x);
        self.y_km.push(position_km.y);
        self.z_km.push(position_km.z);
        self.vx_km_s.push(velocity_km_s.x);
        self.vy_km_s.push(velocity_km_s.y);
        self.vz_km_s.push(velocity_km_s.z);
    }

    /// Position of the particle with index `i`, in km.
    pub fn position_km(&self, i: usize) -> Vec3<f64> {
        Vec3::new(self.x_km[i], self.y_km[i], self.z_km[i])
    }

    /// Velocity of the particle with index `i`, in km/s.
    pub fn velocity_km_s(&self, i: usize) -> Vec3<f64> {
        Vec3::new(self.vx_km_s[i], self.vy_km_s[i], self.vz_km_s[i])
    }

    /// Checks that all arrays have the same length and all values are finite.
    pub fn validate(&self) -> Result<()> {
        let n = self.len();
        let arrays = [
            ("x_km", &self.x_km),
            ("y_km", &self.y_km),
            ("z_km", &self.z_km),
            ("vx_km_s", &self.vx_km_s),
            ("vy_km_s", &self.vy_km_s),
            ("vz_km_s", &self.vz_km_s),
        ];
        for (name, values) in arrays {
            if values.len() != n {
                return Err(Error::Input(format!(
                    "particles: {name} has {} values, id has {n}",
                    values.len()
                )));
            }
            if let Some(i) = values.iter().position(|v| !v.is_finite()) {
                return Err(Error::Input(format!(
                    "particles: {name}[{i}] is not finite"
                )));
            }
        }
        Ok(())
    }
}

/// The particles at the start of a run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InitialState {
    /// Start time in seconds since the epoch.
    pub time_s: f64,
    /// Particle states at `time_s`.
    pub particles: Particles,
}

/// A borrowed, time-stamped view of a backend's particles
/// ([`Backend::particles`](crate::Backend::particles)).
#[derive(Clone, Copy, Debug)]
pub struct ParticleView<'a> {
    /// Time of the states in seconds since the epoch.
    pub time_s: f64,
    /// The particle states (inertial, f64).
    pub particles: &'a Particles,
}
