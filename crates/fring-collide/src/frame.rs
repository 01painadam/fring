//! Peculiar velocities and collision cells of the stochastic model
//! (`SPEC.md` sections 5.1 and 5.2).

use fring_model::physics::circular::circular_speed;
use fring_model::physics::gravity::SaturnField;
use fring_model::{Real, Vec3};

/// Peculiar velocity u = v − v_circ(ρ) φ̂ of a particle (`SPEC.md` 5.2).
///
/// `pos_km` and `vel_km_s` are Saturn-centred and inertial. ρ is the
/// cylindrical radius and v_circ the circular speed in the equatorial plane
/// with J2, J4 and J6. The result is inertial too, so a velocity change found
/// with peculiar velocities is added to `vel_km_s` as it is.
pub fn peculiar_velocity<T: Real>(
    field: &SaturnField<T>,
    pos_km: Vec3<T>,
    vel_km_s: Vec3<T>,
) -> Vec3<T> {
    let rho_km = (pos_km.x * pos_km.x + pos_km.y * pos_km.y).sqrt();
    let v_circ_km_s = circular_speed(field, rho_km);
    let phi_hat = Vec3::new(-pos_km.y / rho_km, pos_km.x / rho_km, T::ZERO);
    vel_km_s - phi_hat * v_circ_km_s
}

/// Integer index of a collision cell (`SPEC.md` 5.1). The derived order
/// (radial, then azimuthal, then vertical) is the cell order of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellKey {
    /// ⌊x / L_x⌋.
    pub radial: i64,
    /// ⌊y / L_y⌋ after the wrap, in 0 to `azimuthal_cells` − 1.
    pub azimuthal: u32,
    /// ⌊z / L_z⌋.
    pub vertical: i64,
}

/// Box cells with three sizes in local coordinates (`SPEC.md` 5.1; AJ, Q2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellGrid {
    /// L_x in km.
    pub radial_km: f64,
    /// L_y in km, adjusted so that `azimuthal_cells` × L_y = the domain length.
    pub azimuthal_km: f64,
    /// L_z in km.
    pub vertical_km: f64,
    /// n_y, the number of cells along the domain.
    pub azimuthal_cells: u32,
    /// Domain length D in km: 2π a₀ for the full ring, or the arc length.
    pub domain_length_km: f64,
}

impl CellGrid {
    /// A grid with the requested sizes. L_y becomes D / n_y with
    /// n_y = round(D / requested L_y), at least 1.
    ///
    /// Returns `None` if a size or the domain length is not finite and > 0.
    pub fn new(
        radial_km: f64,
        azimuthal_km_requested: f64,
        vertical_km: f64,
        domain_length_km: f64,
    ) -> Option<Self> {
        let ok = |v: f64| v.is_finite() && v > 0.0;
        if !(ok(radial_km) && ok(azimuthal_km_requested) && ok(vertical_km) && ok(domain_length_km))
        {
            return None;
        }
        let n = (domain_length_km / azimuthal_km_requested).round().max(1.0);
        if n > f64::from(u32::MAX) {
            return None;
        }
        let azimuthal_cells = n as u32;
        Some(Self {
            radial_km,
            azimuthal_km: domain_length_km / f64::from(azimuthal_cells),
            vertical_km,
            azimuthal_cells,
            domain_length_km,
        })
    }

    /// Cell volume V_c = L_x L_y L_z in km³.
    pub fn volume_km3(&self) -> f64 {
        self.radial_km * self.azimuthal_km * self.vertical_km
    }

    /// The cell of local coordinates (x, y, z) in km. y is the arc length
    /// from the start of the domain, a₀ ((φ − φ₀) mod 2π), with φ₀ = 0 for the
    /// full ring and φ₀ = arc centre − Δθ/2 for the arc (`SPEC.md` 5.1). y is
    /// wrapped into [0, D) first, so a y outside one period is allowed.
    ///
    /// Do not pass a₀ (φ mod 2π) for the arc: wrapped into [0, D), it is
    /// correct only when 360°/Δθ is an integer.
    pub fn key(&self, x_km: f64, y_km: f64, z_km: f64) -> CellKey {
        let y = y_km.rem_euclid(self.domain_length_km);
        // rem_euclid can round up to D itself; that point belongs to the last cell.
        let iy =
            ((y / self.azimuthal_km).floor() as i64).clamp(0, i64::from(self.azimuthal_cells) - 1);
        CellKey {
            radial: (x_km / self.radial_km).floor() as i64,
            azimuthal: iy as u32,
            vertical: (z_km / self.vertical_km).floor() as i64,
        }
    }
}
