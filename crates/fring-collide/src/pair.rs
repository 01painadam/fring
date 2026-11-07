//! The pair operator, the impact geometry and the restitution laws
//! (`SPEC.md` sections 4, 5.4 and 6).
//!
//! These functions are the reference that every model uses: the stochastic
//! model on the CPU (K2) and on the GPU (K5, WGSL twin), and the hard-sphere
//! model (K3). They are generic over [`Real`], so a test can run them in f32
//! to compare with the WGSL twin.

use std::cmp::Ordering;

use fring_model::config::Restitution;
use fring_model::{Real, Vec3};

/// The velocity change and energy loss of one collision (`SPEC.md` section 4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Impulse<T: Real> {
    /// Velocity change of particle i in km/s. Particle j gets the negative.
    pub delta_ui_km_s: Vec3<T>,
    /// Kinetic energy lost by the pair, ¼ (1 − ε²) gₙ², per unit mass of one
    /// particle, in km² s⁻².
    pub energy_loss_km2_s2: T,
    /// The coefficient of restitution that was used.
    pub epsilon: T,
}

impl<T: Real> Impulse<T> {
    /// Adds the velocity change to particle i and subtracts it from particle j.
    #[inline]
    pub fn apply(&self, ui_km_s: &mut Vec3<T>, uj_km_s: &mut Vec3<T>) {
        *ui_km_s += self.delta_ui_km_s;
        *uj_km_s -= self.delta_ui_km_s;
    }
}

/// Normal coefficient of restitution ε(vₙ) for the normal impact speed
/// `vn_km_s` ≥ 0 (`SPEC.md` section 6).
///
/// The power law is evaluated in f64 for every `Real` type: `Real` has no
/// `powf`, and the WGSL twin compares within 10⁻⁶ relative.
pub fn epsilon(law: &Restitution, vn_km_s: f64) -> f64 {
    match *law {
        Restitution::Constant { epsilon } => epsilon,
        Restitution::PowerLaw {
            reference_speed_km_s,
            exponent,
        } => {
            if vn_km_s > reference_speed_km_s {
                (vn_km_s / reference_speed_km_s).powf(-exponent)
            } else {
                1.0
            }
        }
    }
}

/// The collision of particles i and j (`SPEC.md` section 4).
///
/// `g_km_s` is uᵢ − uⱼ and `n_hat` is the unit normal from j to i. Returns
/// `None` if the pair does not approach (gₙ = g · n̂ ≥ 0, or not a number).
/// Otherwise the impulse gives uᵢ += −½ (1 + ε) gₙ n̂ and uⱼ −= the same.
pub fn collide<T: Real>(g_km_s: Vec3<T>, n_hat: Vec3<T>, law: &Restitution) -> Option<Impulse<T>> {
    let gn = g_km_s.dot(n_hat);
    if gn.partial_cmp(&T::ZERO) != Some(Ordering::Less) {
        return None;
    }
    let eps = T::from_f64(epsilon(law, gn.abs().to_f64()));
    let half = T::from_f64(0.5);
    let quarter = T::from_f64(0.25);
    Some(Impulse {
        delta_ui_km_s: n_hat * (-(half * (T::ONE + eps) * gn)),
        energy_loss_km2_s2: quarter * (T::ONE - eps * eps) * gn * gn,
        epsilon: eps,
    })
}

/// The reference orthonormal pair (ê₁, ê₂) normal to the unit vector `g_hat`
/// (`SPEC.md` section 5.4).
pub fn basis<T: Real>(g_hat: Vec3<T>) -> (Vec3<T>, Vec3<T>) {
    let axis = if g_hat.z.abs() < T::from_f64(0.9) {
        Vec3::new(T::ZERO, T::ZERO, T::ONE)
    } else {
        Vec3::new(T::ONE, T::ZERO, T::ZERO)
    };
    let c = g_hat.cross(axis);
    let e1 = c / c.norm();
    let e2 = g_hat.cross(e1);
    (e1, e2)
}

/// The random unit normal of an accepted pair of the stochastic model
/// (`SPEC.md` section 5.4).
///
/// The impact parameter is uniform over the disc of radius 2s normal to g:
/// cos θ = √u₁, ψ = 2π u₂, n̂ = −cos θ ĝ + sin θ (cos ψ ê₁ + sin ψ ê₂).
/// So gₙ = −|g| cos θ < 0. `g_km_s` must not be zero; `u1` and `u2` are in
/// (0, 1).
pub fn impact_normal<T: Real>(g_km_s: Vec3<T>, u1: T, u2: T) -> Vec3<T> {
    let g_hat = g_km_s / g_km_s.norm();
    let (e1, e2) = basis(g_hat);
    let cos_theta = u1.sqrt();
    let sin_theta = (T::ONE - u1).sqrt();
    let psi = T::from_f64(2.0) * T::PI * u2;
    (e1 * psi.cos() + e2 * psi.sin()) * sin_theta - g_hat * cos_theta
}
