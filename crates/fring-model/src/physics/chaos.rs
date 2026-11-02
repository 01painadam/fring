//! Chaos indicator: shadow particle and discrete MEGNO (ICR-003, ADR 0008).
//!
//! The CPU backend uses these functions directly. The GPU kernels in
//! `fring-gpu` have twins of them. The code is plain f64 and compiles for wasm32.
//!
//! # Definition
//!
//! A shadow particle starts at the particle position plus `d0` in a random
//! direction (zero velocity offset). At every global step number that is a
//! multiple of K, with `j = 1, 2, ...` counting the renormalisations:
//!
//! 1. d = |position of shadow − position of particle|, l_j = ln(d / d0);
//! 2. the whole 6D separation is scaled by d0 / d (the shadow keeps its direction);
//! 3. sums: L = Σ l_j, S = Σ (j − 1/2) l_j, Y_j = 2 S / j, Ysum = Σ Y_j.
//!
//! Time is counted in renormalisation intervals K h, so S and Y_j need no
//! unit. A linear growth d ∝ t gives Y → 2, a growth d ∝ exp(λ t) gives
//! Y_j = λ j K h, and the mean ⟨Y⟩ = Ysum / j ≈ λ t / 2.

use crate::Vec3;
use crate::rng::{Counter, Philox, Stream};

/// Unit vector for the initial shadow offset of particle `id`: uniform on the
/// sphere, from the Philox stream [`Stream::CHAOS_SHADOW`] (index = `id`,
/// step 0, draw 0). The same for every backend.
pub fn shadow_direction(rng: &Philox, id: u32) -> Vec3<f64> {
    let [u, w] = rng.uniform_f64(Counter {
        index: id,
        step: 0,
        stream: Stream::CHAOS_SHADOW,
        draw: 0,
    });
    let z = 2.0 * u - 1.0;
    let r = (1.0 - z * z).sqrt();
    let phi = 2.0 * std::f64::consts::PI * w;
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

/// Moves `shadow` back to the position separation `d0_km` from `main` along
/// the 6D separation vector, and returns l = ln(d / d0) with d the separation
/// before. Returns `None` (shadow unchanged) if d is zero or not finite.
pub fn renormalise(main: &[f64; 6], shadow: &mut [f64; 6], d0_km: f64) -> Option<f64> {
    let dx = shadow[0] - main[0];
    let dy = shadow[1] - main[1];
    let dz = shadow[2] - main[2];
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    if !(d.is_finite() && d > 0.0) {
        return None;
    }
    let s = d0_km / d;
    for k in 0..6 {
        shadow[k] = main[k] + (shadow[k] - main[k]) * s;
    }
    Some((d / d0_km).ln())
}

/// The sums of one particle (see the module documentation).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChaosAccum {
    /// L = Σ l_j.
    pub l: f64,
    /// S = Σ (j − 1/2) l_j.
    pub s: f64,
    /// Σ Y_j.
    pub y_sum: f64,
    /// Number of renormalisations so far.
    pub j: u64,
    /// Bit 0: an interval was skipped (zero or non-finite separation).
    pub flags: u32,
}

impl ChaosAccum {
    /// Adds the log growth `ell` of the next interval (`None` for a bad separation).
    pub fn add(&mut self, ell: Option<f64>) {
        self.j += 1;
        let ell = ell.unwrap_or_else(|| {
            self.flags |= 1;
            0.0
        });
        self.l += ell;
        self.s += (self.j as f64 - 0.5) * ell;
        self.y_sum += 2.0 * self.s / self.j as f64;
    }

    /// Mean MEGNO ⟨Y⟩ (0 before the first renormalisation).
    pub fn megno_mean(&self) -> f64 {
        if self.j == 0 {
            0.0
        } else {
            self.y_sum / self.j as f64
        }
    }

    /// FTLE in s⁻¹ for an interval of `interval_s = K h` seconds.
    pub fn ftle_per_s(&self, interval_s: f64) -> f64 {
        if self.j == 0 {
            0.0
        } else {
            self.l / (self.j as f64 * interval_s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_is_unit_and_repeatable() {
        let rng = Philox::from_seed(7);
        let a = shadow_direction(&rng, 3);
        assert!((a.norm() - 1.0).abs() < 1e-15);
        assert_eq!(a, shadow_direction(&rng, 3));
        assert_ne!(a, shadow_direction(&rng, 4));
    }

    #[test]
    fn linear_growth_gives_megno_two() {
        let mut a = ChaosAccum::default();
        for j in 1..=2000u64 {
            a.add(Some((j as f64 / (j - 1).max(1) as f64).ln().max(0.0)));
        }
        // l_1 = 0 here; the later terms are ln(j / (j-1)).
        assert!((a.y_sum / 2000.0 - 2.0).abs() < 0.02, "{}", a.megno_mean());
    }

    #[test]
    fn exponential_growth_gives_half_lambda_t() {
        let (c, n) = (0.05, 1000u64); // l_j = c = lambda K h
        let mut a = ChaosAccum::default();
        for _ in 0..n {
            a.add(Some(c));
        }
        let expect = c * n as f64 / 2.0;
        assert!((a.megno_mean() - expect).abs() / expect < 0.01);
        assert!((a.ftle_per_s(10.0) - c / 10.0).abs() < 1e-15);
    }

    #[test]
    fn renormalise_scales_the_6d_separation() {
        let main = [1.0e5, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut sh = [1.0e5 + 3e-3, 4e-3, 0.0, 1e-5, 1.0, 2e-5];
        let l = renormalise(&main, &mut sh, 1e-3).unwrap();
        assert!((l - 5.0f64.ln()).abs() < 1e-9);
        assert!(((sh[0] - main[0]).powi(2) + (sh[1] - main[1]).powi(2)).sqrt() - 1e-3 < 1e-12);
        assert!((sh[3] - 2e-6).abs() < 1e-15);
        let mut same = main;
        assert!(renormalise(&main, &mut same, 1e-3).is_none());
    }
}
