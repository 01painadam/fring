//! Chaos indicator on the GPU (task G3, ICR-003, ADR 0008): the Rust twin of
//! the kernel `chaos_renorm` in `shaders/chaos.wgsl`, and the host helpers.
//!
//! The kernel does one renormalisation of each particle: the position
//! separation d of the shadow is formed from the double-single difference
//! (never from the hi parts alone), l = ln(d / d0) is added to the sums L, S
//! and Y (double-single accumulators), and the shadow moves back to the
//! separation d0 along the 6D separation vector. The same arithmetic as
//! `fring_model::physics::chaos`, in f32 and DS.

use crate::ds::Ds;
use crate::spike::State;
use fring_model::ChaosReport;

/// Words of accumulator per particle: 4 planes of `vec2<f32>`.
pub const ACC_PLANES: usize = 4;

/// The accumulators of one particle, as in the GPU buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Acc<const M: u8> {
    /// L = Σ l_j (DS).
    pub l: Ds<M>,
    /// S = Σ (j − 1/2) l_j (DS).
    pub s: Ds<M>,
    /// Σ Y_j (DS).
    pub y: Ds<M>,
    /// Number of renormalisations (exact in f32 up to 2^24).
    pub j: f32,
    /// Bit 0 set (value 1.0): a separation was zero or too large.
    pub flags: f32,
}

/// Rust twin of `chaos_renorm` for one particle. `d0` is the f32 constant of
/// the kernel. Only `ln` differs from the GPU (the GPU `log` has a few ulp).
pub fn renorm<const M: u8>(main: &State<M>, shadow: &mut State<M>, acc: &mut Acc<M>, d0: f32) {
    let dp = [0, 1, 2].map(|k| shadow[k] - main[k]);
    let d = (dp[0].hi * dp[0].hi + dp[1].hi * dp[1].hi + dp[2].hi * dp[2].hi).sqrt();
    let j = acc.j + 1.0;
    let mut ell = 0.0f32;
    if d > 0.0 && d < 1.0e18 {
        ell = (d / d0).ln();
        let s = d0 / d;
        for k in 0..6 {
            let diff = if k < 3 { dp[k] } else { shadow[k] - main[k] };
            shadow[k] = main[k] + diff.mul_f(s);
        }
    } else {
        acc.flags = 1.0;
    }
    acc.l = acc.l.add_f(ell);
    acc.s = acc.s.add_f((j - 0.5) * ell);
    acc.y = acc.y.add_f(2.0 * acc.s.hi / j);
    acc.j = j;
}

/// Reads accumulators of `n` particles from the words of the GPU buffer
/// (`4 * cap` vec2, planes of `cap`) and makes the report. `interval_s` is
/// `K h`.
pub fn report(words: &[f32], cap: usize, n: usize, interval_s: f64, time_s: f64) -> ChaosReport {
    let w = |plane: usize, i: usize, k: usize| words[2 * (plane * cap + i) + k] as f64;
    let ds = |plane: usize, i: usize| w(plane, i, 0) + w(plane, i, 1);
    let mut r = ChaosReport {
        time_s,
        ..Default::default()
    };
    for i in 0..n {
        let j = w(3, i, 0);
        r.n_renorm = j as u64;
        let l = ds(0, i);
        r.log_growth.push(l);
        r.ftle_per_s
            .push(if j > 0.0 { l / (j * interval_s) } else { 0.0 });
        r.megno_mean.push(if j > 0.0 { ds(2, i) / j } else { 0.0 });
        r.flags.push(w(3, i, 1) as u32);
    }
    r
}
