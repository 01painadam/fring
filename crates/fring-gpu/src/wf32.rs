//! [`Wf32`]: an f32 that follows the WGSL rules, for Rust twins of the shaders.
//!
//! WGSL guarantees correctly rounded `+`, `-`, `*` and `fma` (when fused) for
//! f32, but not for division and square root (WGSL spec: `x / y` within
//! 2.5 ULP, `sqrt` inherited from `1 / inverseSqrt`), and built-in `sin`
//! and `cos` only within an absolute error of 2⁻¹¹ on [-π, π]. A Rust f32
//! has correctly rounded division and square root, so it is not a faithful
//! twin. `Wf32<M>` is f32 with the division and square root of the error
//! model `M` ([`model`]), and with our own sine and cosine polynomials
//! ([`sin_cos_f32`]) that the shaders also use (`shaders/trig.wgsl`).
//!
//! The model is a const generic parameter, so the same generic physics code
//! (`fring_model::physics`) runs with each model at full speed.

use fring_model::Real;
use fring_model::rng::philox4x32_10;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The error models of division and square root.
pub mod model {
    /// Correctly rounded division and square root (IEEE 754).
    pub const IEEE: u8 = 0;
    /// The Apple M4 GPU as measured by the probe (`docs/adr/0006/probe_m4.txt`):
    /// `a / b = RN(a * r)` with a reciprocal `r` within 0.71 ULP, and
    /// `sqrt(x) = RN(x * s)` with an inverse square root `s` within 0.65 ULP.
    /// The model gives the measured maximum errors (about 1.7 ULP) and about
    /// the same fraction of results that are not correctly rounded.
    pub const M4: u8 = 1;
    /// The WGSL bound: the correctly rounded result moved by -2 to +2 ULP.
    pub const WGSL: u8 = 2;
}

/// An f32 with WGSL arithmetic under the error model `M` (see [`model`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Wf32<const M: u8>(pub f32);

/// A deterministic pseudo-random word from the bits of the operands: the
/// error of a GPU division depends only on its inputs. Uses the Philox
/// generator of `fring-model` (key = model, counter = operands).
#[inline]
fn noise(model: u8, op: u32, a: f32, b: f32) -> u32 {
    philox4x32_10(
        [a.to_bits(), b.to_bits(), op, 0x0006],
        [model as u32, 0x6731],
    )[0]
}

/// `x` moved by `k` units in the last place (toward +inf for k > 0).
#[inline]
fn step_ulps(mut x: f32, k: i32) -> f32 {
    if !x.is_normal() {
        return x;
    }
    for _ in 0..k.abs() {
        x = if k > 0 { x.next_up() } else { x.next_down() };
    }
    x
}

/// `exact` rounded to f32 after a relative perturbation of at most `max_ulp`
/// ULP (uniform, from `w`). Models a hardware estimate within
/// `max_ulp + 1/2` ULP.
#[inline]
fn noisy_round(exact: f64, w: u32, max_ulp: f64) -> f32 {
    let rn = exact as f32;
    if !rn.is_normal() {
        return rn;
    }
    let ulp = crate::probe::ulp_f32(exact);
    let u = (w as f64 + 0.5) / 4_294_967_296.0; // (0, 1)
    (exact + (2.0 * u - 1.0) * max_ulp * ulp) as f32
}

/// `a / b` under the model `M`.
#[inline]
pub fn div<const M: u8>(a: f32, b: f32) -> f32 {
    match M {
        model::IEEE => a / b,
        model::M4 => {
            let r = noisy_round(1.0 / b as f64, noise(M, 1, b, 0.0), 0.21);
            a * r
        }
        _ => {
            let w = noise(M, 2, a, b);
            step_ulps(a / b, (w % 5) as i32 - 2)
        }
    }
}

/// `sqrt(x)` under the model `M` (x >= 0).
#[inline]
pub fn sqrt<const M: u8>(x: f32) -> f32 {
    match M {
        model::IEEE => x.sqrt(),
        model::M4 => {
            if x == 0.0 || !x.is_finite() {
                return x.sqrt();
            }
            let s = noisy_round(1.0 / (x as f64).sqrt(), noise(M, 3, x, 0.0), 0.15);
            x * s
        }
        _ => {
            let w = noise(M, 4, x, 0.0);
            step_ulps(x.sqrt(), (w % 5) as i32 - 2)
        }
    }
}

// Minimax coefficients from docs/adr/0006/remez_sincos.py (Remez exchange,
// t = x² on [0, (π/4)²]): sin x = x + x³ (S1 + t (S2 + t S3)), relative
// error 8.4e-9; cos x = 1 - t/2 + t² (C1 + t (C2 + t C3)), absolute error
// 7.2e-10. Values are the f32 bit patterns printed by the script.
const S1: f32 = f32::from_bits(0xbe2a_aaa3);
const S2: f32 = f32::from_bits(0x3c08_835d);
const S3: f32 = f32::from_bits(0xb94c_838d);
const C1: f32 = f32::from_bits(0x3d2a_aaa7);
const C2: f32 = f32::from_bits(0xbab6_073c);
const C3: f32 = f32::from_bits(0x37cd_37b1);
/// 2/π rounded to f32.
const TWO_OVER_PI: f32 = 0.636_619_75;
/// π/2 = P1 + P2 + P3 (Cody and Waite 1980): P1 and P2 have 12 significant
/// bits, so k P1 and k P2 are exact for |k| < 2^12.
const P1: f32 = 1.570_800_8;
const P2: f32 = -4.453_584_6e-6;
const P3: f32 = -8.705_516e-10;

/// Sine and cosine of `x` (radians, |x| < 6000) in f32: Cody-Waite reduction
/// to r in [-π/4, π/4] and the minimax polynomials above, with fused
/// multiply-adds. Error about 1 ULP. Twin of `sin_cos_f32` in
/// `shaders/trig.wgsl`.
#[inline]
pub fn sin_cos_f32(x: f32) -> (f32, f32) {
    let k = (x * TWO_OVER_PI).round_ties_even();
    // k P1 and k P2 are exact; the last term is fused (as in the shader).
    let r = (-k).mul_add(P3, (x - k * P1) - k * P2);
    let t = r * r;
    let sp = (r * t).mul_add(t.mul_add(t.mul_add(S3, S2), S1), r);
    let cp = (t * t).mul_add(t.mul_add(t.mul_add(C3, C2), C1), (-0.5f32).mul_add(t, 1.0));
    match (k as i32).rem_euclid(4) {
        0 => (sp, cp),
        1 => (cp, -sp),
        2 => (-sp, -cp),
        _ => (-cp, sp),
    }
}

impl<const M: u8> Add for Wf32<M> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self(self.0 + o.0)
    }
}
impl<const M: u8> Sub for Wf32<M> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self(self.0 - o.0)
    }
}
impl<const M: u8> Mul for Wf32<M> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self(self.0 * o.0)
    }
}
impl<const M: u8> Div for Wf32<M> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        Self(div::<M>(self.0, o.0))
    }
}
impl<const M: u8> Neg for Wf32<M> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
impl<const M: u8> AddAssign for Wf32<M> {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl<const M: u8> SubAssign for Wf32<M> {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl<const M: u8> MulAssign for Wf32<M> {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl<const M: u8> DivAssign for Wf32<M> {
    #[inline]
    fn div_assign(&mut self, o: Self) {
        *self = *self / o;
    }
}

impl<const M: u8> Real for Wf32<M> {
    const ZERO: Self = Self(0.0);
    const ONE: Self = Self(1.0);
    const PI: Self = Self(std::f32::consts::PI);
    const EPSILON: Self = Self(f32::EPSILON);

    #[inline]
    fn from_f64(x: f64) -> Self {
        Self(x as f32)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self.0)
    }
    #[inline]
    fn sqrt(self) -> Self {
        Self(sqrt::<M>(self.0))
    }
    #[inline]
    fn abs(self) -> Self {
        Self(self.0.abs())
    }
    #[inline]
    fn sin(self) -> Self {
        Self(sin_cos_f32(self.0).0)
    }
    #[inline]
    fn cos(self) -> Self {
        Self(sin_cos_f32(self.0).1)
    }
    /// Rust's f32 `atan2`: not used on the GPU paths of task G1.
    #[inline]
    fn atan2(self, x: Self) -> Self {
        Self(self.0.atan2(x.0))
    }
    /// Rust's f32 `ln`: not used on the GPU paths of task G1.
    #[inline]
    fn ln(self) -> Self {
        Self(self.0.ln())
    }
    #[inline]
    fn floor(self) -> Self {
        Self(self.0.floor())
    }
    #[inline]
    fn min(self, o: Self) -> Self {
        Self(self.0.min(o.0))
    }
    #[inline]
    fn max(self, o: Self) -> Self {
        Self(self.0.max(o.0))
    }
    /// Fused (the probe shows that `fma` is fused on the M4).
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::ulp_error;

    fn inputs(n: u32) -> impl Iterator<Item = (f32, f32)> {
        (0..n).map(|i| {
            let w = philox4x32_10([i, 0, 0, 0], [7, 7]);
            let f = |m: u32, e: u32| {
                f32::from_bits((m >> 31) << 31 | (127 - 5 + e % 24) << 23 | (m & 0x7f_ffff))
            };
            (f(w[0], w[1]), f(w[2], w[3]))
        })
    }

    #[test]
    fn division_and_sqrt_models_stay_within_their_bounds() {
        let mut worst = [[0.0f64; 2]; 3];
        let mut not_rn = [0usize; 3];
        let n = 20_000;
        for (a, b) in inputs(n) {
            let q = a as f64 / b as f64;
            let s = (a.abs() as f64).sqrt();
            let qs = [div::<0>(a, b), div::<1>(a, b), div::<2>(a, b)];
            let ss = [sqrt::<0>(a.abs()), sqrt::<1>(a.abs()), sqrt::<2>(a.abs())];
            for m in 0..3 {
                worst[m][0] = worst[m][0].max(ulp_error(qs[m], q));
                worst[m][1] = worst[m][1].max(ulp_error(ss[m], s));
                if m == 1 && qs[m] != q as f32 {
                    not_rn[m] += 1;
                }
            }
        }
        assert!(worst[0][0] <= 0.5 && worst[0][1] <= 0.5, "{worst:?}");
        // M4 model: the probe measured 1.71 and 1.73 ULP.
        assert!(worst[1][0] > 1.2 && worst[1][0] < 2.0, "{worst:?}");
        assert!(worst[1][1] > 1.0 && worst[1][1] < 2.0, "{worst:?}");
        let frac = not_rn[1] as f64 / n as f64;
        assert!(frac > 0.15 && frac < 0.45, "{frac}");
        // WGSL bound: 2.5 ULP.
        assert!(worst[2][0] <= 2.5 && worst[2][1] <= 2.5, "{worst:?}");
    }

    #[test]
    fn own_sine_and_cosine_are_accurate() {
        let (mut es, mut ec) = (0.0f64, 0.0f64);
        for i in 0..200_001 {
            let x = (-std::f64::consts::PI + i as f64 * std::f64::consts::PI / 100_000.0) as f32;
            let (s, c) = sin_cos_f32(x);
            let xd = x as f64;
            es = es.max((s as f64 - xd.sin()).abs());
            ec = ec.max((c as f64 - xd.cos()).abs());
        }
        // About 1 ULP of values up to 1 (2^-24 = 6e-8); far better than the
        // WGSL bound of the built-ins (2^-11 = 4.9e-4).
        assert!(es < 1.2e-7 && ec < 1.2e-7, "sin {es:e}, cos {ec:e}");
        // Small angles (the Kepler drift): relative error of sin.
        let mut rel = 0.0f64;
        for i in 1..10_000 {
            let x = i as f32 * 1e-5;
            let s = sin_cos_f32(x).0 as f64;
            rel = rel.max((s / (x as f64).sin() - 1.0).abs());
        }
        assert!(rel < 1.5e-7, "{rel:e}");
    }
}
