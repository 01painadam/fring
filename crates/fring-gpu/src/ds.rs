//! [`Ds`]: double-single numbers (hi + lo, two f32, about 48 significant
//! bits), the Rust twin of `shaders/ds.wgsl`.
//!
//! Algorithms (same operations in the same order as the shader):
//! - `two_sum`: `fast_two_sum` (Dekker 1971) after a sort by magnitude;
//!   `two_prod`: exact product error with a fused multiply-add.
//! - Addition: Joldes, Muller and Popescu 2017 (ACM TOMS 44, "Tight and
//!   rigorous error bounds for basic building blocks of double-word
//!   arithmetic"), Algorithm 6 (bound 3u², u = 2⁻²⁴); DS + f32: Algorithm 4.
//! - Multiplication: Algorithm 12 (bound 5u²); DS × f32: Algorithm 9.
//! - Division: Algorithm 17 (bound 15u²), with TwoSum instead of Fast2Sum in line 4,
//!   because the GPU reciprocal is not correctly rounded.
//! - Square root: one Newton (Karp and Markstein) correction of the f32 root.
//! - Sine and cosine of small angles (|x| <= 0.25): Taylor series.
//!
//! **Guard.** On the GPU the shader multiplies some sums by a runtime 1.0
//! (`ds_one`, from a uniform buffer), because Metal's fast math would
//! otherwise re-associate them and lose the error terms (ADR 0006). In
//! Rust a multiplication by 1.0 is exact and Rust never re-associates, so
//! the twin leaves the guard out and gives the same bits for `+`, `-`, `*`.
//! Division and square root use the f32 error model `M` of [`crate::wf32`],
//! so their bits match the GPU only in distribution, not value by value.

use crate::wf32::{div, sqrt};
use fring_model::Real;
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// `(s, e)` with `s = RN(a + b)` and `s + e = a + b` exactly: Fast2Sum
/// after sorting by magnitude, as in the shader (the error-free result is
/// unique, so Knuth's TwoSum would give the same bits).
#[inline]
pub fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let swap = b.abs() > a.abs();
    let (x, y) = if swap { (b, a) } else { (a, b) };
    fast_two_sum(x, y)
}

/// As [`two_sum`], for `|a| >= |b|` (or `a = 0`).
#[inline]
pub fn fast_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let z = s - a;
    (s, b - z)
}

/// `(p, e)` with `p = RN(a b)` and `p + e = a b` exactly (fused `mul_add`).
#[inline]
pub fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

// 1/n! rounded once from f64 (the shader has the same f32 values).
const INV7: f32 = (1.0f64 / 5_040.0) as f32;
const INV8: f32 = (1.0f64 / 40_320.0) as f32;
const INV9: f32 = (1.0f64 / 362_880.0) as f32;
const INV10: f32 = (1.0f64 / 3_628_800.0) as f32;
const INV11: f32 = (1.0f64 / 39_916_800.0) as f32;
const INV12: f32 = (1.0f64 / 479_001_600.0) as f32;
const INV13: f32 = (1.0f64 / 6_227_020_800.0) as f32;
const INV14: f32 = (1.0f64 / 87_178_291_200.0) as f32;

/// A double-single number `hi + lo` with `|lo| <= ulp(hi)/2`, with the f32
/// error model `M` for the division and square root seeds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ds<const M: u8> {
    /// Leading part: the value rounded to f32.
    pub hi: f32,
    /// Trailing part.
    pub lo: f32,
}

impl<const M: u8> Ds<M> {
    /// The DS number of an f32 (exact).
    #[inline]
    pub const fn from_f32(x: f32) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// The nearest DS number of an f64 (relative error at most 2⁻⁴⁸).
    #[inline]
    pub const fn split(x: f64) -> Self {
        let hi = x as f32;
        Self {
            hi,
            lo: (x - hi as f64) as f32,
        }
    }

    /// DS + f32 (Algorithm 4).
    #[inline]
    pub fn add_f(self, b: f32) -> Self {
        let (sh, sl) = two_sum(self.hi, b);
        let v = self.lo + sl;
        let (hi, lo) = fast_two_sum(sh, v);
        Self { hi, lo }
    }

    /// DS × f32 (Algorithm 9).
    #[inline]
    pub fn mul_f(self, b: f32) -> Self {
        let (ch, cl1) = two_prod(self.hi, b);
        let cl3 = self.lo.mul_add(b, cl1);
        let (hi, lo) = fast_two_sum(ch, cl3);
        Self { hi, lo }
    }

    /// Sine, cosine and 1 - cosine of a small angle (|x| <= 0.25), Taylor
    /// series to x¹³ (truncation below 10⁻¹⁷ relative). The terms from x⁷
    /// on are below 2⁻²² of the result and are summed in f32. Twin of
    /// `ds_sin_cos_small` in `shaders/ds.wgsl`.
    #[inline]
    pub fn sin_cos_small(self) -> (Self, Self, Self) {
        let t = self * self;
        let th = t.hi;
        // sin x = x (1 - t/6 + t²/120 - t³ P(t)), P = 1/5040 - t/362880 + ...
        let ps = th
            .mul_add(th.mul_add(-INV13, INV11), -INV9)
            .mul_add(th, INV7);
        let s_in = (t.mul_f(-ps) + Self::split(1.0 / 120.0)) * t + Self::split(-1.0 / 6.0);
        let sin = self + self * t * s_in;
        // 1 - cos x = t/2 - t²/24 + t³/720 - t⁴ Q(t), Q = 1/40320 - t/3628800 + ...
        let pc = th
            .mul_add(th.mul_add(-INV14, INV12), -INV10)
            .mul_add(th, INV8);
        let c_in = (t.mul_f(-pc) + Self::split(1.0 / 720.0)) * t + Self::split(-1.0 / 24.0);
        let omc = t * (Self::from_f32(0.5) + t * c_in);
        let cos = Self::ONE - omc;
        (sin, cos, omc)
    }
}

impl<const M: u8> Add for Ds<M> {
    type Output = Self;
    /// Algorithm 6 (AccurateDWPlusDW).
    #[inline]
    fn add(self, b: Self) -> Self {
        let (sh, sl) = two_sum(self.hi, b.hi);
        let (th, tl) = two_sum(self.lo, b.lo);
        let c = sl + th;
        let (vh, vl) = fast_two_sum(sh, c);
        let w = tl + vl;
        let (hi, lo) = fast_two_sum(vh, w);
        Self { hi, lo }
    }
}

impl<const M: u8> Neg for Ds<M> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl<const M: u8> Sub for Ds<M> {
    type Output = Self;
    #[inline]
    fn sub(self, b: Self) -> Self {
        self + (-b)
    }
}

impl<const M: u8> Mul for Ds<M> {
    type Output = Self;
    /// Algorithm 12 (DWTimesDW3).
    #[inline]
    fn mul(self, b: Self) -> Self {
        let (ch, cl1) = two_prod(self.hi, b.hi);
        let tl0 = self.lo * b.lo;
        let tl1 = self.hi.mul_add(b.lo, tl0);
        let cl2 = self.lo.mul_add(b.hi, tl1);
        let cl3 = cl1 + cl2;
        let (hi, lo) = fast_two_sum(ch, cl3);
        Self { hi, lo }
    }
}

impl<const M: u8> Div for Ds<M> {
    type Output = Self;
    /// Algorithm 17 (DWDivDW2), with the reciprocal seed of model `M`.
    #[inline]
    fn div(self, b: Self) -> Self {
        let th = div::<M>(1.0, b.hi);
        let rh = (-b.hi).mul_add(th, 1.0);
        let rl = -(b.lo * th);
        let (eh, el) = two_sum(rh, rl);
        let d = Self { hi: eh, lo: el }.mul_f(th);
        let m = d.add_f(th);
        self * m
    }
}

macro_rules! assign {
    ($tr:ident, $f:ident, $op:tt) => {
        impl<const M: u8> $tr for Ds<M> {
            #[inline]
            fn $f(&mut self, o: Self) {
                *self = *self $op o;
            }
        }
    };
}
assign!(AddAssign, add_assign, +);
assign!(SubAssign, sub_assign, -);
assign!(MulAssign, mul_assign, *);
assign!(DivAssign, div_assign, /);

impl<const M: u8> PartialOrd for Ds<M> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        match self.hi.partial_cmp(&o.hi)? {
            Ordering::Equal => self.lo.partial_cmp(&o.lo),
            ord => Some(ord),
        }
    }
}

impl<const M: u8> Real for Ds<M> {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    const PI: Self = Self::split(std::f64::consts::PI);
    /// 2⁻⁴⁸ = u² (u = 2⁻²⁴), the relative precision of DS arithmetic.
    const EPSILON: Self = Self {
        hi: 3.552_713_7e-15,
        lo: 0.0,
    };

    #[inline]
    fn from_f64(x: f64) -> Self {
        Self::split(x)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }
    /// One Newton correction of the f32 root (model `M`):
    /// `s + (x - s²) / (2 s)`. Twin of `ds_sqrt`.
    #[inline]
    fn sqrt(self) -> Self {
        if self.hi <= 0.0 {
            return Self::ZERO;
        }
        let s = sqrt::<M>(self.hi);
        let (ph, pl) = two_prod(s, s);
        let e = ((self.hi - ph) - pl) + self.lo;
        let c = div::<M>(e, 2.0 * s);
        let (hi, lo) = fast_two_sum(s, c);
        Self { hi, lo }
    }
    #[inline]
    fn abs(self) -> Self {
        if self.hi < 0.0 { -self } else { self }
    }
    /// Taylor series for |x| <= 0.25 (the Kepler drift at 150 steps per
    /// orbit has |ΔE| <= 0.05). Larger angles use f64 on the host; the shader
    /// has no such branch.
    #[inline]
    fn sin(self) -> Self {
        if self.hi.abs() <= 0.25 {
            self.sin_cos_small().0
        } else {
            Self::split(self.to_f64().sin())
        }
    }
    /// As [`Ds::sin`].
    #[inline]
    fn cos(self) -> Self {
        if self.hi.abs() <= 0.25 {
            self.sin_cos_small().1
        } else {
            Self::split(self.to_f64().cos())
        }
    }
    /// Through f64 (host only; not on the GPU paths of task G1).
    fn atan2(self, x: Self) -> Self {
        Self::split(self.to_f64().atan2(x.to_f64()))
    }
    /// Through f64 (host only; not on the GPU paths of task G1).
    fn ln(self) -> Self {
        Self::split(self.to_f64().ln())
    }
    fn floor(self) -> Self {
        Self::split(self.to_f64().floor())
    }
    fn min(self, o: Self) -> Self {
        if o < self { o } else { self }
    }
    fn max(self, o: Self) -> Self {
        if o > self { o } else { self }
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wf32::model;
    use fring_model::rng::philox4x32_10;

    type D = Ds<{ model::M4 }>;

    /// Random f64 values with exponents in [-5, 18], as in the GPU probe.
    fn values(n: u32, seed: u32) -> impl Iterator<Item = f64> {
        (0..n).map(move |i| {
            let w = philox4x32_10([i, seed, 0, 0], [3, 9]);
            let m = 1.0 + (w[0] as f64 + w[1] as f64 / 4_294_967_296.0) / 4_294_967_296.0;
            let e = -5 + (w[2] % 24) as i32;
            let s = if w[3] & 1 == 1 { -1.0 } else { 1.0 };
            s * m * 2f64.powi(e)
        })
    }

    #[test]
    fn error_free_transforms_are_exact() {
        for (a, b) in values(50_000, 1).zip(values(50_000, 2)) {
            let (a, b) = (a as f32, b as f32);
            let exact_sum = a as f64 + b as f64;
            let (s, e) = two_sum(a, b);
            assert_eq!(s as f64 + e as f64, exact_sum);
            let (big, small) = if a.abs() >= b.abs() { (a, b) } else { (b, a) };
            let (s, e) = fast_two_sum(big, small);
            assert_eq!(s as f64 + e as f64, exact_sum);
            let (p, e) = two_prod(a, b);
            assert_eq!(p as f64 + e as f64, a as f64 * b as f64);
        }
    }

    #[test]
    fn ds_arithmetic_meets_its_bounds() {
        let u2 = 2f64.powi(-48);
        let mut worst = [0.0f64; 4];
        for (x, y) in values(50_000, 3).zip(values(50_000, 4)) {
            let (a, b) = (D::split(x), D::split(y));
            let (xa, yb) = (a.to_f64(), b.to_f64());
            let rel = |got: D, exact: f64| (got.to_f64() - exact).abs() / exact.abs() / u2;
            if xa + yb != 0.0 {
                worst[0] = worst[0].max(rel(a + b, xa + yb));
            }
            worst[1] = worst[1].max(rel(a * b, xa * yb));
            worst[2] = worst[2].max(rel(a / b, xa / yb));
            worst[3] = worst[3].max(rel(a.abs().sqrt(), xa.abs().sqrt()));
        }
        // In units of u² = 2^-48 (u = 2^-24). Bounds of Joldes et al. 2017:
        // add 3u², mul 5u², div 15u² (with a correctly rounded reciprocal).
        assert!(worst[0] < 3.0, "add {worst:?}");
        assert!(worst[1] < 5.0, "mul {worst:?}");
        assert!(worst[2] < 15.0, "div {worst:?}");
        assert!(worst[3] < 10.0, "sqrt {worst:?}");
    }

    #[test]
    fn small_angle_sine_and_cosine() {
        let mut worst = [0.0f64; 3];
        for i in 1..=2_000 {
            let x = i as f64 * 1.25e-4 - 0.125;
            let (s, c, omc) = D::split(x).sin_cos_small();
            let xd = D::split(x).to_f64();
            worst[0] = worst[0].max((s.to_f64() / xd.sin() - 1.0).abs());
            worst[1] = worst[1].max((c.to_f64() - xd.cos()).abs());
            let omc_exact = 2.0 * (0.5 * xd).sin().powi(2);
            if omc_exact > 0.0 {
                worst[2] = worst[2].max((omc.to_f64() / omc_exact - 1.0).abs());
            }
        }
        let u2 = 2f64.powi(-48);
        assert!(worst[0] < 4.0 * u2, "{worst:?}");
        assert!(worst[1] < 2.0 * u2, "{worst:?}");
        assert!(worst[2] < 6.0 * u2, "{worst:?}");
    }
}
