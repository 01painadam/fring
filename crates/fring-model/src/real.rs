//! The float trait [`Real`], for code that must run in both f64 and f32.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A real number type: `f64` for the CPU reference, `f32` to mirror the GPU.
///
/// Force functions and analytic formulas in `fring-model` are generic over
/// `Real`. The same Rust code then runs in f64 (the reference) and in f32 (to
/// compare with the WGSL shaders). A double-single type for task G1 can also
/// implement this trait.
///
/// The trait is small on purpose. Add a method only when a physics function
/// needs it, and only if WGSL has the same function.
pub trait Real:
    Copy
    + Debug
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Send
    + Sync
    + 'static
{
    /// 0.
    const ZERO: Self;
    /// 1.
    const ONE: Self;
    /// π, rounded to this type.
    const PI: Self;
    /// Machine epsilon: the difference between 1 and the next larger value.
    const EPSILON: Self;

    /// Converts from f64, rounding to the nearest value.
    fn from_f64(x: f64) -> Self;
    /// Converts to f64. This is exact for f32 and f64.
    fn to_f64(self) -> f64;

    /// Square root.
    fn sqrt(self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Sine of an angle in radians.
    fn sin(self) -> Self;
    /// Cosine of an angle in radians.
    fn cos(self) -> Self;
    /// Four-quadrant arctangent of `self / x`, in radians, in [-π, π].
    fn atan2(self, x: Self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Largest integer value not greater than `self`.
    fn floor(self) -> Self;
    /// Smaller of two values.
    fn min(self, other: Self) -> Self;
    /// Larger of two values.
    fn max(self, other: Self) -> Self;
    /// `self * a + b` with one rounding (fused multiply-add, WGSL `fma`).
    fn mul_add(self, a: Self, b: Self) -> Self;
}

macro_rules! impl_real {
    ($t:ident) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const PI: Self = std::$t::consts::PI;
            const EPSILON: Self = $t::EPSILON;

            #[inline]
            fn from_f64(x: f64) -> Self {
                // `as` rounds to the nearest value (ties to even).
                x as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                f64::from(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                $t::sqrt(self)
            }
            #[inline]
            fn abs(self) -> Self {
                $t::abs(self)
            }
            #[inline]
            fn sin(self) -> Self {
                $t::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                $t::cos(self)
            }
            #[inline]
            fn atan2(self, x: Self) -> Self {
                $t::atan2(self, x)
            }
            #[inline]
            fn ln(self) -> Self {
                $t::ln(self)
            }
            #[inline]
            fn floor(self) -> Self {
                $t::floor(self)
            }
            #[inline]
            fn min(self, other: Self) -> Self {
                $t::min(self, other)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                $t::max(self, other)
            }
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                $t::mul_add(self, a, b)
            }
        }
    };
}

impl_real!(f32);
impl_real!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    /// A generic function, as force functions will be.
    fn hypot<T: Real>(a: T, b: T) -> T {
        (a * a + b * b).sqrt()
    }

    #[test]
    fn same_generic_code_in_f32_and_f64() {
        assert_eq!(hypot(3.0_f64, 4.0), 5.0);
        assert_eq!(hypot(3.0_f32, 4.0), 5.0);
    }

    #[test]
    fn from_f64_rounds_to_nearest_f32() {
        // 1 + 2^-24 is half-way between two f32 values; ties go to even (1.0).
        let x = 1.0 + 2f64.powi(-24);
        assert_eq!(<f32 as Real>::from_f64(x), 1.0);
        let y = 1.0 + 3.0 * 2f64.powi(-24);
        assert_eq!(<f32 as Real>::from_f64(y), 1.0 + 2f32.powi(-22));
    }

    #[test]
    fn constants() {
        assert_eq!(<f32 as Real>::EPSILON, 2f32.powi(-23));
        assert_eq!(<f64 as Real>::EPSILON, 2f64.powi(-52));
        assert_eq!(<f32 as Real>::PI.to_f64(), std::f32::consts::PI as f64);
    }
}
