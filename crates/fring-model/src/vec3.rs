//! A small 3-vector, generic over [`Real`].

use crate::Real;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A 3-vector with components of type `T` (`f64` or `f32`).
///
/// The unit is in the name of the variable that holds it, for example
/// `position_km: Vec3<f64>`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3<T> {
    /// x component.
    pub x: T,
    /// y component.
    pub y: T,
    /// z component.
    pub z: T,
}

impl<T: Real> Vec3<T> {
    /// Makes a vector from its components.
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    pub const fn zero() -> Self {
        Self::new(T::ZERO, T::ZERO, T::ZERO)
    }

    /// Dot product.
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product `self × other`.
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Squared length.
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }

    /// Length.
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// Converts each component to another float type (for example f64 to f32).
    pub fn cast<U: Real>(self) -> Vec3<U> {
        Vec3::new(
            U::from_f64(self.x.to_f64()),
            U::from_f64(self.y.to_f64()),
            U::from_f64(self.z.to_f64()),
        )
    }

    /// The components as an array `[x, y, z]`.
    pub fn to_array(self) -> [T; 3] {
        [self.x, self.y, self.z]
    }
}

impl<T: Real> From<[T; 3]> for Vec3<T> {
    fn from(a: [T; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
}

impl<T: Real> Add for Vec3<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl<T: Real> Sub for Vec3<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl<T: Real> Neg for Vec3<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Vector times scalar.
impl<T: Real> Mul<T> for Vec3<T> {
    type Output = Self;
    fn mul(self, s: T) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

/// Vector divided by scalar.
impl<T: Real> Div<T> for Vec3<T> {
    type Output = Self;
    fn div(self, s: T) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s)
    }
}

impl<T: Real> AddAssign for Vec3<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Real> SubAssign for Vec3<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algebra() {
        let a = Vec3::new(1.0_f64, 2.0, 3.0);
        let b = Vec3::new(-2.0_f64, 0.5, 4.0);
        assert_eq!(a + b, Vec3::new(-1.0, 2.5, 7.0));
        assert_eq!(a - b, Vec3::new(3.0, 1.5, -1.0));
        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a / 2.0, Vec3::new(0.5, 1.0, 1.5));
        assert_eq!(a.dot(b), 11.0);
        assert_eq!(Vec3::new(3.0_f64, 4.0, 12.0).norm(), 13.0);
    }

    #[test]
    fn cross_product_is_right_handed() {
        let x = Vec3::new(1.0_f64, 0.0, 0.0);
        let y = Vec3::new(0.0_f64, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn cast_f64_to_f32() {
        let p = Vec3::new(140_220.0_f64, -1.0e-3, 0.25);
        let q: Vec3<f32> = p.cast();
        assert_eq!(q, Vec3::new(140_220.0_f32, -1.0e-3, 0.25));
    }
}
