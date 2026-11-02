//! The camera of the page: the view plane, pan and zoom.
//!
//! Plain Rust with no web types, so the unit tests run on the host.
//! Units: km for the world, pixels for the screen. Screen y points down, world y up.
//!
//! The "world" of the camera is the view plane. `ViewTransform` maps the
//! inertial frame (the frame of the integrator) into it: the inertial view is
//! the identity; the rotating view turns by -n0 t; in a moon view the plane is
//! centred on the moon and turns with it.

/// Smallest half-height of the view, km.
pub const MIN_HALF_HEIGHT_KM: f64 = 5.0;
/// Largest half-height of the view, km.
pub const MAX_HALF_HEIGHT_KM: f64 = 1.0e6;

/// Map from the inertial frame to the view plane: `v = R(angle) (p - origin)`,
/// where `R` turns anticlockwise. A turn without a mirror, so the view is
/// always seen from the north (orbits go anticlockwise).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewTransform {
    origin_km: [f64; 2],
    cos: f64,
    sin: f64,
}

impl ViewTransform {
    /// The inertial view: the view plane is the inertial frame.
    pub const IDENTITY: Self = Self {
        origin_km: [0.0, 0.0],
        cos: 1.0,
        sin: 0.0,
    };

    /// Moves `origin_km` (inertial frame, km) to the view origin, then turns
    /// anticlockwise by `angle_rad`.
    pub fn new(origin_km: [f64; 2], angle_rad: f64) -> Self {
        let (sin, cos) = angle_rad.sin_cos();
        Self {
            origin_km,
            cos,
            sin,
        }
    }

    /// The inertial-frame point (km) that this map takes to the view-plane
    /// point `eye_km`: `origin + R(-angle) eye`. The projection shader
    /// computes `R (p - q)`, which is the view position relative to the eye.
    pub fn source_of(&self, eye_km: [f64; 2]) -> [f64; 2] {
        [
            self.origin_km[0] + self.cos * eye_km[0] + self.sin * eye_km[1],
            self.origin_km[1] - self.sin * eye_km[0] + self.cos * eye_km[1],
        ]
    }

    /// Cosine and sine of the turn angle.
    pub fn cos_sin(&self) -> [f64; 2] {
        [self.cos, self.sin]
    }

    /// The view-plane position (km) of the inertial-frame point `p_km`. All in
    /// f64, so the result has the precision of the input.
    pub fn apply(&self, p_km: [f64; 2]) -> [f64; 2] {
        let x = p_km[0] - self.origin_km[0];
        let y = p_km[1] - self.origin_km[1];
        [self.cos * x - self.sin * y, self.sin * x + self.cos * y]
    }
}

/// A 2D camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// View-plane point (km) at the centre of the window.
    pub centre_km: [f64; 2],
    /// Half of the height of the window, in km.
    pub half_height_km: f64,
}

impl Camera {
    /// A camera at the origin that shows the whole ring `a0_km` with some margin.
    pub fn whole_ring(a0_km: f64) -> Self {
        Self {
            centre_km: [0.0, 0.0],
            half_height_km: 1.3 * a0_km,
        }
    }

    /// A camera on a moon at the view origin, `gap_km` from the ring core. The
    /// half-height is two gaps, so the core is half-way to the window edge.
    pub fn moon_view(gap_km: f64) -> Self {
        Self {
            centre_km: [0.0, 0.0],
            half_height_km: (2.0 * gap_km).clamp(MIN_HALF_HEIGHT_KM, MAX_HALF_HEIGHT_KM),
        }
    }

    /// Pixels per km for a window of `height_px`.
    pub fn px_per_km(&self, height_px: f64) -> f64 {
        0.5 * height_px / self.half_height_km
    }

    /// World position (km) under the screen point `(x_px, y_px)`.
    pub fn screen_to_world(&self, x_px: f64, y_px: f64, size_px: [f64; 2]) -> [f64; 2] {
        let s = self.px_per_km(size_px[1]);
        [
            self.centre_km[0] + (x_px - 0.5 * size_px[0]) / s,
            self.centre_km[1] - (y_px - 0.5 * size_px[1]) / s,
        ]
    }

    /// Moves the view by a mouse drag of `(dx_px, dy_px)`: the world follows the cursor.
    pub fn pan_pixels(&mut self, dx_px: f64, dy_px: f64, size_px: [f64; 2]) {
        let s = self.px_per_km(size_px[1]);
        self.centre_km[0] -= dx_px / s;
        self.centre_km[1] += dy_px / s;
    }

    /// Zooms by `factor` (below 1 zooms in) and keeps the world point under the cursor fixed.
    pub fn zoom_about(&mut self, x_px: f64, y_px: f64, factor: f64, size_px: [f64; 2]) {
        let before = self.screen_to_world(x_px, y_px, size_px);
        self.half_height_km =
            (self.half_height_km * factor).clamp(MIN_HALF_HEIGHT_KM, MAX_HALF_HEIGHT_KM);
        let after = self.screen_to_world(x_px, y_px, size_px);
        self.centre_km[0] += before[0] - after[0];
        self.centre_km[1] += before[1] - after[1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: [f64; 2] = [1200.0, 800.0];

    #[test]
    fn identity_transform_changes_nothing() {
        let p = [140_222.43, -3.5];
        assert_eq!(ViewTransform::IDENTITY.apply(p), p);
    }

    /// A quarter turn anticlockwise about (10, 0): the point (11, 0) goes to (0, 1).
    #[test]
    fn transform_moves_then_turns_anticlockwise() {
        let t = ViewTransform::new([10.0, 0.0], std::f64::consts::FRAC_PI_2);
        let v = t.apply([11.0, 0.0]);
        assert!(v[0].abs() < 1e-15 && (v[1] - 1.0).abs() < 1e-15, "{v:?}");
        assert_eq!(t.apply([10.0, 0.0]), [0.0, 0.0]);
    }

    #[test]
    fn moon_view_is_centred_and_clamped() {
        let c = Camera::moon_view(1000.0);
        assert_eq!(c.centre_km, [0.0, 0.0]);
        assert_eq!(c.half_height_km, 2000.0);
        assert_eq!(Camera::moon_view(0.0).half_height_km, MIN_HALF_HEIGHT_KM);
    }

    #[test]
    fn window_centre_maps_to_camera_centre() {
        let c = Camera {
            centre_km: [10.0, -20.0],
            half_height_km: 100.0,
        };
        let w = c.screen_to_world(600.0, 400.0, SIZE);
        assert!((w[0] - 10.0).abs() < 1e-12 && (w[1] + 20.0).abs() < 1e-12);
        // Top of the window is +half_height in world y.
        let top = c.screen_to_world(600.0, 0.0, SIZE);
        assert!((top[1] - (-20.0 + 100.0)).abs() < 1e-9);
    }

    #[test]
    fn zoom_keeps_the_point_under_the_cursor() {
        let mut c = Camera::whole_ring(140222.43);
        let cursor = (900.0, 150.0);
        let before = c.screen_to_world(cursor.0, cursor.1, SIZE);
        for f in [0.5, 0.5, 0.1, 2.0, 0.01] {
            c.zoom_about(cursor.0, cursor.1, f, SIZE);
            let w = c.screen_to_world(cursor.0, cursor.1, SIZE);
            assert!((w[0] - before[0]).abs() < 1e-6 && (w[1] - before[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn zoom_is_clamped() {
        let mut c = Camera::whole_ring(140222.43);
        c.zoom_about(0.0, 0.0, 1e-12, SIZE);
        assert_eq!(c.half_height_km, MIN_HALF_HEIGHT_KM);
        c.zoom_about(0.0, 0.0, 1e12, SIZE);
        assert_eq!(c.half_height_km, MAX_HALF_HEIGHT_KM);
    }

    #[test]
    fn drag_moves_the_world_with_the_cursor() {
        let mut c = Camera {
            centre_km: [0.0, 0.0],
            half_height_km: 400.0,
        };
        let w = c.screen_to_world(300.0, 300.0, SIZE);
        c.pan_pixels(50.0, -20.0, SIZE);
        // The world point that was under (300, 300) is now under (350, 280).
        let moved = c.screen_to_world(350.0, 280.0, SIZE);
        assert!((moved[0] - w[0]).abs() < 1e-9 && (moved[1] - w[1]).abs() < 1e-9);
    }
}
