//! Co-moving arc domain with azimuthal wrapping (task D1, `docs/PLAN.md`
//! section 7.6).
//!
//! The arc is the wedge of half-width `Δθ/2` about the direction of longitude
//! `centre + n0 t` (inertial, from the x axis). It turns at the mean motion
//! `n0` of the reference orbit, so it is a fixed wedge of the rotating frame
//! (which coincides with the inertial frame at `t = 0`).
//!
//! # The wrap
//!
//! A particle outside the wedge is rotated about the z axis by `±Δθ`, in the
//! inertial frame, with both position **and** velocity. The rotation
//! `R(α)` commutes with the frame rotation, so in the rotating and in the
//! local frame the particle keeps its `x`, `z`, `vx`, `vy`, `vz` exactly and
//! only `y = a0 φ` changes by `±a0 Δθ`.
//!
//! Saturn's field (point mass, J2, J4, J6) is axisymmetric, so `|r|`, `|v|`,
//! the energy `E`, the angular momentum `Lz` and the rotating-frame
//! Hamiltonian `E - n0 Lz` do not change in a wrap. The moons do not wrap
//! (they move in the inertial frame), so with moons the moon term of the
//! energy jumps by the potential difference between the two phases; that is
//! the approximation of PLAN 7.6, measured by task D2.
//!
//! # Use
//!
//! Call [`ArcGeometry::wrap`] after every step, with the time of the state.
//! Positions and velocities are Saturn-centred and inertial (km, km/s); `t_s`
//! is in seconds since the epoch of the rotating frame. After the wrap the
//! rotating-frame longitude of the particle is in `[φ0, φ0 + Δθ]` (closed:
//! a particle exactly on an edge is not moved), with
//! `φ0 = `[`ArcGeometry::rotating_start_rad`]. A cell index for collisions
//! should clamp to the last cell, because rounding can leave a particle a few
//! ulp above the upper edge.

use super::frames::reference_mean_motion;
use super::gravity::SaturnField;
use crate::config::Domain;
use crate::{Error, Real, Result, Vec3};

/// Most rotations of one wrap. A particle that needs more has left the arc by
/// more than `MAX_WRAPS` widths in one step, and [`ArcGeometry::wrap`]
/// returns `None`.
pub const MAX_WRAPS: u32 = 2;

/// The geometry of an arc (angles in radians, rates in rad s⁻¹).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArcGeometry {
    width_rad: f64,
    centre_rad: f64,
    n0_rad_s: f64,
}

/// The result of a wrap: the state in the arc and the number of arc widths
/// that it moved. `k` is signed: the new state is the old state rotated by
/// `k Δθ` (counter-clockwise for `k > 0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wrapped<T: Real> {
    /// Saturn-centred inertial position, km.
    pub pos_km: Vec3<T>,
    /// Saturn-centred inertial velocity, km/s.
    pub vel_km_s: Vec3<T>,
    /// Number of widths added to the longitude (0 if the particle was inside).
    pub k: i32,
}

/// `v` rotated about +z by the angle with cosine `c` and sine `s`.
#[inline]
fn rot<T: Real>(v: Vec3<T>, c: T, s: T) -> Vec3<T> {
    Vec3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

impl ArcGeometry {
    /// An arc of width `width_rad` in `(0, 2π]` with its centre at inertial
    /// longitude `centre_rad` at `t = 0`, turning at `n0_rad_s`.
    ///
    /// # Errors
    /// [`Error::Config`] if an argument is not finite or the width is outside
    /// `(0, 2π]`.
    pub fn new(width_rad: f64, centre_rad: f64, n0_rad_s: f64) -> Result<Self> {
        let ok = width_rad.is_finite()
            && centre_rad.is_finite()
            && n0_rad_s.is_finite()
            && width_rad > 0.0
            && width_rad <= std::f64::consts::TAU;
        if !ok {
            return Err(Error::Config(format!(
                "arc: width {width_rad} rad must be in (0, 2 pi], centre {centre_rad} and \
                 n0 {n0_rad_s} must be finite"
            )));
        }
        Ok(Self {
            width_rad,
            centre_rad,
            n0_rad_s,
        })
    }

    /// The arc of a config domain, or `None` for the full ring. `n0` is the
    /// mean motion of the reference orbit ([`reference_mean_motion`]).
    ///
    /// # Errors
    /// As [`ArcGeometry::new`].
    pub fn from_domain(domain: &Domain, n0_rad_s: f64) -> Result<Option<Self>> {
        match *domain {
            Domain::FullRing => Ok(None),
            Domain::Arc {
                width_deg,
                centre_longitude_deg,
            } => Self::new(
                width_deg.to_radians(),
                centre_longitude_deg.to_radians(),
                n0_rad_s,
            )
            .map(Some),
        }
    }

    /// As [`ArcGeometry::from_domain`], with `n0` from the field and the
    /// reference radius `a0_km`.
    ///
    /// # Errors
    /// As [`ArcGeometry::new`].
    pub fn from_domain_in_field(
        domain: &Domain,
        field: &SaturnField<f64>,
        a0_km: f64,
    ) -> Result<Option<Self>> {
        Self::from_domain(domain, reference_mean_motion(field, a0_km))
    }

    /// Arc width Δθ, rad.
    pub fn width_rad(&self) -> f64 {
        self.width_rad
    }

    /// Inertial longitude of the arc centre at `t = 0`, rad.
    pub fn centre_rad(&self) -> f64 {
        self.centre_rad
    }

    /// Mean motion of the arc, rad/s.
    pub fn n0_rad_s(&self) -> f64 {
        self.n0_rad_s
    }

    /// Arc length `a0 Δθ` on the reference circle, km (the `D` of
    /// `fring-collide/SPEC.md` section 5.1).
    pub fn arc_length_km(&self, a0_km: f64) -> f64 {
        a0_km * self.width_rad
    }

    /// Rotating-frame longitude of the start of the arc,
    /// `φ0 = centre - Δθ/2`, rad (not reduced to a range).
    pub fn rotating_start_rad(&self) -> f64 {
        self.centre_rad - 0.5 * self.width_rad
    }

    /// Inertial longitude of the arc centre at time `t_s`, in `(-π, π]`.
    pub fn centre_at_rad(&self, t_s: f64) -> f64 {
        let tau = std::f64::consts::TAU;
        let a = (self.centre_rad + self.n0_rad_s * t_s).rem_euclid(tau);
        if a > std::f64::consts::PI { a - tau } else { a }
    }

    /// Cosine and sine of the inertial longitude of the arc centre at `t_s`,
    /// computed in f64 (the GPU table holds these).
    pub fn centre_cos_sin(&self, t_s: f64) -> (f64, f64) {
        let a = self.centre_at_rad(t_s);
        (a.cos(), a.sin())
    }

    /// Cosine and sine of the half width, `(cos Δθ/2, sin Δθ/2)`.
    pub fn half_width_cos_sin(&self) -> (f64, f64) {
        let h = 0.5 * self.width_rad;
        (h.cos(), h.sin())
    }

    /// Cosine and sine of the width, `(cos Δθ, sin Δθ)`.
    pub fn width_cos_sin(&self) -> (f64, f64) {
        (self.width_rad.cos(), self.width_rad.sin())
    }

    /// True if the position is outside the wedge at time `t_s`. The test is
    /// `X sin(Δθ/2) - |Y| cos(Δθ/2) < 0` in the frame of the arc centre
    /// (X along the centre, Y across it), which is `|ψ| > Δθ/2` for every
    /// width up to 2π without an `atan2`. Returns the sign of `Y` too.
    #[inline]
    fn outside<T: Real>(pos: Vec3<T>, cp: T, sp: T, ch: T, sh: T) -> Option<bool> {
        let x = pos.x * cp + pos.y * sp;
        let y = pos.y * cp - pos.x * sp;
        // `!(a >= 0)` and not `a < 0`: a NaN counts as outside.
        let inside = x * sh - y.abs() * ch >= T::ZERO;
        if inside { None } else { Some(y >= T::ZERO) }
    }

    /// Puts a Saturn-centred inertial state into the arc at time `t_s` with
    /// any number of rotations (for the start of a run, when the particles
    /// were made all round the ring), then applies [`ArcGeometry::wrap`] for
    /// rounding. Returns `None` if the position is not finite.
    ///
    /// Note: folding a uniform ring distribution this way is not uniform in the
    /// arc when `2π / Δθ` is not a whole number (the first part of the arc is
    /// covered once more). Draw the longitudes inside the arc instead.
    pub fn fold<T: Real>(
        &self,
        pos_km: Vec3<T>,
        vel_km_s: Vec3<T>,
        t_s: f64,
    ) -> Option<Wrapped<T>> {
        let lon = pos_km.y.to_f64().atan2(pos_km.x.to_f64());
        if !lon.is_finite() {
            return None;
        }
        let tau = std::f64::consts::TAU;
        let d = {
            let d = (lon - self.centre_at_rad(t_s)).rem_euclid(tau);
            if d > std::f64::consts::PI { d - tau } else { d }
        };
        let m = (d / self.width_rad).round();
        let a = -m * self.width_rad;
        let (c, s) = (T::from_f64(a.cos()), T::from_f64(a.sin()));
        let w = self.wrap(rot(pos_km, c, s), rot(vel_km_s, c, s), t_s)?;
        Some(Wrapped {
            k: w.k - m as i32,
            ..w
        })
    }

    /// Wraps a Saturn-centred inertial state into the arc at time `t_s` (see
    /// the module documentation). Returns `None` if more than [`MAX_WRAPS`]
    /// rotations would be needed, or the position is not finite.
    pub fn wrap<T: Real>(
        &self,
        pos_km: Vec3<T>,
        vel_km_s: Vec3<T>,
        t_s: f64,
    ) -> Option<Wrapped<T>> {
        let (cp, sp) = self.centre_cos_sin(t_s);
        let (ch, sh) = self.half_width_cos_sin();
        let (cw, sw) = self.width_cos_sin();
        let (cp, sp, ch, sh, cw, sw) = (
            T::from_f64(cp),
            T::from_f64(sp),
            T::from_f64(ch),
            T::from_f64(sh),
            T::from_f64(cw),
            T::from_f64(sw),
        );
        let (mut p, mut v, mut k) = (pos_km, vel_km_s, 0i32);
        for it in 0..=MAX_WRAPS {
            let Some(ccw_side) = Self::outside(p, cp, sp, ch, sh) else {
                return Some(Wrapped {
                    pos_km: p,
                    vel_km_s: v,
                    k,
                });
            };
            if it == MAX_WRAPS {
                return None;
            }
            // Beyond the counter-clockwise edge: rotate by -Δθ, else by +Δθ.
            let s = if ccw_side { -sw } else { sw };
            p = rot(p, cw, s);
            v = rot(v, cw, s);
            k += if ccw_side { -1 } else { 1 };
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::saturn_from_toml_str;
    use crate::physics::frames::{inertial_to_rotating, rotating_to_local};

    const A0: f64 = 140_222.43;

    fn field() -> SaturnField<f64> {
        let sp = saturn_from_toml_str(include_str!("../../../../params/saturn.toml")).unwrap();
        SaturnField::from_params(&sp)
    }

    fn n0() -> f64 {
        reference_mean_motion(&field(), A0)
    }

    /// Position at inertial longitude `lon` and radius `r`, with a circular-ish velocity.
    fn state(lon: f64, r: f64, vr: f64) -> (Vec3<f64>, Vec3<f64>) {
        let (c, s) = (lon.cos(), lon.sin());
        let vc = (37_931_206.0_f64 / r).sqrt();
        (
            Vec3::new(r * c, r * s, 0.3),
            Vec3::new(vr * c - vc * s, vr * s + vc * c, 0.001),
        )
    }

    fn lon(p: Vec3<f64>) -> f64 {
        p.y.atan2(p.x)
    }

    /// `a - b` reduced to `(-π, π]`.
    fn dang(a: f64, b: f64) -> f64 {
        let tau = std::f64::consts::TAU;
        let d = (a - b).rem_euclid(tau);
        if d > std::f64::consts::PI { d - tau } else { d }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    const WIDTHS_DEG: [f64; 8] = [5.0, 7.0, 10.0, 30.0, 120.0, 180.0, 270.0, 359.0];
    const CENTRES_DEG: [f64; 4] = [0.0, -170.0, 175.0, 90.0];
    const TIMES_S: [f64; 3] = [0.0, 12_345.6, 9.0e8];

    #[test]
    fn inside_is_untouched_and_outside_moves_by_one_width() {
        for &w in &WIDTHS_DEG {
            for &c in &CENTRES_DEG {
                for &t in &TIMES_S {
                    let g = ArcGeometry::new(w.to_radians(), c.to_radians(), n0()).unwrap();
                    let psi = g.centre_at_rad(t);
                    let h = 0.5 * g.width_rad();
                    for (off, want_k) in [
                        (0.0, 0),
                        (0.999 * h, 0),
                        (-0.999 * h, 0),
                        (1.001 * h, -1),
                        (-1.001 * h, 1),
                    ] {
                        let (p, v) = state(psi + off, A0, 0.01);
                        let r = g.wrap(p, v, t).unwrap();
                        assert_eq!(r.k, want_k, "w {w} c {c} t {t} off {off}");
                        if want_k == 0 {
                            assert_eq!(r.pos_km, p);
                            assert_eq!(r.vel_km_s, v);
                        } else {
                            let d = dang(lon(r.pos_km), psi);
                            assert!(d.abs() <= h + 1e-12, "w {w} off {off} d {d}");
                            // Moved by exactly k widths.
                            let moved = dang(lon(r.pos_km), lon(p));
                            let want = dang(f64::from(want_k) * g.width_rad(), 0.0);
                            assert!(dang(moved, want).abs() < 1e-12, "moved {moved} want {want}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exactly_on_an_edge_is_inside_to_rounding() {
        let g = ArcGeometry::new(10f64.to_radians(), 0.3, n0()).unwrap();
        for sign in [-1.0, 1.0] {
            let (p, v) = state(0.3 + sign * g.width_rad() * 0.5, A0, 0.0);
            let r = g.wrap(p, v, 0.0).unwrap();
            // Either side of the edge by rounding; the result is in the arc either way.
            assert!(dang(lon(r.pos_km), 0.3).abs() <= 0.5 * g.width_rad() + 1e-12);
            assert!(r.k.abs() <= 1);
        }
    }

    #[test]
    fn full_circle_never_wraps() {
        let g = ArcGeometry::new(std::f64::consts::TAU, 1.0, n0()).unwrap();
        for i in 0..720 {
            let a = (f64::from(i) * 0.5).to_radians() - std::f64::consts::PI;
            let (p, v) = state(a, A0, 0.0);
            assert_eq!(g.wrap(p, v, 5.0e5).unwrap().k, 0, "lon {a}");
        }
    }

    #[test]
    fn wide_arcs_use_the_right_test() {
        // 270 degrees: the empty part is 90 degrees wide, behind the centre.
        let g = ArcGeometry::new(270f64.to_radians(), 0.0, n0()).unwrap();
        let (p, v) = state(std::f64::consts::PI, A0, 0.0); // 180 deg from the centre: outside
        assert_ne!(g.wrap(p, v, 0.0).unwrap().k, 0);
        let (p, v) = state(130f64.to_radians(), A0, 0.0); // inside (< 135)
        assert_eq!(g.wrap(p, v, 0.0).unwrap().k, 0);
        let (p, v) = state(140f64.to_radians(), A0, 0.0); // outside
        assert_eq!(g.wrap(p, v, 0.0).unwrap().k, -1);
    }

    #[test]
    fn two_rotations_work_and_three_fail() {
        let g = ArcGeometry::new(10f64.to_radians(), 0.0, n0()).unwrap();
        let w = g.width_rad();
        // Offset 2.3 widths: 23 deg -> 13 -> 3 deg: k = -2.
        let (p, v) = state(2.3 * w, A0, 0.0);
        let r = g.wrap(p, v, 0.0).unwrap();
        assert_eq!(r.k, -2);
        assert!(lon(r.pos_km).abs() <= 0.5 * w);
        let (p, v) = state(-2.3 * w, A0, 0.0);
        assert_eq!(g.wrap(p, v, 0.0).unwrap().k, 2);
        // Offset 3.2 widths needs 3 rotations.
        let (p, v) = state(3.2 * w, A0, 0.0);
        assert!(g.wrap(p, v, 0.0).is_none());
    }

    #[test]
    fn not_finite_fails() {
        let g = ArcGeometry::new(0.2, 0.0, n0()).unwrap();
        let (_, v) = state(0.0, A0, 0.0);
        assert!(g.wrap(Vec3::new(f64::NAN, 1.0, 0.0), v, 0.0).is_none());
    }

    #[test]
    fn bad_geometry_is_refused() {
        assert!(ArcGeometry::new(0.0, 0.0, 1e-4).is_err());
        assert!(ArcGeometry::new(7.0, 0.0, 1e-4).is_err());
        assert!(ArcGeometry::new(1.0, f64::NAN, 1e-4).is_err());
        assert!(
            ArcGeometry::from_domain(&Domain::FullRing, 1e-4)
                .unwrap()
                .is_none()
        );
    }

    /// 10^4 random wraps: E and Lz do not change (relative < 1e-13), and in the
    /// local frame x, z, vx, vy, vz do not change and y changes by k a0 Δθ.
    #[test]
    fn wrap_preserves_energy_and_the_local_state() {
        let f = field();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let w = 7f64.to_radians();
        let g = ArcGeometry::new(w, -0.4, n0()).unwrap();
        let (mut worst_e, mut worst_l, mut worst_loc, mut worst_y) = (0f64, 0f64, 0f64, 0f64);
        let mut wraps = 0;
        for _ in 0..10_000 {
            let t = rng.next() * 1.0e9; // up to 30 years
            let psi = g.centre_at_rad(t);
            let r = 139_000.0 + 3_000.0 * rng.next();
            let off = (rng.next() - 0.5) * 2.0 * 2.2 * w; // up to 1.1 widths... 2.2 w each side
            let (mut p, mut v) = state(psi + off, r, 0.5 * (rng.next() - 0.5));
            v.z = 0.01 * (rng.next() - 0.5);
            p.z = 5.0 * (rng.next() - 0.5);
            let Some(wr) = g.wrap(p, v, t) else { continue };
            if wr.k == 0 {
                continue;
            }
            wraps += 1;
            let e0 = f.specific_energy(p, v);
            let e1 = f.specific_energy(wr.pos_km, wr.vel_km_s);
            let l0 = p.x * v.y - p.y * v.x;
            let l1 = wr.pos_km.x * wr.vel_km_s.y - wr.pos_km.y * wr.vel_km_s.x;
            worst_e = worst_e.max(((e1 - e0) / e0).abs());
            worst_l = worst_l.max(((l1 - l0) / l0).abs());
            let n0 = g.n0_rad_s();
            let (pr0, vr0) = inertial_to_rotating(p, v, n0, t);
            let (pr1, vr1) = inertial_to_rotating(wr.pos_km, wr.vel_km_s, n0, t);
            let l_0 = rotating_to_local(pr0, vr0, A0);
            let l_1 = rotating_to_local(pr1, vr1, A0);
            for (a, b) in [
                (l_0.x_km, l_1.x_km),
                (l_0.z_km, l_1.z_km),
                (l_0.vx_km_s, l_1.vx_km_s),
                (l_0.vy_km_s, l_1.vy_km_s),
                (l_0.vz_km_s, l_1.vz_km_s),
            ] {
                worst_loc = worst_loc.max((a - b).abs());
            }
            // y changes by k a0 w (mod 2 pi a0).
            let dy = l_1.y_km - l_0.y_km;
            let want = A0 * dang(f64::from(wr.k) * w, 0.0);
            let tau_a0 = std::f64::consts::TAU * A0;
            let diff = (dy - want).rem_euclid(tau_a0);
            worst_y = worst_y.max(diff.min(tau_a0 - diff));
        }
        eprintln!(
            "{wraps} wraps: dE/E {worst_e:.2e}, dLz/Lz {worst_l:.2e}, local {worst_loc:.2e} km, \
             y {worst_y:.2e} km"
        );
        assert!(wraps > 5_000);
        assert!(worst_e < 1e-13, "dE/E {worst_e}");
        assert!(worst_l < 1e-13, "dLz/Lz {worst_l}");
        assert!(worst_loc < 1e-8, "local {worst_loc}");
        assert!(worst_y < 1e-8, "y {worst_y}");
    }

    #[test]
    fn fold_puts_any_longitude_into_the_arc() {
        let g = ArcGeometry::new(7f64.to_radians(), -0.4, n0()).unwrap();
        for i in 0..360 {
            let t = 1.0e8;
            let (p, v) = state(f64::from(i).to_radians() - 3.0, A0, 0.01);
            let r = g.fold(p, v, t).unwrap();
            let d = dang(lon(r.pos_km), g.centre_at_rad(t));
            assert!(d.abs() <= 0.5 * g.width_rad() + 1e-12, "lon {i} d {d}");
            assert!((r.pos_km.norm() - p.norm()).abs() < 1e-9);
            assert!((r.vel_km_s.norm() - v.norm()).abs() < 1e-12);
        }
    }

    #[test]
    fn f32_wrap_agrees_with_f64() {
        let g = ArcGeometry::new(10f64.to_radians(), 0.2, n0()).unwrap();
        let (p, v) = state(g.centre_at_rad(100.0) + 0.12, A0, 0.01);
        let a = g.wrap(p, v, 100.0).unwrap();
        let b = g.wrap(p.cast::<f32>(), v.cast::<f32>(), 100.0).unwrap();
        assert_eq!(a.k, b.k);
        assert!((b.pos_km.x as f64 - a.pos_km.x).abs() < 0.05);
        assert!((b.vel_km_s.y as f64 - a.vel_km_s.y).abs() < 1e-5);
    }
}
