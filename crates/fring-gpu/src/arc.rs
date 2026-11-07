//! Task D1: the arc wrap in the GPU kernels: the data in the kick table and
//! the Rust twin of the WGSL functions in `shaders/arc.wgsl`. The f64
//! definition is [`fring_model::physics::ArcGeometry::wrap`].
//!
//! # Layout in the kick table (part of the interface of the arc kernels)
//!
//! A kick row has [`crate::wh::ROW`] = 36 f32 words. Words 33 and 34 of row `k`
//! (a kick row, `k < n`) and of the extra row `n` hold the cosine and the sine
//! of the inertial longitude of the arc centre at the time of that row: the
//! kick time `start + (first + k + 1/2) h` for a kick row, and the sync time
//! at the end of the block, `start + (first + n) h`, for row `n`. The extra
//! row `n + 1` holds, from word 0: `cos Δθ` (hi, lo), `sin Δθ` (hi, lo),
//! `cos Δθ/2`, `sin Δθ/2`, and then the cosine and sine of the arc centre at
//! the sync time at the start of the block, `start + first h`.
//!
//! # Wrap sites
//!
//! Site 0 is the start of the block, site `k + 1` is before kick `k`, and site
//! `n + 1` is the end of the block. At each site a particle outside the wedge
//! is rotated by `∓Δθ` (position and velocity, DS). Every particle is inside
//! the arc, to a few ulp, after every block.

use crate::ds::Ds;
use crate::wh::ROW;
use fring_model::physics::{ArcGeometry, MAX_WRAPS};

/// Word of the cosine of the arc centre in a kick row.
pub const PSI_COS: usize = 33;
/// Word of the sine of the arc centre in a kick row.
pub const PSI_SIN: usize = 34;
/// Words per particle of the chaos wrap log in the status buffer: the number
/// of events, then up to `ARC_LOG_WORDS - 1` events `(site << 2) | code`.
pub const ARC_LOG_WORDS: usize = 8;

/// Writes the arc centre angles into words 33 and 34 of the `n` kick rows of
/// `packed` (as made by [`crate::wh::pack_table`]) for the steps `first` to
/// `first + n - 1`.
pub fn set_kick_angles(
    packed: &mut [f32],
    geom: &ArcGeometry,
    start_s: f64,
    h_s: f64,
    first: u64,
    n: usize,
) {
    for k in 0..n {
        let t = start_s + ((first + k as u64) as f64 + 0.5) * h_s;
        let (c, s) = geom.centre_cos_sin(t);
        packed[k * ROW + PSI_COS] = c as f32;
        packed[k * ROW + PSI_SIN] = s as f32;
    }
}

/// The two extra rows after the `n` kick rows (rows `n` and `n + 1`; see the
/// module documentation).
pub fn extra_rows(
    geom: &ArcGeometry,
    start_s: f64,
    h_s: f64,
    first: u64,
    n: usize,
) -> [f32; 2 * ROW] {
    let mut out = [0f32; 2 * ROW];
    let (c, s) = geom.centre_cos_sin(start_s + (first + n as u64) as f64 * h_s);
    out[PSI_COS] = c as f32;
    out[PSI_SIN] = s as f32;
    let (cw, sw) = geom.width_cos_sin();
    let (ch, sh) = geom.half_width_cos_sin();
    let (cd, sd) = (Ds::<0>::split(cw), Ds::<0>::split(sw));
    let r = &mut out[ROW..];
    r[0] = cd.hi;
    r[1] = cd.lo;
    r[2] = sd.hi;
    r[3] = sd.lo;
    r[4] = ch as f32;
    r[5] = sh as f32;
    let (c0, s0) = geom.centre_cos_sin(start_s + first as f64 * h_s);
    r[6] = c0 as f32;
    r[7] = s0 as f32;
    out
}

/// The constants of the extra row `n + 1`, as the kernel reads them.
#[derive(Clone, Copy, Debug)]
pub struct ArcConsts<const M: u8> {
    /// `cos Δθ`.
    pub cw: Ds<M>,
    /// `sin Δθ`.
    pub sw: Ds<M>,
    /// `cos Δθ/2`.
    pub ch: f32,
    /// `sin Δθ/2`.
    pub sh: f32,
}

impl<const M: u8> ArcConsts<M> {
    /// The constants of `geom`, as in [`extra_rows`].
    pub fn new(geom: &ArcGeometry) -> Self {
        let (cw, sw) = geom.width_cos_sin();
        let (ch, sh) = geom.half_width_cos_sin();
        Self {
            cw: Ds::split(cw),
            sw: Ds::split(sw),
            ch: ch as f32,
            sh: sh as f32,
        }
    }
}

/// The result of [`wrap_ds`].
#[derive(Clone, Copy, Debug)]
pub struct WrapDs<const M: u8> {
    /// Position, km.
    pub p: [Ds<M>; 3],
    /// Velocity, km/s.
    pub v: [Ds<M>; 3],
    /// The direction of the first and the second rotation: `-1` for `-Δθ`,
    /// `+1` for `+Δθ`, `0` for none.
    pub dirs: [i32; 2],
    /// False if more than [`MAX_WRAPS`] rotations were needed.
    pub ok: bool,
}

fn rot<const M: u8>(a: &[Ds<M>; 3], c: Ds<M>, s: Ds<M>) -> [Ds<M>; 3] {
    [a[0] * c - a[1] * s, a[0] * s + a[1] * c, a[2]]
}

/// Twin of `arc_wrap` in `shaders/arc.wgsl`: wraps `(p, v)` into the wedge
/// whose centre has cosine and sine `psi`.
pub fn wrap_ds<const M: u8>(
    p: [Ds<M>; 3],
    v: [Ds<M>; 3],
    psi: (f32, f32),
    c: &ArcConsts<M>,
) -> WrapDs<M> {
    let (mut q, mut u, mut dirs) = (p, v, [0i32; 2]);
    for it in 0..=MAX_WRAPS as usize {
        let x = q[0].hi * psi.0 + q[1].hi * psi.1;
        let y = q[1].hi * psi.0 - q[0].hi * psi.1;
        if x * c.sh - y.abs() * c.ch >= 0.0 {
            return WrapDs {
                p: q,
                v: u,
                dirs,
                ok: true,
            };
        }
        if it == MAX_WRAPS as usize {
            break;
        }
        let (dir, s) = if y >= 0.0 { (-1, -c.sw) } else { (1, c.sw) };
        q = rot(&q, c.cw, s);
        u = rot(&u, c.cw, s);
        dirs[it] = dir;
    }
    WrapDs {
        p: q,
        v: u,
        dirs,
        ok: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fring_model::Vec3;

    const M4: u8 = crate::wf32::model::M4;

    fn geom() -> ArcGeometry {
        ArcGeometry::new(10f64.to_radians(), 0.3, 1.1756e-4).unwrap()
    }

    #[test]
    fn the_layout_is_in_the_free_words_of_a_row() {
        // pack_table fills words 0 to 32 of a row; 33 to 35 are free.
        const { assert!(PSI_COS == 33 && PSI_SIN == 34 && PSI_SIN < ROW) };
        let g = geom();
        let mut table = vec![0f32; 3 * ROW];
        set_kick_angles(&mut table[..ROW], &g, 1.0e6, 356.0, 10, 1);
        let t = 1.0e6 + 10.5 * 356.0;
        let (c, s) = g.centre_cos_sin(t);
        assert_eq!(table[PSI_COS], c as f32);
        assert_eq!(table[PSI_SIN], s as f32);
        assert!(table[..33].iter().all(|x| *x == 0.0));
        let ex = extra_rows(&g, 1.0e6, 356.0, 10, 1);
        let (c, s) = g.centre_cos_sin(1.0e6 + 11.0 * 356.0);
        assert_eq!((ex[PSI_COS], ex[PSI_SIN]), (c as f32, s as f32));
        let (cw, sw) = g.width_cos_sin();
        assert!((f64::from(ex[ROW]) + f64::from(ex[ROW + 1]) - cw).abs() < 1e-15);
        assert!((f64::from(ex[ROW + 2]) + f64::from(ex[ROW + 3]) - sw).abs() < 1e-15);
        let (c0, s0) = g.centre_cos_sin(1.0e6 + 10.0 * 356.0);
        assert_eq!((ex[ROW + 6], ex[ROW + 7]), (c0 as f32, s0 as f32));
    }

    /// The DS twin against the f64 wrap, for k = 0, +-1, +-2 and a failure.
    #[test]
    fn the_twin_agrees_with_the_f64_wrap() {
        let g = geom();
        let t = 5.0e8;
        let psi = g.centre_cos_sin(t);
        let psi32 = (psi.0 as f32, psi.1 as f32);
        let consts = ArcConsts::<M4>::new(&g);
        let w = g.width_rad();
        let mid = g.centre_at_rad(t);
        for (off, k) in [
            (0.0, 0),
            (0.45 * w, 0),
            (1.0 * w, -1),
            (-1.0 * w, 1),
            (1.9 * w, -2),
            (-2.1 * w, 2),
            (3.0 * w, 99),
        ] {
            let lon: f64 = mid + off;
            let (p, v) = (
                Vec3::new(140_000.0 * lon.cos(), 140_000.0 * lon.sin(), 3.0),
                Vec3::new(-16.4 * lon.sin(), 16.4 * lon.cos(), 0.001),
            );
            let st = crate::spike::state_from_f64::<M4>(p, v);
            let tw = wrap_ds::<M4>([st[0], st[1], st[2]], [st[3], st[4], st[5]], psi32, &consts);
            let rf = g.wrap(p, v, t);
            if k == 99 {
                assert!(!tw.ok && rf.is_none());
                continue;
            }
            let rf = rf.unwrap();
            assert!(tw.ok);
            assert_eq!(tw.dirs[0] + tw.dirs[1], rf.k);
            assert_eq!(rf.k, k);
            let d = |a: Ds<M4>, b: f64| (f64::from(a.hi) + f64::from(a.lo) - b).abs();
            assert!(d(tw.p[0], rf.pos_km.x) < 1e-8, "x");
            assert!(d(tw.p[1], rf.pos_km.y) < 1e-8, "y");
            assert!(d(tw.v[0], rf.vel_km_s.x) < 1e-11, "vx");
            assert!(d(tw.v[1], rf.vel_km_s.y) < 1e-11, "vy");
        }
    }
}
