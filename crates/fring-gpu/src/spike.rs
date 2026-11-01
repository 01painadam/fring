//! The precision options of task G1 (ADR 0006) as Rust twins of the WGSL
//! kernels in `shaders/spike.wgsl`.
//!
//! Every option runs the Wisdom-Holman step of `fring_model::physics`
//! (ADR 0003: half drift, then kick and drift, with the half drifts of
//! neighbouring steps merged) with the moons of a `KickTable` (ADR 0004).
//! They differ only in the number format:
//!
//! - **A**: f32 throughout (`wh_steps_moons::<Wf32>`).
//! - **B**: state in double-single (DS); drift and kick in f32 from f32 copies
//!   of the state; the f32 increments are added to the DS state.
//! - **B+**: as B, but the drift scalars that cancel (1/a, a, n, e cos E₀,
//!   e sin E₀, the mean anomaly n dt) are DS; the kick as in D.
//! - **C**: DS throughout (`wh_steps_moons::<Ds>`).
//! - **D**: DS state and DS drift, except the Kepler equation, which is
//!   solved in f32 and then corrected by one DS Newton step; the kick has
//!   f32 magnitudes applied along the DS position, and DS moon separations.
//!
//! The state of every option is held as six [`Ds`] values (km, km/s); for
//! option A only the `hi` parts are used.

use crate::ds::Ds;
use crate::wf32::{Wf32, sin_cos_f32};
use fring_model::physics::{KickTable, SaturnField, kepler_drift, wh_steps_moons};
use fring_model::{MoonEphemeris, Real, SaturnParams, Vec3};

/// A precision option (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Opt {
    /// f32 throughout.
    A,
    /// DS state, f32 increments.
    B,
    /// DS state, DS drift scalars, f32 Kepler solve and coefficients.
    BPlus,
    /// DS throughout.
    C,
    /// DS drift with an f32 Kepler solve and one DS Newton step; f32 kick
    /// magnitudes along the DS position.
    D,
}

impl Opt {
    /// All options, cheapest first (by the expected cost).
    pub const ALL: [Opt; 5] = [Opt::A, Opt::B, Opt::BPlus, Opt::D, Opt::C];

    /// Short name for tables and CSV files.
    pub fn name(self) -> &'static str {
        match self {
            Opt::A => "a",
            Opt::B => "b",
            Opt::BPlus => "b+",
            Opt::C => "c",
            Opt::D => "d",
        }
    }

    /// The entry point of the option in `shaders/spike.wgsl`.
    pub fn entry(self) -> &'static str {
        match self {
            Opt::A => "step_a",
            Opt::B => "step_b",
            Opt::BPlus => "step_bp",
            Opt::C => "step_c",
            Opt::D => "step_d",
        }
    }
}

/// The state of one particle: x, y, z (km), vx, vy, vz (km/s), each DS.
pub type State<const M: u8> = [Ds<M>; 6];

/// Run constants in every number format.
#[derive(Clone, Debug)]
pub struct Consts<const M: u8> {
    /// Saturn's field in f32.
    pub field_w: SaturnField<Wf32<M>>,
    /// Saturn's field in DS.
    pub field_d: SaturnField<Ds<M>>,
    /// 1/GM in DS (s² km⁻³).
    pub inv_gm_d: Ds<M>,
    /// GM of the moons in f32 (km³ s⁻²).
    pub gm_moons_w: Vec<Wf32<M>>,
    /// The step in seconds. Must be exact in f32 (ADR 0006).
    pub h_s: f64,
}

impl<const M: u8> Consts<M> {
    /// Constants for Saturn `p`, the moons of `eph` and the step `h_s`.
    ///
    /// # Panics
    /// If `h_s` is not exactly representable in f32.
    pub fn new(p: &SaturnParams, eph: &MoonEphemeris, h_s: f64) -> Self {
        assert_eq!(h_s as f32 as f64, h_s, "the step must be exact in f32");
        Self {
            field_w: SaturnField::from_params(p),
            field_d: SaturnField::from_params(p),
            inv_gm_d: Ds::split(1.0 / p.gm_km3_s2),
            gm_moons_w: eph
                .tracks
                .iter()
                .map(|t| Wf32::from_f64(t.gm_km3_s2))
                .collect(),
            h_s,
        }
    }
}

/// The moon tables of one block of steps in f32 and in DS, filled from the
/// same ephemeris (moon positions rounded from f64; indirect term computed
/// in each format, as `KickTable::fill` does).
#[derive(Clone, Debug)]
pub struct Tables<const M: u8> {
    /// f32 table (options A and B, and the indirect term of B+ and D).
    pub w: KickTable<Wf32<M>>,
    /// DS table (moon positions of B+, C and D; indirect term of C).
    pub d: KickTable<Ds<M>>,
}

impl<const M: u8> Tables<M> {
    /// Empty tables for the moons of `eph`.
    pub fn new(eph: &MoonEphemeris) -> Self {
        Self {
            w: KickTable::for_ephemeris(eph),
            d: KickTable::for_ephemeris(eph),
        }
    }

    /// Fills both tables for `n` steps from step `first` (see `KickTable::fill`).
    pub fn fill(
        &mut self,
        eph: &MoonEphemeris,
        start_s: f64,
        h_s: f64,
        first: u64,
        n: usize,
    ) -> fring_model::Result<()> {
        self.w.fill(eph, start_s, h_s, first, n)?;
        self.d.fill(eph, start_s, h_s, first, n)
    }

    /// The tables packed for the GPU: 24 f32 per step. Words 0 to 11: moon
    /// positions as DS (moon j, component c: `6 j + 2 c` hi, `+1` lo); 12 to
    /// 17: indirect term as DS; 18 to 20: indirect term in f32; 21 to 23: 0.
    pub fn pack(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(24 * self.d.len());
        for k in 0..self.d.len() {
            let mut w = [0f32; 24];
            for (j, p) in self.d.moon_positions_km(k).iter().enumerate().take(2) {
                for (c, v) in p.to_array().iter().enumerate() {
                    w[6 * j + 2 * c] = v.hi;
                    w[6 * j + 2 * c + 1] = v.lo;
                }
            }
            let ind = self.d.indirect_km_s2(k).to_array();
            for c in 0..3 {
                w[12 + 2 * c] = ind[c].hi;
                w[12 + 2 * c + 1] = ind[c].lo;
            }
            let indw = self.w.indirect_km_s2(k).to_array();
            for c in 0..3 {
                w[18 + c] = indw[c].0;
            }
            out.extend_from_slice(&w);
        }
        out
    }
}

/// The state of an f64 position and velocity in DS.
pub fn state_from_f64<const M: u8>(p: Vec3<f64>, v: Vec3<f64>) -> State<M> {
    [p.x, p.y, p.z, v.x, v.y, v.z].map(Ds::split)
}

/// The state as f64 position and velocity (`hi + lo`).
pub fn state_to_f64<const M: u8>(s: &State<M>) -> (Vec3<f64>, Vec3<f64>) {
    let f = s.map(|d| d.to_f64());
    (Vec3::new(f[0], f[1], f[2]), Vec3::new(f[3], f[4], f[5]))
}

fn split_state<const M: u8>(s: &State<M>) -> (Vec3<Ds<M>>, Vec3<Ds<M>>) {
    (Vec3::new(s[0], s[1], s[2]), Vec3::new(s[3], s[4], s[5]))
}

fn join_state<const M: u8>(p: Vec3<Ds<M>>, v: Vec3<Ds<M>>) -> State<M> {
    [p.x, p.y, p.z, v.x, v.y, v.z]
}

fn hi<const M: u8>(v: Vec3<Ds<M>>) -> Vec3<Wf32<M>> {
    Vec3::new(Wf32(v.x.hi), Wf32(v.y.hi), Wf32(v.z.hi))
}

/// `v + w` for a DS vector and an f32 vector.
fn add_wv<const M: u8>(v: Vec3<Ds<M>>, w: Vec3<Wf32<M>>) -> Vec3<Ds<M>> {
    Vec3::new(v.x.add_f(w.x.0), v.y.add_f(w.y.0), v.z.add_f(w.z.0))
}

/// `v * s` for a DS vector and an f32 scalar.
fn mul_vf<const M: u8>(v: Vec3<Ds<M>>, s: f32) -> Vec3<Ds<M>> {
    Vec3::new(v.x.mul_f(s), v.y.mul_f(s), v.z.mul_f(s))
}

/// Advances one particle by the `tab.d.len()` steps of a block with option
/// `opt` (h from `c`). Returns `None` and leaves the state unchanged if an
/// orbit is not bound.
pub fn advance<const M: u8>(
    opt: Opt,
    c: &Consts<M>,
    tab: &Tables<M>,
    s: &mut State<M>,
) -> Option<()> {
    match opt {
        Opt::A => {
            let (p, v) = split_state(s);
            let (mut p, mut v) = (hi(p), hi(v));
            wh_steps_moons(&c.field_w, &tab.w, &mut p, &mut v, Wf32::from_f64(c.h_s))?;
            let d = |x: Wf32<M>| Ds::from_f32(x.0);
            *s = [p.x, p.y, p.z, v.x, v.y, v.z].map(d);
            Some(())
        }
        Opt::C => {
            let (mut p, mut v) = split_state(s);
            wh_steps_moons(&c.field_d, &tab.d, &mut p, &mut v, Ds::split(c.h_s))?;
            *s = join_state(p, v);
            Some(())
        }
        Opt::B | Opt::BPlus | Opt::D => {
            let n = tab.d.len();
            if n == 0 {
                return Some(());
            }
            let h = Ds::<M>::split(c.h_s);
            let half = Ds::<M>::split(0.5 * c.h_s);
            let (mut p, mut v) = split_state(s);
            (p, v) = drift(opt, c, p, v, half)?;
            for k in 0..n {
                v = kick(opt, c, tab, k, p, v);
                let dt = if k + 1 == n { half } else { h };
                (p, v) = drift(opt, c, p, v, dt)?;
            }
            *s = join_state(p, v);
            Some(())
        }
    }
}

fn drift<const M: u8>(
    opt: Opt,
    c: &Consts<M>,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
    dt: Ds<M>,
) -> Option<(Vec3<Ds<M>>, Vec3<Ds<M>>)> {
    match opt {
        Opt::B => drift_b(c, p, v, Wf32(dt.hi)),
        Opt::BPlus => drift_bplus(c, p, v, dt),
        _ => drift_d(c, p, v, dt),
    }
}

fn kick<const M: u8>(
    opt: Opt,
    c: &Consts<M>,
    tab: &Tables<M>,
    k: usize,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
) -> Vec3<Ds<M>> {
    match opt {
        Opt::B => kick_b(c, tab, k, p, v),
        _ => kick_dir(c, tab, k, p, v),
    }
}

/// Option B: the f32 Kepler drift of the f32 copies, as increments added to
/// the DS state. The increments use the same formulas as
/// `fring_model::physics::kepler_drift` with f − 1 and ġ − 1 written out.
fn drift_b<const M: u8>(
    c: &Consts<M>,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
    dt: Wf32<M>,
) -> Option<(Vec3<Ds<M>>, Vec3<Ds<M>>)> {
    let (pw, vw) = (hi(p), hi(v));
    let gm = c.field_w.gm_km3_s2;
    let one = Wf32::<M>::ONE;
    let two = Wf32::<M>(2.0);
    let r0 = pw.norm();
    let v2 = vw.norm_squared();
    let alpha = two / r0 - v2 / gm;
    if !(alpha.0.is_finite() && alpha.0 > 0.0) {
        return None;
    }
    let a = one / alpha;
    let n = (gm / (a * a * a)).sqrt();
    let sqrt_gma = (gm * a).sqrt();
    let ecos0 = one - r0 / a;
    let esin0 = pw.dot(vw) / sqrt_gma;
    let m = n * dt;
    let x = newton_f32(ecos0, esin0, m);
    let (s, cs) = sin_cos_f32(x.0);
    let (s, cs) = (Wf32::<M>(s), Wf32::<M>(cs));
    let sh = Wf32::<M>(sin_cos_f32(0.5 * x.0).0);
    let omc = two * sh * sh;
    let r = a * (one - ecos0 * cs + esin0 * s);
    let fm1 = -((a / r0) * omc);
    let g = dt - (x - s) / n;
    let fdot = -(sqrt_gma / (r * r0)) * s;
    let gm1 = -((a / r) * omc);
    let dp = Vec3::new(
        fm1 * pw.x + g * vw.x,
        fm1 * pw.y + g * vw.y,
        fm1 * pw.z + g * vw.z,
    );
    let dv = Vec3::new(
        fdot * pw.x + gm1 * vw.x,
        fdot * pw.y + gm1 * vw.y,
        fdot * pw.z + gm1 * vw.z,
    );
    Some((add_wv(p, dp), add_wv(v, dv)))
}

/// Newton's method for x = ΔE in f32, as in `kepler_drift` (start at m,
/// stop when the step is below 4 ε max(1, |x|), at most 40 iterations).
fn newton_f32<const M: u8>(ecos0: Wf32<M>, esin0: Wf32<M>, m: Wf32<M>) -> Wf32<M> {
    let one = Wf32::<M>::ONE;
    let mut x = m;
    for _ in 0..40 {
        let (s, c) = sin_cos_f32(x.0);
        let (s, c) = (Wf32(s), Wf32(c));
        let f = x - ecos0 * s + esin0 * (one - c) - m;
        let fp = one - ecos0 * c + esin0 * s;
        let dx = f / fp;
        x -= dx;
        if dx.0.abs() <= 4.0 * f32::EPSILON * x.0.abs().max(1.0) {
            break;
        }
    }
    x
}

/// The DS scalars of a Kepler drift: 1/r0, α = 1/a, a, √(GM a), n, e cos E₀,
/// e sin E₀ and m = n dt. Uses 1/GM to save divisions:
/// α = 2/r0 − v²/GM, n = √(GM a) α², e sin E₀ = (r·v) n a / GM.
struct DsScalars<const M: u8> {
    r0: Ds<M>,
    ir0: Ds<M>,
    alpha: Ds<M>,
    a: Ds<M>,
    sqrt_gma: Ds<M>,
    n: Ds<M>,
    ecos0: Ds<M>,
    esin0: Ds<M>,
    m: Ds<M>,
}

fn ds_scalars<const M: u8>(
    c: &Consts<M>,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
    dt: Ds<M>,
) -> Option<DsScalars<M>> {
    let one = Ds::<M>::ONE;
    let r0 = p.norm_squared().sqrt();
    let ir0 = one / r0;
    let alpha = ir0.mul_f(2.0) - v.norm_squared() * c.inv_gm_d;
    if !(alpha.hi.is_finite() && alpha.hi > 0.0) {
        return None;
    }
    let a = one / alpha;
    let sqrt_gma = (c.field_d.gm_km3_s2 * a).sqrt();
    let n = sqrt_gma * alpha * alpha;
    let ecos0 = one - r0 * alpha;
    let esin0 = p.dot(v) * n * a * c.inv_gm_d;
    let m = n * dt;
    Some(DsScalars {
        r0,
        ir0,
        alpha,
        a,
        sqrt_gma,
        n,
        ecos0,
        esin0,
        m,
    })
}

/// Option B+: DS scalars, f32 Kepler solve and coefficients, DS update.
fn drift_bplus<const M: u8>(
    c: &Consts<M>,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
    dt: Ds<M>,
) -> Option<(Vec3<Ds<M>>, Vec3<Ds<M>>)> {
    let k = ds_scalars(c, p, v, dt)?;
    let w = |d: Ds<M>| Wf32::<M>(d.hi);
    let (a, r0, n, sqrt_gma) = (w(k.a), w(k.r0), w(k.n), w(k.sqrt_gma));
    let (ecos0, esin0) = (w(k.ecos0), w(k.esin0));
    let one = Wf32::<M>::ONE;
    let two = Wf32::<M>(2.0);
    let x = newton_f32(ecos0, esin0, w(k.m));
    let (s, cs) = sin_cos_f32(x.0);
    let (s, cs) = (Wf32::<M>(s), Wf32::<M>(cs));
    let sh = Wf32::<M>(sin_cos_f32(0.5 * x.0).0);
    let omc = two * sh * sh;
    let r = a * (one - ecos0 * cs + esin0 * s);
    let fm1 = -((a / r0) * omc);
    let g = w(dt) - (x - s) / n;
    let fdot = -(sqrt_gma / (r * r0)) * s;
    let gm1 = -((a / r) * omc);
    let p1 = p + (mul_vf(p, fm1.0) + mul_vf(v, g.0));
    let v1 = v + (mul_vf(p, fdot.0) + mul_vf(v, gm1.0));
    Some((p1, v1))
}

/// Option D: DS drift; the Kepler equation is solved in f32, then one DS
/// Newton step with the DS small-angle series corrects x, sin x, cos x and
/// 1 − cos x to first order in the step.
fn drift_d<const M: u8>(
    c: &Consts<M>,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
    dt: Ds<M>,
) -> Option<(Vec3<Ds<M>>, Vec3<Ds<M>>)> {
    let k = ds_scalars(c, p, v, dt)?;
    let one = Ds::<M>::ONE;
    let (ec, es) = (Wf32::<M>(k.ecos0.hi), Wf32::<M>(k.esin0.hi));
    let x32 = newton_f32(ec, es, Wf32(k.m.hi));
    let xd = Ds::<M>::from_f32(x32.0);
    let (sx, cx, omcx) = xd.sin_cos_small();
    let f = xd - k.ecos0 * sx + k.esin0 * omcx - k.m;
    let fp = Wf32::<M>::ONE - ec * Wf32(cx.hi) + es * Wf32(sx.hi);
    let d = (Wf32::<M>(f.hi) / fp).0;
    let x = xd.add_f(-d);
    let s = sx - cx.mul_f(d);
    let cs = cx + sx.mul_f(d);
    let omc = omcx - sx.mul_f(d);
    let r = k.a * (one - k.ecos0 * cs + k.esin0 * s);
    let ir = one / r;
    let fm1 = -(omc * k.a * k.ir0);
    let g = dt.add_f(-(Wf32::<M>((x - s).hi) / Wf32(k.n.hi)).0);
    let fdot = -(k.sqrt_gma * ir * k.ir0 * s);
    let gm1 = -(omc * k.a * ir);
    let _ = (k.alpha, k.r0);
    let p1 = p + (p * fm1 + v * g);
    let v1 = v + (p * fdot + v * gm1);
    Some((p1, v1))
}

/// Option B kick: the f32 kick acceleration at the f32 position, times h,
/// added to the DS velocity.
fn kick_b<const M: u8>(
    c: &Consts<M>,
    tab: &Tables<M>,
    k: usize,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
) -> Vec3<Ds<M>> {
    let a = fring_model::physics::kick_acceleration(
        &c.field_w,
        &c.gm_moons_w,
        tab.w.moon_positions_km(k),
        tab.w.indirect_km_s2(k),
        hi(p),
    );
    add_wv(v, a * Wf32(c.h_s as f32))
}

/// Kick of B+ and D: Saturn's J terms as f32 magnitudes (the formulas of
/// `SaturnField::acceleration_without_point_mass`) times the DS position;
/// the moons with DS separations and f32 forces; the f32 indirect term.
fn kick_dir<const M: u8>(
    c: &Consts<M>,
    tab: &Tables<M>,
    k: usize,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
) -> Vec3<Ds<M>> {
    let f = &c.field_w;
    let cw = |x: f32| Wf32::<M>(x);
    let pw = hi(p);
    let r2 = pw.norm_squared();
    let r = r2.sqrt();
    let u = pw.z / r;
    let u2 = u * u;
    let q = f.reference_radius_km / r;
    let q2 = q * q;
    let q4 = q2 * q2;
    let q6 = q4 * q2;
    let p2 = (cw(3.0) * u2 - Wf32::<M>::ONE) * cw(0.5);
    let p4 = ((cw(35.0) * u2 - cw(30.0)) * u2 + cw(3.0)) * cw(0.125);
    let p6 = (((cw(231.0) * u2 - cw(315.0)) * u2 + cw(105.0)) * u2 - cw(5.0)) * cw(0.0625);
    let d2 = cw(3.0) * u;
    let d4 = ((cw(35.0) * u2 - cw(15.0)) * u) * cw(0.5);
    let d6 = (((cw(1386.0) * u2 - cw(1260.0)) * u2 + cw(210.0)) * u) * cw(0.0625);
    let j2q = f.j2 * q2;
    let j4q = f.j4 * q4;
    let j6q = f.j6 * q6;
    let aa = cw(3.0) * j2q * p2 + cw(5.0) * j4q * p4 + cw(7.0) * j6q * p6;
    let b = j2q * d2 + j4q * d4 + j6q * d6;
    let radial = -aa - u * b;
    let kk = -f.gm_km3_s2 / (r2 * r);
    let h = cw(c.h_s as f32);
    let sxy = (kk * radial * h).0;
    let sz = (kk * b * r * h).0;
    // Moons: separation in DS, force in f32; plus the indirect term.
    let mut am = Vec3::new(Wf32::<M>::ZERO, Wf32::<M>::ZERO, Wf32::<M>::ZERO);
    for (gm, mp) in c.gm_moons_w.iter().zip(tab.d.moon_positions_km(k)) {
        let d = Vec3::new(
            Wf32::<M>((mp.x - p.x).hi),
            Wf32::<M>((mp.y - p.y).hi),
            Wf32::<M>((mp.z - p.z).hi),
        );
        let dd2 = d.norm_squared();
        am += d * (*gm / (dd2 * dd2.sqrt()));
    }
    let small = (am + tab.w.indirect_km_s2(k)) * h;
    let dv = Vec3::new(p.x.mul_f(sxy), p.y.mul_f(sxy), p.z.mul_f(sxy).add_f(sz));
    add_wv(v + dv, small)
}

/// One Kepler drift of option A for tests: `kepler_drift::<Wf32>`.
pub fn drift_a<const M: u8>(
    c: &Consts<M>,
    p: Vec3<Wf32<M>>,
    v: Vec3<Wf32<M>>,
    dt: Wf32<M>,
) -> Option<(Vec3<Wf32<M>>, Vec3<Wf32<M>>)> {
    kepler_drift(c.field_w.gm_km3_s2, p, v, dt)
}

/// Osculating semi-major axis (km) of an f64 state about GM (km³ s⁻²).
pub fn semi_major_axis(gm: f64, p: Vec3<f64>, v: Vec3<f64>) -> f64 {
    1.0 / (2.0 / p.norm() - v.norm_squared() / gm)
}

/// Angle (rad) from the projection of `a` on the x-y plane to that of `b`.
pub fn phase_difference(a: Vec3<f64>, b: Vec3<f64>) -> f64 {
    (a.x * b.y - a.y * b.x).atan2(a.x * b.x + a.y * b.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wf32::model;
    use fring_model::params::saturn_from_toml_str;
    use fring_model::physics::{circular_speed, wh_steps};

    const M4: u8 = model::M4;

    fn saturn() -> SaturnParams {
        saturn_from_toml_str(include_str!("../../../params/saturn.toml")).unwrap()
    }

    /// Every option follows the f64 reference over 10 orbits without moons
    /// (1500 steps): A to f32 accuracy, the DS options much better.
    #[test]
    fn options_follow_f64_over_ten_orbits() {
        let sp = saturn();
        let f64f = SaturnField::<f64>::from_params(&sp);
        let eph = MoonEphemeris::none();
        let h = 356.25;
        let c = Consts::<M4>::new(&sp, &eph, h);
        let mut tab = Tables::<M4>::new(&eph);
        tab.fill(&eph, 0.0, h, 0, 1500).unwrap();
        let a0 = 140_222.43;
        let vc = circular_speed(&f64f, a0);
        let (p0, v0) = (Vec3::new(a0, 0.0, 10.0), Vec3::new(0.3, vc, 0.01));
        let (mut pr, mut vr) = (p0, v0);
        wh_steps(&f64f, &mut pr, &mut vr, h, 1500).unwrap();
        let ar = semi_major_axis(sp.gm_km3_s2, pr, vr);
        // Bounds: about 5 times the values measured with the M4 model
        // (a: 0.22 km, 1.0e-4 rad; b: 9.0e-3, 1.2e-5; b+: 1.6e-4, 5.8e-8;
        // c: 1.4e-8, 9.4e-12; d: 3.5e-6, 1.3e-8).
        let expect = [
            (Opt::A, 1.0, 5e-4),
            (Opt::B, 0.05, 6e-5),
            (Opt::BPlus, 1e-3, 3e-7),
            (Opt::C, 1e-7, 5e-11),
            (Opt::D, 2e-5, 7e-8),
        ];
        for (opt, da_max, dl_max) in expect {
            let mut s = state_from_f64::<M4>(p0, v0);
            advance(opt, &c, &tab, &mut s).unwrap();
            let (p, v) = state_to_f64(&s);
            let da = (semi_major_axis(sp.gm_km3_s2, p, v) - ar).abs();
            let dl = phase_difference(pr, p).abs();
            println!("{}: da {da:.3e} km, dphase {dl:.3e} rad", opt.name());
            assert!(da < da_max && dl < dl_max, "{}: {da:e} {dl:e}", opt.name());
        }
    }
}
