//! The production Wisdom-Holman kernels (task G2, ADR 0006 and 0007): the
//! host side and the Rust twins of `shaders/wh.wgsl`.
//!
//! - [`Mode`]: option D (science runs) or option C (reference mode).
//! - [`pack_table`]: the kick table of a block of steps as the GPU reads it
//!   (moon positions and the indirect term from an f64 `KickTable`, rounded
//!   to DS; the f32 indirect term scaled by `h / f32(h)`).
//! - [`params`]: the run constants (a uniform buffer).
//! - [`HostTable`], [`advance`] and the small twins: the same arithmetic in
//!   Rust, with the error model `M` of [`crate::wf32`], to test the shader.

use crate::ds::Ds;
use crate::spike::{Consts, State, drift_d, newton_f32};
use crate::wf32::Wf32;
use fring_model::physics::{KickTable, kepler_drift, kick_acceleration};
use fring_model::{Error, Real, Result, Vec3};

/// Most moons the kernels take.
pub const MAX_MOONS: usize = 4;
/// f32 words per step in the kick table.
pub const ROW: usize = 36;
/// 32-bit words of the run constants (9 vec4).
pub const PARAM_WORDS: usize = 36;

/// The number format of the kernels (ADR 0006).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Option D: DS state and drift, f32 kick magnitudes. Science runs.
    D,
    /// Option C: DS throughout. Reference mode.
    C,
}

impl Mode {
    /// The entry point of the block kernel in `shaders/wh.wgsl`.
    pub fn entry(self) -> &'static str {
        match self {
            Mode::D => "step_d",
            Mode::C => "step_c",
        }
    }

    /// `"d"` or `"c"`.
    pub fn name(self) -> &'static str {
        match self {
            Mode::D => "d",
            Mode::C => "c",
        }
    }
}

/// Checks the number of moons.
pub fn check_moons(n: usize) -> Result<()> {
    if n > MAX_MOONS {
        return Err(Error::Backend(format!(
            "the GPU backend takes at most {MAX_MOONS} moons, the ephemeris has {n}"
        )));
    }
    Ok(())
}

/// The kick table of `table` as 36 f32 per step: moon j at `6 j` (DS x, y, z),
/// the indirect term as DS at 24, and as f32 times `scale` (`h / f32(h)`,
/// [`kick_scale`]) at 30.
pub fn pack_table(table: &KickTable<f64>, scale: f64) -> Vec<f32> {
    let mut out = vec![0f32; table.len() * ROW];
    for k in 0..table.len() {
        let row = &mut out[k * ROW..(k + 1) * ROW];
        for (j, p) in table
            .moon_positions_km(k)
            .iter()
            .enumerate()
            .take(MAX_MOONS)
        {
            for (c, x) in p.to_array().iter().enumerate() {
                let d = Ds::<0>::split(*x);
                row[6 * j + 2 * c] = d.hi;
                row[6 * j + 2 * c + 1] = d.lo;
            }
        }
        let ind = table.indirect_km_s2(k).to_array();
        for (c, x) in ind.iter().enumerate() {
            let d = Ds::<0>::split(*x);
            row[24 + 2 * c] = d.hi;
            row[24 + 2 * c + 1] = d.lo;
            row[30 + c] = (*x * scale) as f32;
        }
    }
    out
}

/// The run constants for the uniform buffer (`prm` in `wh.wgsl`).
pub fn params<const M: u8>(c: &Consts<M>, gm_moons_km3_s2: &[f64]) -> Vec<f32> {
    let f = &c.field_d;
    let h = Ds::<M>::split(c.h_s);
    let half = Ds::<M>::split(0.5 * c.h_s);
    let ds2 = |d: Ds<M>| [d.hi, d.lo];
    let mut v: Vec<f32> = Vec::with_capacity(PARAM_WORDS);
    v.extend(ds2(f.gm_km3_s2));
    v.extend(ds2(h));
    v.extend(ds2(c.inv_gm_d));
    v.extend(ds2(f.reference_radius_km));
    v.extend(ds2(f.j2));
    v.extend(ds2(f.j4));
    v.extend(ds2(f.j6));
    v.extend(ds2(half));
    v.extend([1.0, 0.0, 0.0, 0.0]); // the guard ds_one
    v.extend([c.field_w.j2.0, c.field_w.j4.0, c.field_w.j6.0, 0.0]);
    let mut gm_s = [0f32; MAX_MOONS];
    let mut gm_d = [Ds::<M>::ZERO; MAX_MOONS];
    for (j, g) in gm_moons_km3_s2.iter().enumerate().take(MAX_MOONS) {
        gm_s[j] = c.gm_moons_w[j].0;
        gm_d[j] = Ds::split(*g);
    }
    v.extend(gm_s);
    for d in gm_d {
        v.extend(ds2(d));
    }
    debug_assert_eq!(v.len(), PARAM_WORDS);
    v
}

/// The control uniform of one dispatch: steps, particles, plane stride, moons.
pub fn control(n_steps: usize, n_particles: usize, cap: usize, n_moons: usize) -> [u32; 4] {
    [
        n_steps as u32,
        n_particles as u32,
        cap as u32,
        n_moons as u32,
    ]
}

/// The kick table in the form of the Rust twins (what the GPU reads, in Rust).
#[derive(Clone, Debug)]
pub struct HostTable<const M: u8> {
    /// GM of the moons in DS.
    pub gm_ds: Vec<Ds<M>>,
    /// Moon positions, `len × n_moons`, step-major, DS.
    pub pos: Vec<Vec3<Ds<M>>>,
    /// Indirect term, DS (option C).
    pub ind_ds: Vec<Vec3<Ds<M>>>,
    /// Indirect term, f32, scaled by `h / f32(h)` (option D).
    pub ind_w: Vec<Vec3<Wf32<M>>>,
    n_moons: usize,
}

impl<const M: u8> HostTable<M> {
    /// The table that [`pack_table`] packs for `table`.
    pub fn from_kick_table(table: &KickTable<f64>, scale: f64) -> Self {
        let n_moons = table.n_moons();
        let ds3 = |p: Vec3<f64>| Vec3::new(Ds::split(p.x), Ds::split(p.y), Ds::split(p.z));
        let mut pos = Vec::new();
        let mut ind_ds = Vec::new();
        let mut ind_w = Vec::new();
        for k in 0..table.len() {
            pos.extend(table.moon_positions_km(k).iter().map(|p| ds3(*p)));
            let i = table.indirect_km_s2(k);
            ind_ds.push(ds3(i));
            ind_w.push(Vec3::new(
                Wf32((i.x * scale) as f32),
                Wf32((i.y * scale) as f32),
                Wf32((i.z * scale) as f32),
            ));
        }
        Self {
            gm_ds: table.gm_km3_s2().iter().map(|&g| Ds::split(g)).collect(),
            pos,
            ind_ds,
            ind_w,
            n_moons,
        }
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.ind_ds.len()
    }

    /// True if there are no steps.
    pub fn is_empty(&self) -> bool {
        self.ind_ds.is_empty()
    }

    /// Moon positions of step `k`.
    pub fn moons(&self, k: usize) -> &[Vec3<Ds<M>>] {
        &self.pos[k * self.n_moons..(k + 1) * self.n_moons]
    }
}

fn split_state<const M: u8>(s: &State<M>) -> (Vec3<Ds<M>>, Vec3<Ds<M>>) {
    (Vec3::new(s[0], s[1], s[2]), Vec3::new(s[3], s[4], s[5]))
}

fn join_state<const M: u8>(p: Vec3<Ds<M>>, v: Vec3<Ds<M>>) -> State<M> {
    [p.x, p.y, p.z, v.x, v.y, v.z]
}

fn hi3<const M: u8>(v: Vec3<Ds<M>>) -> Vec3<Wf32<M>> {
    Vec3::new(Wf32(v.x.hi), Wf32(v.y.hi), Wf32(v.z.hi))
}

/// Twin of `kick_dir` in `wh.wgsl` for step `k`: the J terms as f32
/// magnitudes (constants scaled by `h / f32(h)`) along the DS position, the
/// moons with DS separations and f32 forces, the scaled f32 indirect term;
/// everything times `f32(h)`, added to the DS velocity.
pub fn kick_dir<const M: u8>(
    c: &Consts<M>,
    tab: &HostTable<M>,
    k: usize,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
) -> Vec3<Ds<M>> {
    let f = &c.field_w;
    let cw = |x: f32| Wf32::<M>(x);
    let pw = hi3(p);
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
    let mut am = Vec3::new(Wf32::<M>::ZERO, Wf32::<M>::ZERO, Wf32::<M>::ZERO);
    for (gm, mp) in c.gm_moons_w.iter().zip(tab.moons(k)) {
        let d = Vec3::new(
            Wf32::<M>((mp.x - p.x).hi),
            Wf32::<M>((mp.y - p.y).hi),
            Wf32::<M>((mp.z - p.z).hi),
        );
        let dd2 = d.norm_squared();
        am += d * (*gm / (dd2 * dd2.sqrt()));
    }
    let small = (am + tab.ind_w[k]) * h;
    let dv = Vec3::new(p.x.mul_f(sxy), p.y.mul_f(sxy), p.z.mul_f(sxy).add_f(sz));
    let w = v + dv;
    Vec3::new(
        w.x.add_f(small.x.0),
        w.y.add_f(small.y.0),
        w.z.add_f(small.z.0),
    )
}

/// Twin of `kick_acc_c`: the whole kick acceleration in DS at step `k`.
pub fn kick_acc_c<const M: u8>(
    c: &Consts<M>,
    tab: &HostTable<M>,
    k: usize,
    p: Vec3<Ds<M>>,
) -> Vec3<Ds<M>> {
    kick_acceleration(&c.field_d, &tab.gm_ds, tab.moons(k), tab.ind_ds[k], p)
}

/// Twin of one block of `step_d` or `step_c` for one particle: advances
/// `s` by the `tab.len()` steps of the table. Returns `None` and leaves the
/// state unchanged if an orbit is not bound.
pub fn advance<const M: u8>(
    mode: Mode,
    c: &Consts<M>,
    tab: &HostTable<M>,
    s: &mut State<M>,
) -> Option<()> {
    let n = tab.len();
    if n == 0 {
        return Some(());
    }
    let h = Ds::<M>::split(c.h_s);
    let half = Ds::<M>::split(0.5 * c.h_s);
    let (mut p, mut v) = split_state(s);
    match mode {
        Mode::D => {
            (p, v) = drift_d(c, p, v, half)?;
            for k in 0..n {
                v = kick_dir(c, tab, k, p, v);
                let dt = if k + 1 == n { half } else { h };
                (p, v) = drift_d(c, p, v, dt)?;
            }
        }
        Mode::C => {
            let gm = c.field_d.gm_km3_s2;
            (p, v) = kepler_drift(gm, p, v, half)?;
            for k in 0..n {
                let a = kick_acc_c(c, tab, k, p);
                v = Vec3::new(v.x + a.x * h, v.y + a.y * h, v.z + a.z * h);
                let dt = if k + 1 == n { half } else { h };
                (p, v) = kepler_drift(gm, p, v, dt)?;
            }
        }
    }
    *s = join_state(p, v);
    Some(())
}

/// Twin of `newton_f32` in `wh.wgsl`.
pub fn newton<const M: u8>(ecos0: f32, esin0: f32, m: f32) -> f32 {
    newton_f32(Wf32::<M>(ecos0), Wf32::<M>(esin0), Wf32::<M>(m)).0
}

/// Twin of `t_ds_scalars`: r0, a, n, e cos E₀, e sin E₀ and m = n dt of a
/// Kepler drift of length `dt` (the DS scalars of option D). `None` if the
/// orbit is not bound.
pub fn scalars<const M: u8>(
    c: &Consts<M>,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
    dt: Ds<M>,
) -> Option<[Ds<M>; 6]> {
    let k = crate::spike::ds_scalars(c, p, v, dt)?;
    Some([k.r0, k.a, k.n, k.ecos0, k.esin0, k.m])
}

/// Twin of `drift_d` (option D Kepler drift of length `dt`).
pub fn drift<const M: u8>(
    c: &Consts<M>,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
    dt: Ds<M>,
) -> Option<(Vec3<Ds<M>>, Vec3<Ds<M>>)> {
    drift_d(c, p, v, dt)
}

/// Twin of `kepler_c` (DS Kepler drift of length `dt`).
pub fn kepler<const M: u8>(
    c: &Consts<M>,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
    dt: Ds<M>,
) -> Option<(Vec3<Ds<M>>, Vec3<Ds<M>>)> {
    kepler_drift(c.field_d.gm_km3_s2, p, v, dt)
}
