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
/// 32-bit words of the run constants (11 vec4; the last two hold the moon
/// radii of ICR-006: R squared in km² for each moon at word 36, and the rule
/// flag at word 40).
pub const PARAM_WORDS: usize = 44;

/// The number format of the kernels (ADR 0006).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Option D: DS state and drift, f32 kick magnitudes. Science runs.
    D,
    /// Option D+: as D with the J2 term of the kick in DS (ADR 0006 addendum).
    Dp,
    /// Option C: DS throughout. Reference mode.
    C,
}

impl Mode {
    /// The entry point of the block kernel in `shaders/wh.wgsl`.
    pub fn entry(self) -> &'static str {
        match self {
            Mode::D => "step_d",
            Mode::Dp => "step_dp",
            Mode::C => "step_c",
        }
    }

    /// The entry point with the moon-radius rule of ICR-006, or `None` for
    /// option C, which has no such rule.
    pub fn entry_rule(self) -> Option<&'static str> {
        match self {
            Mode::D => Some("step_d_r"),
            Mode::Dp => Some("step_dp_r"),
            Mode::C => None,
        }
    }

    /// `"d"` or `"c"`.
    pub fn name(self) -> &'static str {
        match self {
            Mode::D => "d",
            Mode::Dp => "dp",
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

/// `base` (the words of [`params`]) with the radii of the rule in km: R squared
/// in km² for each moon at word 36, and the rule flag at word 40.
pub fn with_radii(base: &[f32], radii_km: &[f64]) -> Vec<f32> {
    let mut v = base.to_vec();
    for (j, r) in radii_km.iter().enumerate().take(MAX_MOONS) {
        v[36 + j] = (r * r) as f32;
    }
    v[40] = 1.0;
    v
}

/// The two extra rows after the kick rows of a block, for the moon-radius rule
/// (ICR-006): row 0 holds the moon positions at the sync time at the start of
/// the block, `run_start_s + first_step h`, and row 1 those at the end,
/// `run_start_s + (first_step + n_steps) h`. The state of the particles is at
/// those times, so the half drifts before the first kick and after the last
/// kick can be checked too. Same layout as a kick row (moon j as DS at `6 j`).
///
/// A time outside the ephemeris gives a row with [`SYNC_NONE_KM`] in x of moon
/// 0 (the shader then skips that half drift). Moons beyond [`MAX_MOONS`] are
/// ignored.
pub fn sync_rows(
    eph: &fring_model::MoonEphemeris,
    run_start_s: f64,
    h_s: f64,
    first_step: u64,
    n_steps: usize,
) -> [f32; 2 * ROW] {
    let mut out = [0f32; 2 * ROW];
    let mut states = vec![fring_model::BodyState::default(); eph.tracks.len()];
    for (r, step) in [first_step, first_step + n_steps as u64]
        .into_iter()
        .enumerate()
    {
        let t = run_start_s + step as f64 * h_s;
        let row = &mut out[r * ROW..(r + 1) * ROW];
        if fring_model::physics::interpolate_moons_into(eph, t, &mut states).is_err() {
            row[0] = SYNC_NONE_KM;
            continue;
        }
        for (j, st) in states.iter().enumerate().take(MAX_MOONS) {
            for (c, x) in st.position_km.to_array().iter().enumerate() {
                let d = Ds::<0>::split(*x);
                row[6 * j + 2 * c] = d.hi;
                row[6 * j + 2 * c + 1] = d.lo;
            }
        }
    }
    out
}

/// The run constants for the uniform buffer (`prm` in `wh.wgsl`), with the
/// moon-radius rule off.
pub fn params<const M: u8>(c: &Consts<M>, gm_moons_km3_s2: &[f64]) -> Vec<f32> {
    params_with_radii(c, gm_moons_km3_s2, None)
}

/// As [`params`], with the moon radii of the radius rule (ICR-006) in km, one
/// for each moon (at most [`MAX_MOONS`]). A radius of 0 switches that moon off.
/// `None` switches the rule off.
pub fn params_with_radii<const M: u8>(
    c: &Consts<M>,
    gm_moons_km3_s2: &[f64],
    radii_km: Option<&[f64]>,
) -> Vec<f32> {
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
    v.extend([0.0; 8]);
    debug_assert_eq!(v.len(), PARAM_WORDS);
    match radii_km {
        Some(r) => with_radii(&v, r),
        None => v,
    }
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

    /// Number of moons.
    pub fn n_moons(&self) -> usize {
        self.n_moons
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

/// Twin of `kick_dir_plus` in `wh.wgsl` for step `k` (option D+): the J2 term
/// in DS with DS `h`; J4, J6 and the moons as in [`kick_dir`].
pub fn kick_dir_plus<const M: u8>(
    c: &Consts<M>,
    tab: &HostTable<M>,
    k: usize,
    p: Vec3<Ds<M>>,
    v: Vec3<Ds<M>>,
) -> Vec3<Ds<M>> {
    let fd = &c.field_d;
    let ds = |x: f32| Ds::<M>::from_f32(x);
    let r2d = p.norm_squared();
    let rd = r2d.sqrt();
    let ud = p.z / rd;
    let u2d = ud * ud;
    let qd = fd.reference_radius_km / rd;
    let q2d = qd * qd;
    let j2q = fd.j2 * q2d;
    let p2d = (ds(3.0) * u2d - ds(1.0)) * ds(0.5);
    let d2d = ds(3.0) * ud;
    let aa2 = ds(3.0) * j2q * p2d;
    let b2 = j2q * d2d;
    let radial2 = (ds(0.0) - aa2) - ud * b2;
    let kkd = (-fd.gm_km3_s2) / (r2d * rd);
    let hd = Ds::<M>::split(c.h_s);
    let sxy2 = kkd * radial2 * hd;
    let sz2 = kkd * b2 * rd * hd;

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
    let p4 = ((cw(35.0) * u2 - cw(30.0)) * u2 + cw(3.0)) * cw(0.125);
    let p6 = (((cw(231.0) * u2 - cw(315.0)) * u2 + cw(105.0)) * u2 - cw(5.0)) * cw(0.0625);
    let d4 = ((cw(35.0) * u2 - cw(15.0)) * u) * cw(0.5);
    let d6 = (((cw(1386.0) * u2 - cw(1260.0)) * u2 + cw(210.0)) * u) * cw(0.0625);
    let j4q = f.j4 * q4;
    let j6q = f.j6 * q6;
    let aa = cw(5.0) * j4q * p4 + cw(7.0) * j6q * p6;
    let b = j4q * d4 + j6q * d6;
    let radial = -aa - u * b;
    let kk = -f.gm_km3_s2 / (r2 * r);
    let h = cw(c.h_s as f32);
    let sxy = sxy2.add_f((kk * radial * h).0);
    let sz = sz2.add_f((kk * b * r * h).0);
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
    let dv = Vec3::new(p.x * sxy, p.y * sxy, p.z * sxy + sz);
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
        Mode::D | Mode::Dp => {
            (p, v) = drift_d(c, p, v, half)?;
            for k in 0..n {
                v = if mode == Mode::D {
                    kick_dir(c, tab, k, p, v)
                } else {
                    kick_dir_plus(c, tab, k, p, v)
                };
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

/// The moon positions at the two sync times of a block as the shader reads
/// them (rows `n` and `n + 1` of the table, see [`sync_rows`]); `None` for a
/// time outside the ephemeris.
pub type SyncMoons<const M: u8> = [Option<Vec<Vec3<Ds<M>>>>; 2];

/// Reads the rows of [`sync_rows`] for `n_moons` moons.
pub fn sync_moons<const M: u8>(rows: &[f32; 2 * ROW], n_moons: usize) -> SyncMoons<M> {
    let one = |r: usize| {
        let row = &rows[r * ROW..(r + 1) * ROW];
        if !moon_row_valid(row[0]) {
            return None;
        }
        let ds = |i: usize| Ds::<M> {
            hi: row[i],
            lo: row[i + 1],
        };
        Some(
            (0..n_moons.min(MAX_MOONS))
                .map(|j| Vec3::new(ds(6 * j), ds(6 * j + 2), ds(6 * j + 4)))
                .collect(),
        )
    };
    [one(0), one(1)]
}

/// What [`advance_dropping`] found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DropOutcome {
    /// 0 live, 1 unbound or invalid orbit, `2 + k` inside moon `k`.
    pub reason: u32,
    /// For `reason >= 2`: the drift that hit, 0 for the half drift before the
    /// first kick, `k + 1` for the drift after kick `k` (the last one is the
    /// half drift to the sync time at the end of the block).
    pub segment: Option<usize>,
    /// For `reason >= 2`: the squared distances of the two ends of the drift
    /// that hit, km².
    pub ends_d2_km2: [f32; 2],
    /// Smallest squared distance to each moon over the drifts that were
    /// checked, km² (infinity for a moon never checked). Over the whole block
    /// if nothing hit, up to the hit otherwise.
    pub min_d2_km2: [f32; MAX_MOONS],
}

/// Twin of one block of `step_d_r` or `step_dp_r` (ICR-006) for one particle:
/// as [`advance`] for the modes D and Dp, and also the radius rule. `r2_km2`
/// holds the squared radius of each moon (0 switches a moon off), `sync` the
/// moon positions at the two ends of the block. The state `s` changes only
/// if the particle is not dropped.
///
/// The moon-radius test is done for each drift (including the half drifts at
/// the two ends of the block) on the position relative to each moon, linear
/// between its ends; a drift is skipped when either end has no moon position.
pub fn advance_dropping<const M: u8>(
    mode: Mode,
    c: &Consts<M>,
    tab: &HostTable<M>,
    sync: &SyncMoons<M>,
    r2_km2: &[f32],
    s: &mut State<M>,
) -> DropOutcome {
    assert!(mode != Mode::C, "the radius rule is for the modes D and Dp");
    let n = tab.len();
    let nm = tab.n_moons().min(MAX_MOONS);
    let mut out = DropOutcome {
        reason: 0,
        segment: None,
        ends_d2_km2: [0.0; 2],
        min_d2_km2: [f32::INFINITY; MAX_MOONS],
    };
    if n == 0 {
        return out;
    }
    let h = Ds::<M>::split(c.h_s);
    let half = Ds::<M>::split(0.5 * c.h_s);
    let (mut p, mut v) = split_state(s);
    let rel_to = |moons: Option<&[Vec3<Ds<M>>]>, p: Vec3<Ds<M>>| -> Option<Vec<[f32; 3]>> {
        moons.map(|m| m.iter().take(nm).map(|mp| moon_rel(p, *mp)).collect())
    };
    let mut prev = rel_to(sync[0].as_deref(), p);
    // Checks the drift that ended at `p`; the moon positions at its end are `end`.
    let check = |seg: usize,
                 prev: &mut Option<Vec<[f32; 3]>>,
                 end: Option<&[Vec3<Ds<M>>]>,
                 p: Vec3<Ds<M>>,
                 out: &mut DropOutcome| {
        let next = rel_to(end, p);
        if let (Some(a), Some(b)) = (prev.as_ref(), next.as_ref()) {
            for j in 0..nm {
                let d2 = segment_min_d2::<M>(a[j], b[j]);
                out.min_d2_km2[j] = out.min_d2_km2[j].min(d2);
                if out.reason == 0 && r2_km2.get(j).is_some_and(|r2| *r2 > 0.0 && d2 < *r2) {
                    let sq = |x: [f32; 3]| x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                    out.reason = 2 + j as u32;
                    out.segment = Some(seg);
                    out.ends_d2_km2 = [sq(a[j]), sq(b[j])];
                }
            }
        }
        *prev = next;
    };
    let unbound = DropOutcome { reason: 1, ..out };
    let Some((p1, v1)) = drift_d(c, p, v, half) else {
        return unbound;
    };
    (p, v) = (p1, v1);
    check(0, &mut prev, Some(tab.moons(0)), p, &mut out);
    for k in 0..n {
        if out.reason != 0 {
            return out;
        }
        v = if mode == Mode::D {
            kick_dir(c, tab, k, p, v)
        } else {
            kick_dir_plus(c, tab, k, p, v)
        };
        let dt = if k + 1 == n { half } else { h };
        let Some((p1, v1)) = drift_d(c, p, v, dt) else {
            return DropOutcome { reason: 1, ..out };
        };
        (p, v) = (p1, v1);
        let end = if k + 1 == n {
            sync[1].as_deref()
        } else {
            Some(tab.moons(k + 1))
        };
        check(k + 1, &mut prev, end, p, &mut out);
    }
    if out.reason != 0 {
        return out;
    }
    *s = join_state(p, v);
    out
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

/// Twin of `moon_rel` in `wh.wgsl`: the position of the particle relative to
/// a moon, `p - m`, as f32. The subtraction is done in double-single, so the
/// large equal parts cancel before the rounding to f32.
pub fn moon_rel<const M: u8>(p: Vec3<Ds<M>>, m: Vec3<Ds<M>>) -> [f32; 3] {
    [(p.x - m.x).hi, (p.y - m.y).hi, (p.z - m.z).hi]
}

/// Twin of `seg_min_d2` in `wh.wgsl`: the smallest squared distance from the
/// origin to the segment from `a` to `b` (km²), in f32 with the order of the
/// shader.
pub fn segment_min_d2<const M: u8>(a: [f32; 3], b: [f32; 3]) -> f32 {
    let w = |x: f32| Wf32::<M>(x);
    let dot = |u: [Wf32<M>; 3], v: [Wf32<M>; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let (a, b) = (a.map(w), b.map(w));
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let dd = dot(d, d);
    let mut s = w(0.0);
    if dd.0 > 0.0 {
        s = Wf32::<M>(((-dot(a, d)) / dd).0.clamp(0.0, 1.0));
    }
    let c = [a[0] + d[0] * s, a[1] + d[1] * s, a[2] + d[2] * s];
    dot(c, c).0
}

/// True if the moon row of the kick table holds a real position and not the
/// sentinel for a time outside the ephemeris ([`SYNC_NONE_KM`]).
pub fn moon_row_valid(x_hi_km: f32) -> bool {
    x_hi_km < 1.0e20
}

/// The x position in the table that marks "no moon position" (a time outside
/// the ephemeris): the radius rule skips that drift.
pub const SYNC_NONE_KM: f32 = 1.0e30;

#[cfg(test)]
mod radius_tests {
    use super::*;
    use crate::wf32::model;

    const M4: u8 = model::M4;

    #[test]
    fn segment_distance_cases() {
        // Crosses 20 km from the centre, ends 70 km away: the minimum is 20.
        let d = segment_min_d2::<M4>([-70.0, 20.0, 0.0], [70.0, 20.0, 0.0]);
        assert!((d - 400.0).abs() < 1e-3, "{d}");
        // Ends before the closest point: the end point is the minimum.
        let d = segment_min_d2::<M4>([-70.0, 20.0, 0.0], [-30.0, 20.0, 0.0]);
        assert!((d - (900.0 + 400.0)).abs() < 1e-2, "{d}");
        // Zero length.
        let d = segment_min_d2::<M4>([3.0, 4.0, 0.0], [3.0, 4.0, 0.0]);
        assert_eq!(d, 25.0);
        // Grazing at radius + 1 km is above the radius, at radius - 1 below.
        let r = 43.09f32;
        let above = segment_min_d2::<M4>([-60.0, r + 1.0, 0.0], [60.0, r + 1.0, 0.0]);
        let below = segment_min_d2::<M4>([-60.0, r - 1.0, 0.0], [60.0, r - 1.0, 0.0]);
        assert!(above > r * r && below < r * r);
    }

    #[test]
    fn sync_rows_match_the_ephemeris_and_use_the_sentinel() {
        let sp =
            fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
                .unwrap();
        let moons =
            fring_model::params::moons_from_toml_str(include_str!("../../../params/moons.toml"))
                .unwrap();
        let h = 356.2537;
        let eph =
            fring_model::physics::build_ephemeris(&sp, &moons, 10.0 * h + 7200.0, 1800.0).unwrap();
        let rows = sync_rows(&eph, 0.0, h, 3, 4);
        for (r, t) in [(0, 3.0 * h), (1, 7.0 * h)] {
            let st = fring_model::physics::interpolate_moons(&eph, t).unwrap();
            for (j, m) in st.moons.iter().enumerate() {
                let p = m.position_km.to_array();
                for c in 0..3 {
                    let v = f64::from(rows[r * ROW + 6 * j + 2 * c])
                        + f64::from(rows[r * ROW + 6 * j + 2 * c + 1]);
                    assert!((v - p[c]).abs() < 1e-9 * p[c].abs().max(1.0), "{r} {j} {c}");
                }
            }
        }
        // The end is outside the ephemeris: sentinel in the second row only.
        let rows = sync_rows(&eph, 0.0, h, 0, 1_000_000);
        assert!(moon_row_valid(rows[0]) && !moon_row_valid(rows[ROW]));
    }

    #[test]
    fn dropping_twin_equals_advance_when_nothing_drops_and_freezes_otherwise() {
        use fring_model::physics::{
            EpicycleElements, KickTable, SaturnField, build_ephemeris, epicycle_state,
        };
        let sp =
            fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
                .unwrap();
        let moons =
            fring_model::params::moons_from_toml_str(include_str!("../../../params/moons.toml"))
                .unwrap();
        let h = 356.2537;
        let eph = build_ephemeris(&sp, &moons, 20.0 * h + 7200.0, 1800.0).unwrap();
        let gms: Vec<f64> = eph.tracks.iter().map(|t| t.gm_km3_s2).collect();
        let c = Consts::<M4>::for_step(&sp, &gms, h);
        let mut table = KickTable::<f64>::for_ephemeris(&eph);
        let n = 6;
        table.fill(&eph, 0.0, h, 0, n).unwrap();
        let scale = crate::spike::kick_scale(h);
        let tab = HostTable::<M4>::from_kick_table(&table, scale);
        let sync = sync_moons::<M4>(&sync_rows(&eph, 0.0, h, 0, n), 2);
        let field = SaturnField::<f64>::from_params(&sp);
        for mode in [Mode::D, Mode::Dp] {
            for k in 0..12 {
                let el = EpicycleElements {
                    a_km: 139_000.0 + 250.0 * k as f64,
                    e: 0.0024,
                    i_rad: 1e-4,
                    mean_anomaly_rad: 0.5 * k as f64,
                    z_phase_rad: 0.2 * k as f64,
                    mean_longitude_rad: 0.9 * k as f64,
                };
                let (p, v) = epicycle_state(&field, &el);
                let s0 = crate::spike::state_from_f64::<M4>(p, v);
                let mut a = s0;
                advance(mode, &c, &tab, &mut a).unwrap();
                let mut b = s0;
                let o = advance_dropping(mode, &c, &tab, &sync, &[0.0, 0.0], &mut b);
                assert_eq!(o.reason, 0);
                assert_eq!(a, b, "rule off: same bits as advance");
                // Radius 1e6 km: dropped in the first drift, state frozen.
                let mut f = s0;
                let o = advance_dropping(mode, &c, &tab, &sync, &[1.0e12, 0.0], &mut f);
                assert_eq!((o.reason, o.segment), (2, Some(0)));
                assert_eq!(f, s0, "frozen");
            }
        }
        // Unbound: three times the escape speed.
        let (p, v) = epicycle_state(
            &field,
            &EpicycleElements {
                a_km: 140_000.0,
                e: 0.0,
                i_rad: 0.0,
                mean_anomaly_rad: 0.0,
                z_phase_rad: 0.0,
                mean_longitude_rad: 0.0,
            },
        );
        let mut s = crate::spike::state_from_f64::<M4>(p, v * 3.0);
        let s0 = s;
        let o = advance_dropping(Mode::D, &c, &tab, &sync, &[1.0, 1.0], &mut s);
        assert_eq!(o.reason, 1);
        assert_eq!(s, s0);
    }

    #[test]
    fn params_have_the_radii_and_the_flag() {
        let sp =
            fring_model::params::saturn_from_toml_str(include_str!("../../../params/saturn.toml"))
                .unwrap();
        let gms = [0.01071, 0.00926];
        let c = Consts::<M4>::for_step(&sp, &gms, 360.0);
        let off = params(&c, &gms);
        assert_eq!(off.len(), PARAM_WORDS);
        assert!(off[36..].iter().all(|x| *x == 0.0));
        let on = params_with_radii(&c, &gms, Some(&[43.09, 40.63]));
        assert_eq!(on[..36], off[..36]);
        assert_eq!(on[36], (43.09f64 * 43.09) as f32);
        assert_eq!(on[37], (40.63f64 * 40.63) as f32);
        assert_eq!(on[38], 0.0);
        assert_eq!(on[40], 1.0);
    }
}
