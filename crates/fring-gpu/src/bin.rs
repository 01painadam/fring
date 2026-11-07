//! Binning by cell (task K5-BIN): the cell of each particle, a counting sort
//! of the particles into hashed buckets, and a sort of each bucket by
//! (cell key, particle id). The GPU side is [`GpuBinner`]
//! (`shaders/bin.wgsl`); the rest of this file is its Rust twin and the f64
//! reference.
//!
//! # The cell (`crates/fring-collide/SPEC.md` 5.1)
//!
//! With the local coordinates of PLAN 4.2 (x = ρ − a₀, y_c = a₀ ((φ − φ₀)
//! mod 2π), z), the cell is (⌊x/L_x⌋, ⌊y_c/L_y⌋, ⌊z/L_z⌋), where φ is the
//! longitude in the rotating frame (the inertial longitude minus n₀ t) and
//! L_y = D / round(D / L_y,config), so that whole cells fill the domain
//! length D. x, y and z come from the double-single (DS) state (SPEC 5.4).
//!
//! # Why hashed buckets
//!
//! A dense array of cells does not fit a full ring (about 4 × 10¹⁰ cells for
//! the core). The key (ix, iy, iz) is packed in two `u32` words (22, 21 and 21
//! bits, ordered like the triple). The key is hashed into one of M buckets
//! (M = the next power of two ≥ the number of particles), the particles are
//! counted and placed by a counting sort over the buckets, and each member of
//! a bucket is put at its rank in (key, id) order. Then every cell is one run of equal keys in
//! `sorted`/`sorted_keys`, with its members in id order (SPEC 5.5, Q4), and
//! the result does not depend on the order of the atomic operations.
//!
//! # Output of a bin pass
//!
//! - `sorted[p]`: particle ids in bucket order; inside a bucket in (key, id)
//!   order. From `n_binned` on the entries are [`NO_ID`].
//! - `sorted_keys[p]`: the key of `sorted[p]`.
//! - `bucket_start[b]`: first position of bucket b; `bucket_start[M]` =
//!   `n_binned`.
//! - A particle that is dropped (status reason ≠ 0), has a bad state, or lies
//!   outside the range of the key (|ix| ≥ 2²¹ − 1, |iz| ≥ 2²⁰ − 1, or a
//!   longitude outside the arc) gets the key [`NO_KEY`] and is not binned.
//!   Their numbers are `n_not_binned` and `n_out_of_range`.
//!
//! # Precision
//!
//! See the test `twin_agrees_with_f64_up_to_the_edge_rule` in this file for
//! the rule that links the DS twin to the f64 reference.
//!
//! The DS arithmetic relies on the guards of `ds.wgsl`: the caller must have
//! run the DS self-test ([`crate::probe::SelfTest`]), as for the step
//! kernels.

use crate::ds::Ds;
use crate::gpu::Gpu;
use crate::wf32::model;
use crate::wh_gpu::grid;
use fring_model::config::{Collisions, Domain};
use fring_model::{Error, Real, Result, RunConfig};
use std::f64::consts::{PI, TAU};

/// The twin works in this f32 model (the Apple M4 as measured).
pub type Dsm = Ds<{ model::M4 }>;

/// Id value of an empty entry of `sorted`.
pub const NO_ID: u32 = u32::MAX;
/// The key of a particle that is not binned.
pub const NO_KEY: [u32; 2] = [u32::MAX, u32::MAX];
/// Largest number of azimuthal cells (21 bits).
pub const MAX_NY: u32 = 1 << 21;
/// Largest |ix| (the key has 22 bits for ix + 2²¹, and the last value is kept).
pub const MAX_IX: i32 = (1 << 21) - 2;
/// Largest |iz| (21 bits for iz + 2²⁰).
pub const MAX_IZ: i32 = (1 << 20) - 2;
/// Sectors of the longitude: k π/32, k = −32..=32.
const SECTORS: usize = 65;
/// 32/π as an f32 (twin of the literal in the shader).
const SECTOR_PER_RAD: f32 = 10.185916;
/// Words of the `prm` uniform (135 vec4).
pub const PRM_WORDS: usize = 135 * 4;
// 1/5, 1/7, ... 1/15 as f32, and 1/3 as DS (the same literals as the shader).
const C5: f32 = 0.2;
const C7: f32 = 0.14285715;
const C9: f32 = 0.11111111;
const C11: f32 = 0.09090909;
const C13: f32 = 0.07692308;
const C15: f32 = 0.06666667;
const THIRD: Dsm = Dsm {
    hi: 0.33333334,
    lo: -9.934108e-09,
};
/// Rounding at the end of the domain: y − D below this is the last cell, km.
const END_TOLERANCE_KM: f32 = 1.0e-6;

/// The cell grid: sizes, domain and frame (km, rad, s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinGeometry {
    /// Reference radius a₀, km.
    pub a0_km: f64,
    /// Mean motion n₀ of the rotating frame, rad/s.
    pub n0_rad_s: f64,
    /// Radial cell size L_x, km.
    pub lx_km: f64,
    /// Azimuthal cell size L_y = D / n_y, km (adjusted).
    pub ly_km: f64,
    /// Vertical cell size L_z, km.
    pub lz_km: f64,
    /// Number of azimuthal cells.
    pub ny: u32,
    /// Azimuthal length of the domain D, km.
    pub domain_km: f64,
    /// Longitude φ₀ of the start of the domain in the rotating frame, rad.
    pub phi0_rad: f64,
}

impl BinGeometry {
    /// A grid with the given sizes. `domain_km` = D (at most 2π a₀) and
    /// `phi0_rad` = φ₀. L_y is adjusted to D / round(D / `ly_km`).
    ///
    /// # Errors
    /// [`Error::Config`] if a size is not finite and positive, if D is more
    /// than 2π a₀, or if n_y is more than [`MAX_NY`].
    pub fn new(
        a0_km: f64,
        n0_rad_s: f64,
        cell_km: [f64; 3],
        domain_km: f64,
        phi0_rad: f64,
    ) -> Result<Self> {
        let pos = |name: &str, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(Error::Config(format!(
                    "binning: {name} must be finite and positive, not {v}"
                )))
            }
        };
        pos("a0_km", a0_km)?;
        pos("n0_rad_s", n0_rad_s)?;
        pos("cell_radial_km", cell_km[0])?;
        pos("cell_azimuthal_km", cell_km[1])?;
        pos("cell_vertical_km", cell_km[2])?;
        pos("domain length", domain_km)?;
        if !phi0_rad.is_finite() {
            return Err(Error::Config("binning: phi0 is not finite".to_string()));
        }
        if domain_km > TAU * a0_km * (1.0 + 1e-12) {
            return Err(Error::Config(format!(
                "binning: domain length {domain_km} km is more than the ring {} km",
                TAU * a0_km
            )));
        }
        let ny = (domain_km / cell_km[1]).round().max(1.0);
        if ny > f64::from(MAX_NY) {
            return Err(Error::Config(format!(
                "binning: {ny} azimuthal cells; at most {MAX_NY} (make cell_azimuthal_km larger)"
            )));
        }
        Ok(Self {
            a0_km,
            n0_rad_s,
            lx_km: cell_km[0],
            ly_km: domain_km / ny,
            lz_km: cell_km[2],
            ny: ny as u32,
            domain_km,
            phi0_rad,
        })
    }

    /// The grid of a run config with `model = "stochastic"`: the cell sizes
    /// of `[collisions]`, a₀ of `[frame]`, and the domain (the full ring:
    /// D = 2π a₀, φ₀ = 0; the arc: D = a₀ Δθ, φ₀ = centre − Δθ/2, SPEC 5.1).
    /// `n0_rad_s` is `physics::frames::reference_mean_motion` of a₀.
    ///
    /// # Errors
    /// [`Error::Config`] if the model is not stochastic, or as [`Self::new`].
    pub fn from_config(config: &RunConfig, n0_rad_s: f64) -> Result<Self> {
        let Collisions::Stochastic {
            cell_radial_km,
            cell_azimuthal_km,
            cell_vertical_km,
            ..
        } = config.collisions
        else {
            return Err(Error::Config(
                "binning needs collisions.model = \"stochastic\"".to_string(),
            ));
        };
        let a0 = config.frame.reference_semi_major_axis_km;
        let (d, phi0) = match config.domain {
            Domain::FullRing => (TAU * a0, 0.0),
            Domain::Arc {
                width_deg,
                centre_longitude_deg,
            } => {
                let w = width_deg.to_radians();
                (a0 * w, centre_longitude_deg.to_radians() - 0.5 * w)
            }
        };
        Self::new(
            a0,
            n0_rad_s,
            [cell_radial_km, cell_azimuthal_km, cell_vertical_km],
            d,
            phi0,
        )
    }

    /// α = (n₀ t + φ₀) mod 2π in f64: the inertial longitude minus α is
    /// (φ − φ₀) mod 2π. The host forms it in f64 and passes it as DS.
    pub fn alpha_rad(&self, time_s: f64) -> f64 {
        (self.n0_rad_s * time_s + self.phi0_rad).rem_euclid(TAU)
    }
}

/// A cell: indices along x, y_c and z.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cell {
    /// ⌊x/L_x⌋ (radial).
    pub ix: i32,
    /// ⌊y_c/L_y⌋ (azimuthal), in 0..n_y.
    pub iy: u32,
    /// ⌊z/L_z⌋ (vertical).
    pub iz: i32,
}

/// What the binning finds for one particle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binned {
    /// In this cell.
    Cell(Cell),
    /// Dropped, or a bad state (zero or huge distance from Saturn).
    NotBinned,
    /// Outside the range of the key, or outside the arc.
    OutOfRange,
}

/// Packs a cell into the two key words (twin of `pack_key`).
pub fn pack_key(c: Cell) -> [u32; 2] {
    let hi = (((c.ix + (1 << 21)) as u32) << 10) | (c.iy >> 11);
    let lo = ((c.iy & 2047) << 21) | ((c.iz + (1 << 20)) as u32);
    [hi, lo]
}

/// The cell of a key (inverse of [`pack_key`]); `None` for [`NO_KEY`].
pub fn unpack_key(k: [u32; 2]) -> Option<Cell> {
    if k == NO_KEY {
        return None;
    }
    Some(Cell {
        ix: (k[0] >> 10) as i32 - (1 << 21),
        iy: ((k[0] & 1023) << 11) | (k[1] >> 21),
        iz: (k[1] & ((1 << 21) - 1)) as i32 - (1 << 20),
    })
}

/// The key of a binning result.
pub fn key_of(b: Binned) -> [u32; 2] {
    match b {
        Binned::Cell(c) => pack_key(c),
        _ => NO_KEY,
    }
}

/// The hash of a key (twin of `hash_key`; 32-bit mix of Chris Wellons'
/// "lowbias32" kind, wrapping).
pub fn hash_key(k: [u32; 2]) -> u32 {
    let mut h = k[1] ^ k[0].wrapping_mul(0x85EB_CA6B);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^ (h >> 16)
}

/// The words of the `prm` uniform for `g` (see the header of `bin.wgsl`).
pub fn params(g: &BinGeometry) -> Vec<f32> {
    let mut w = vec![0f32; PRM_WORDS];
    let mut put = |vec4: usize, lane: usize, x: f64| {
        let d = Dsm::split(x);
        w[4 * vec4 + lane] = d.hi;
        w[4 * vec4 + lane + 1] = d.lo;
    };
    put(1, 0, g.a0_km);
    put(1, 2, g.lx_km);
    put(2, 0, g.ly_km);
    put(2, 2, g.lz_km);
    put(3, 0, TAU);
    put(3, 2, g.domain_km);
    for u in 0..SECTORS {
        let ang = (u as f64 - 32.0) * PI / 32.0;
        put(5 + 2 * u, 0, ang.cos());
        put(5 + 2 * u, 2, ang.sin());
        put(6 + 2 * u, 0, ang);
    }
    w[0] = 1.0;
    w
}

fn ds_at(prm: &[f32], vec4: usize, lane: usize) -> Dsm {
    Dsm {
        hi: prm[4 * vec4 + lane],
        lo: prm[4 * vec4 + lane + 1],
    }
}

fn negp(a: Dsm) -> bool {
    a.hi < 0.0 || (a.hi == 0.0 && a.lo < 0.0)
}

fn ds_floor(q: Dsm) -> f32 {
    let f = q.hi.floor();
    if q.hi == f && q.lo < 0.0 { f - 1.0 } else { f }
}

/// atan t for |t| < 0.0493 in DS (twin of `ds_atan_small`).
fn ds_atan_small(t: Dsm) -> Dsm {
    let u = t * t;
    let uh = u.hi;
    let p = uh.mul_add(
        uh.mul_add(uh.mul_add(uh.mul_add(uh.mul_add(-C15, C13), -C11), C9), -C7),
        C5,
    );
    let tail = (uh * uh) * p;
    let s = (Dsm::ONE - u * THIRD).add_f(tail);
    t * s
}

/// The cell of one particle from the DS inertial position (km), as the shader
/// computes it (twin of `cell_of` without the status test). `prm` is
/// [`params`], `alpha` is [`BinGeometry::alpha_rad`] as DS and `ny` is
/// `BinGeometry::ny`. Also returns the local coordinates (x, y_c, z) in km
/// (as f64, for the error measurement).
pub fn cell_ds(prm: &[f32], ny: u32, alpha: Dsm, p: [Dsm; 3]) -> (Binned, [f64; 3]) {
    let [px, py, pz] = p;
    let bad = (Binned::NotBinned, [0.0; 3]);
    let r2 = px * px + py * py;
    if !(r2.hi > 0.0 && r2.hi < 1.0e18) {
        return bad;
    }
    let rho = r2.sqrt();
    let coarse = py.hi.atan2(px.hi);
    let k = ((coarse * SECTOR_PER_RAD + 0.5).floor() as i32).clamp(-32, 32);
    let row = 5 + 2 * (k + 32) as usize;
    let ck = ds_at(prm, row, 0);
    let sk = ds_at(prm, row, 2);
    let phik = ds_at(prm, row + 1, 0);
    let xp = px * ck + py * sk;
    let yp = py * ck - px * sk;
    let t = yp / xp;
    let phi = phik + ds_atan_small(t);
    let twopi = ds_at(prm, 3, 0);
    let mut psi = phi - alpha;
    if negp(psi) {
        psi += twopi;
    }
    if negp(psi) {
        psi += twopi;
    }
    if !negp(psi - twopi) {
        psi -= twopi;
    }
    let a0 = ds_at(prm, 1, 0);
    let y = a0 * psi;
    let x = rho - a0;
    let local = [x.to_f64(), y.to_f64(), pz.to_f64()];
    let fx = ds_floor(x / ds_at(prm, 1, 2));
    let mut fy = ds_floor(y / ds_at(prm, 2, 0));
    let fz = ds_floor(pz / ds_at(prm, 2, 2));
    let nyf = ny as f32;
    if fy >= nyf {
        let over = y - ds_at(prm, 3, 2);
        if over.hi < END_TOLERANCE_KM {
            fy = nyf - 1.0;
        }
    }
    if !(fx.abs() <= MAX_IX as f32 && fz.abs() <= MAX_IZ as f32 && fy >= 0.0 && fy < nyf) {
        return (Binned::OutOfRange, local);
    }
    (
        Binned::Cell(Cell {
            ix: fx as i32,
            iy: fy as u32,
            iz: fz as i32,
        }),
        local,
    )
}

/// The same cell in f64 (the reference): `alpha` from
/// [`BinGeometry::alpha_rad`], `p` the inertial position in km. Also returns
/// the local coordinates (x, y_c, z) in km.
pub fn cell_f64(g: &BinGeometry, alpha: f64, p: [f64; 3]) -> (Binned, [f64; 3]) {
    let r2 = p[0] * p[0] + p[1] * p[1];
    if !(r2 > 0.0 && r2 < 1.0e18) {
        return (Binned::NotBinned, [0.0; 3]);
    }
    let rho = r2.sqrt();
    let psi = (p[1].atan2(p[0]) - alpha).rem_euclid(TAU);
    let (x, y, z) = (rho - g.a0_km, g.a0_km * psi, p[2]);
    let local = [x, y, z];
    let fx = (x / g.lx_km).floor();
    let mut fy = (y / g.ly_km).floor();
    let fz = (z / g.lz_km).floor();
    let nyf = f64::from(g.ny);
    if fy >= nyf && y - g.domain_km < f64::from(END_TOLERANCE_KM) {
        fy = nyf - 1.0;
    }
    if !(fx.abs() <= f64::from(MAX_IX) && fz.abs() <= f64::from(MAX_IZ) && fy >= 0.0 && fy < nyf) {
        return (Binned::OutOfRange, local);
    }
    (
        Binned::Cell(Cell {
            ix: fx as i32,
            iy: fy as u32,
            iz: fz as i32,
        }),
        local,
    )
}

/// Distance in km of the local coordinates to the nearest cell boundary on
/// each axis (x, y, z). The azimuthal boundaries include 0 and D.
pub fn edge_distance_km(g: &BinGeometry, local: [f64; 3]) -> [f64; 3] {
    let d = |v: f64, l: f64| (v - l * (v / l).round()).abs();
    [
        d(local[0], g.lx_km),
        d(local[1], g.ly_km),
        d(local[2], g.lz_km),
    ]
}

/// The result of a bin pass (see the module documentation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinOutput {
    /// Particle ids in bucket order, (key, id) order inside a bucket; [`NO_ID`]
    /// from `n_binned` on. Length: the number of particles.
    pub sorted: Vec<u32>,
    /// The key of `sorted[p]` ([`NO_KEY`] from `n_binned` on).
    pub sorted_keys: Vec<[u32; 2]>,
    /// First position of each bucket, and `n_binned` at the end (M + 1 words).
    pub bucket_start: Vec<u32>,
    /// Number of binned particles.
    pub n_binned: u32,
    /// Particles dropped or with a bad state.
    pub n_not_binned: u32,
    /// Particles outside the range of the key or outside the arc.
    pub n_out_of_range: u32,
}

/// The counting sort in Rust, step for step as the shader: clear, count,
/// exclusive prefix sum, scatter (here in index order), sort of each bucket
/// by (key, id), emit. `m` buckets (a power of two).
pub fn bin_twin(cells: &[Binned], m: usize) -> BinOutput {
    assert!(
        m.is_power_of_two(),
        "the number of buckets is a power of two"
    );
    let n = cells.len();
    let keys: Vec<[u32; 2]> = cells.iter().map(|&b| key_of(b)).collect();
    let bucket = |k: [u32; 2]| (hash_key(k) as usize) & (m - 1);
    let mut counts = vec![0u32; m + 1];
    let (mut not_binned, mut out_of_range) = (0u32, 0u32);
    for (b, &k) in cells.iter().zip(&keys) {
        match b {
            Binned::Cell(_) => counts[bucket(k)] += 1,
            Binned::NotBinned => not_binned += 1,
            Binned::OutOfRange => out_of_range += 1,
        }
    }
    let mut start = vec![0u32; m + 1];
    for b in 0..m {
        start[b + 1] = start[b] + counts[b];
    }
    let total = start[m] as usize;
    let mut sorted = vec![NO_ID; n];
    let mut fill = start.clone();
    for (i, &k) in keys.iter().enumerate() {
        if k != NO_KEY {
            let b = bucket(k);
            sorted[fill[b] as usize] = i as u32;
            fill[b] += 1;
        }
    }
    for b in 0..m {
        let (s, e) = (start[b] as usize, start[b + 1] as usize);
        sorted[s..e].sort_by_key(|&i| (keys[i as usize], i));
    }
    let sorted_keys = (0..n)
        .map(|p| {
            if p < total {
                keys[sorted[p] as usize]
            } else {
                NO_KEY
            }
        })
        .collect();
    BinOutput {
        sorted,
        sorted_keys,
        bucket_start: start,
        n_binned: total as u32,
        n_not_binned: not_binned,
        n_out_of_range: out_of_range,
    }
}

/// The reference of the tests: the binned particles sorted by (key, id) with
/// the standard library, as one list of ids for each cell, in key order.
pub fn cells_reference(cells: &[Binned]) -> Vec<(Cell, Vec<u32>)> {
    let mut v: Vec<(Cell, u32)> = cells
        .iter()
        .enumerate()
        .filter_map(|(i, b)| match b {
            Binned::Cell(c) => Some((*c, i as u32)),
            _ => None,
        })
        .collect();
    v.sort_by_key(|&(c, i)| (pack_key(c), i));
    let mut out: Vec<(Cell, Vec<u32>)> = Vec::new();
    for (c, i) in v {
        match out.last_mut() {
            Some((lc, ids)) if *lc == c => ids.push(i),
            _ => out.push((c, vec![i])),
        }
    }
    out
}

/// The cells of a [`BinOutput`] as runs of equal keys: one list of ids for
/// each cell, in order of position. Equal keys that are not in one run give
/// two entries with the same cell (a defect that tests look for).
pub fn runs(out: &BinOutput) -> Vec<(Cell, Vec<u32>)> {
    let mut v: Vec<(Cell, Vec<u32>)> = Vec::new();
    let mut last: Option<[u32; 2]> = None;
    for p in 0..out.n_binned as usize {
        let k = out.sorted_keys[p];
        let c = unpack_key(k).expect("a binned particle has a cell");
        if last == Some(k) {
            v.last_mut().expect("run").1.push(out.sorted[p]);
        } else {
            v.push((c, vec![out.sorted[p]]));
            last = Some(k);
        }
    }
    v
}

/// The WGSL source: DS library and the binning kernels.
pub const BIN_WGSL: &str = concat!(
    include_str!("../shaders/ds.wgsl"),
    include_str!("../shaders/bin.wgsl")
);
/// Counts per workgroup of the scan (`SCAN_BLOCK` in the shader).
const SCAN_BLOCK: usize = 1024;
/// Info words after the bucket counts.
const INFO_WORDS: usize = 8;
/// Workgroup size of the per-particle and per-bucket kernels.
const BIN_WG: u32 = 64;

/// The passes of one bin pass, in order. Any prefix of the list is a valid
/// partial pass (for timing); only all of them give the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Zero the counts and the info words.
    Clear,
    /// Cell key of each particle; count the particles of each bucket.
    Keys,
    /// Exclusive scan of the counts inside blocks of 1024.
    ScanLocal,
    /// Scan of the block totals (one workgroup).
    ScanSums,
    /// Add the block offsets: bucket starts.
    ScanAdd,
    /// Place each particle in its bucket (atomic).
    Scatter,
    /// Place each particle in its bucket by its rank in (key, id) order, one
    /// thread per particle.
    Rank,
    /// Sorted keys, the number binned, the empty tail.
    Emit,
}

impl Stage {
    /// All passes in order.
    pub const ALL: [Stage; 8] = [
        Stage::Clear,
        Stage::Keys,
        Stage::ScanLocal,
        Stage::ScanSums,
        Stage::ScanAdd,
        Stage::Scatter,
        Stage::Rank,
        Stage::Emit,
    ];

    fn entry(self) -> &'static str {
        match self {
            Stage::Clear => "bin_clear",
            Stage::Keys => "bin_keys",
            Stage::ScanLocal => "scan_local",
            Stage::ScanSums => "scan_sums",
            Stage::ScanAdd => "scan_add",
            Stage::Scatter => "bin_scatter",
            Stage::Rank => "bin_rank",
            Stage::Emit => "bin_emit",
        }
    }
}

/// Bins the particles of a state buffer by cell on the GPU (see the module
/// documentation). It records its passes into the caller's command encoder
/// and never waits for the GPU, so it also builds for wasm32. The blocking
/// helpers that read results back are native only.
///
/// The state is the 6-plane DS buffer of [`crate::stepper::GpuStepper`]
/// (inertial, Saturn-centred) at the time given to [`Self::encode`]; the
/// status buffer gives the drop reasons.
pub struct GpuBinner {
    gpu: Gpu,
    geom: BinGeometry,
    n: usize,
    m: usize,
    mpad: usize,
    nb: usize,
    pipelines: Vec<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    prm_buf: wgpu::Buffer,
    ctl_buf: wgpu::Buffer,
    state: wgpu::Buffer,
    status: wgpu::Buffer,
    counts: wgpu::Buffer,
    start: wgpu::Buffer,
    keys: wgpu::Buffer,
    sorted: wgpu::Buffer,
    skeys: wgpu::Buffer,
}

fn le_bytes_f32(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl GpuBinner {
    /// A binner for the `n` particles of `state` and `status` (`cap` entries
    /// per plane, `8 + cap` status words) with the grid `geom`. `buckets` is
    /// M, a power of two; `None` takes the next power of two ≥ `n` (2²⁴ for
    /// 10⁷ particles: the counts take 64 MiB). A test may pass a small M to
    /// force different cells into one bucket.
    ///
    /// The caller must have run the DS self-test ([`crate::probe::SelfTest`]).
    ///
    /// # Errors
    /// [`Error::Config`] if M is not a power of two or is too large for a
    /// dispatch (more than 65535 × 1024 − 1 buckets), [`Error::Backend`] if
    /// a buffer does not fit the limits of the device or the buffers given
    /// are too small.
    pub fn new(
        gpu: &Gpu,
        state: &wgpu::Buffer,
        status: &wgpu::Buffer,
        n: usize,
        cap: usize,
        geom: BinGeometry,
        buckets: Option<usize>,
    ) -> Result<Self> {
        let m = buckets.unwrap_or_else(|| n.max(1).next_power_of_two());
        if !m.is_power_of_two() {
            return Err(Error::Config(format!(
                "binning: the number of buckets {m} is not a power of two"
            )));
        }
        let mpad = (m + 1).next_multiple_of(SCAN_BLOCK);
        let nb = mpad / SCAN_BLOCK;
        if nb > 65535 || cap < n || u32::try_from(mpad + INFO_WORDS + nb).is_err() {
            return Err(Error::Config(format!(
                "binning: {m} buckets for {n} particles (capacity {cap}) is too many"
            )));
        }
        let need_state = (6 * cap * 8) as u64;
        let need_status = (4 * (8 + cap)) as u64;
        if state.size() < need_state || status.size() < need_status {
            return Err(Error::Backend(format!(
                "binning: the state buffer ({} bytes) or the status buffer ({} bytes) is smaller than \
                 {need_state} or {need_status} bytes for capacity {cap}",
                state.size(),
                status.size()
            )));
        }
        let sizes = [
            (4 * (mpad + INFO_WORDS)) as u64,
            (4 * (mpad + nb)) as u64,
            (8 * cap) as u64,
            (4 * cap) as u64,
        ];
        let limit = gpu
            .limits
            .max_storage_buffer_binding_size
            .min(gpu.limits.max_buffer_size);
        if sizes.iter().any(|&s| s > limit) || gpu.limits.max_storage_buffers_per_shader_stage < 7 {
            return Err(Error::Backend(format!(
                "binning {n} particles into {m} buckets needs buffers of {sizes:?} bytes and 7 storage \
                 buffers; {} allows {limit} bytes and {} buffers",
                gpu.info.name, gpu.limits.max_storage_buffers_per_shader_stage
            )));
        }
        let d = gpu.device();
        let q = gpu.queue();
        let mk = |size: u64, usage: wgpu::BufferUsages| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size.max(16),
                usage,
                mapped_at_creation: false,
            })
        };
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let uni = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let prm_buf = mk((PRM_WORDS * 4) as u64, uni);
        q.write_buffer(&prm_buf, 0, &le_bytes_f32(&params(&geom)));
        let ctl_buf = mk(32, uni);
        let ctl: [u32; 8] = [
            n as u32,
            cap as u32,
            (m - 1) as u32,
            geom.ny,
            mpad as u32,
            nb as u32,
            m as u32,
            0,
        ];
        let ctl_bytes: Vec<u8> = ctl.iter().flat_map(|x| x.to_le_bytes()).collect();
        q.write_buffer(&ctl_buf, 0, &ctl_bytes);
        let counts = mk(sizes[0], rw);
        let start = mk(sizes[1], rw);
        let keys = mk(sizes[2], rw);
        let sorted = mk(sizes[3], rw);
        let skeys = mk(sizes[2], rw);

        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bin"),
            source: wgpu::ShaderSource::Wgsl(BIN_WGSL.into()),
        });
        let st = |ro| wgpu::BufferBindingType::Storage { read_only: ro };
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let u = wgpu::BufferBindingType::Uniform;
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bin"),
            entries: &[
                entry(0, st(true)),
                entry(1, st(true)),
                entry(2, u),
                entry(3, u),
                entry(4, u),
                entry(5, st(false)),
                entry(6, st(false)),
                entry(7, st(false)),
                entry(8, st(false)),
                entry(9, st(false)),
            ],
        });
        let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bin"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = Stage::ALL
            .iter()
            .map(|s| {
                d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(s.entry()),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(s.entry()),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        Ok(Self {
            gpu: gpu.clone(),
            geom,
            n,
            m,
            mpad,
            nb,
            pipelines,
            layout,
            prm_buf,
            ctl_buf,
            state: state.clone(),
            status: status.clone(),
            counts,
            start,
            keys,
            sorted,
            skeys,
        })
    }

    /// A binner for the particles of `stepper`.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn for_stepper(
        gpu: &Gpu,
        stepper: &crate::stepper::GpuStepper,
        geom: BinGeometry,
    ) -> Result<Self> {
        Self::new(
            gpu,
            stepper.state_buffer(),
            stepper.status_buffer(),
            stepper.len(),
            stepper.capacity(),
            geom,
            None,
        )
    }

    /// Records a whole bin pass for the state at `time_s` (seconds since the
    /// start of the run) into `encoder`: the passes of [`Stage::ALL`] in one
    /// compute pass. The caller submits. `time_s` is used for the rotating
    /// frame (α = n₀ t + φ₀ in f64, passed as DS); each call has its own
    /// small uniform buffer, so several calls can share one submission.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, time_s: f64) {
        self.encode_stages(encoder, time_s, Stage::ALL.len());
    }

    /// As [`Self::encode`] with only the first `stages` passes of
    /// [`Stage::ALL`] (the cost of a prefix; the result is complete only with
    /// all of them).
    pub fn encode_stages(&self, encoder: &mut wgpu::CommandEncoder, time_s: f64, stages: usize) {
        let d = self.gpu.device();
        let alpha = Dsm::split(self.geom.alpha_rad(time_s));
        let tdyn = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bin tdyn"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        {
            let mut v = tdyn.get_mapped_range_mut(..).expect("range");
            v.copy_from_slice(&le_bytes_f32(&[alpha.hi, alpha.lo, 0.0, 0.0]));
        }
        tdyn.unmap();
        fn e(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }
        }
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bin"),
            layout: &self.layout,
            entries: &[
                e(0, &self.state),
                e(1, &self.status),
                e(2, &self.prm_buf),
                e(3, &self.ctl_buf),
                e(4, &tdyn),
                e(5, &self.counts),
                e(6, &self.start),
                e(7, &self.keys),
                e(8, &self.sorted),
                e(9, &self.skeys),
            ],
        });
        let g = |count: usize| grid(count.max(1), BIN_WG);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bin"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind, &[]);
        for (k, stage) in Stage::ALL.iter().enumerate().take(stages) {
            let (x, y) = match stage {
                Stage::Clear => g(self.mpad + INFO_WORDS),
                Stage::Keys | Stage::Scatter | Stage::Rank | Stage::Emit => g(self.n),
                Stage::ScanLocal => (self.nb as u32, 1),
                Stage::ScanSums => (1, 1),
                Stage::ScanAdd => g(self.mpad),
            };
            pass.set_pipeline(&self.pipelines[k]);
            pass.dispatch_workgroups(x, y, 1);
        }
    }

    /// The grid.
    pub fn geometry(&self) -> &BinGeometry {
        &self.geom
    }

    /// Number of buckets M.
    pub fn buckets(&self) -> usize {
        self.m
    }

    /// `sorted`: `n` `u32` ids (usage `STORAGE | COPY_SRC`); see the module
    /// documentation.
    pub fn sorted_buffer(&self) -> &wgpu::Buffer {
        &self.sorted
    }

    /// `sorted_keys`: `n` keys of two `u32` (hi, lo).
    pub fn sorted_keys_buffer(&self) -> &wgpu::Buffer {
        &self.skeys
    }

    /// `bucket_start`: M + 1 `u32` words at the start of this buffer (more
    /// words follow; they are scratch).
    pub fn bucket_start_buffer(&self) -> &wgpu::Buffer {
        &self.start
    }

    /// The key of each particle, in particle order (`cap` keys of two `u32`).
    pub fn particle_keys_buffer(&self) -> &wgpu::Buffer {
        &self.keys
    }

    /// The counts buffer: M bucket counts (spent after the pass), zeros up to
    /// word `info_offset_words()`, then the info words: number binned, number
    /// not binned, number out of range.
    pub fn info_buffer(&self) -> &wgpu::Buffer {
        &self.counts
    }

    /// Word offset of the info words in [`Self::info_buffer`].
    pub fn info_offset_words(&self) -> usize {
        self.mpad
    }

    /// Reads `words` `u32` of `buf` from word `offset` (blocking; for tests
    /// and tools).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_words(&self, buf: &wgpu::Buffer, offset: usize, words: usize) -> Vec<u32> {
        let d = self.gpu.device();
        let size = (4 * words).max(4) as u64;
        let stage = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = d.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, (4 * offset) as u64, &stage, 0, size);
        self.gpu.queue().submit([enc.finish()]);
        stage.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
        d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let v: Vec<u32> = {
            let view = stage.get_mapped_range(..).expect("range");
            view.as_chunks::<4>()
                .0
                .iter()
                .take(words)
                .map(|b| u32::from_le_bytes(*b))
                .collect()
        };
        stage.unmap();
        v
    }

    /// Reads the result of the last pass (blocking; for tests and tools).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_output(&self) -> BinOutput {
        let info = self.read_words(&self.counts, self.mpad, 3);
        let sorted = self.read_words(&self.sorted, 0, self.n);
        let sk = self.read_words(&self.skeys, 0, 2 * self.n);
        BinOutput {
            sorted,
            sorted_keys: sk.chunks(2).map(|c| [c[0], c[1]]).collect(),
            bucket_start: self.read_words(&self.start, 0, self.m + 1),
            n_binned: info[0],
            n_not_binned: info[1],
            n_out_of_range: info[2],
        }
    }

    /// Reads the key of each of the `n` particles (blocking).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_particle_keys(&self) -> Vec<[u32; 2]> {
        let w = self.read_words(&self.keys, 0, 2 * self.n);
        w.chunks(2).map(|c| [c[0], c[1]]).collect()
    }

    /// Runs the first `stages` passes in one submission and waits; returns the
    /// wall time from submit to the end (blocking; for tests and the bench).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_timed(&self, time_s: f64, stages: usize) -> std::time::Duration {
        let mut enc = self
            .gpu
            .device()
            .create_command_encoder(&Default::default());
        self.encode_stages(&mut enc, time_s, stages);
        let cmd = enc.finish();
        let t0 = std::time::Instant::now();
        self.gpu.queue().submit([cmd]);
        self.gpu
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        t0.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fring_model::rng::{philox4x32_10, u01_f64};

    const A0: f64 = 140_222.43;
    const N0: f64 = 1.1756e-4;

    fn geometry(domain_km: f64, phi0: f64) -> BinGeometry {
        BinGeometry::new(A0, N0, [0.05, 10.0, 0.04], domain_km, phi0).unwrap()
    }

    /// `n` uniform numbers in (0, 1) for (seed, stream).
    fn uniform(seed: u32, stream: u32, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let w = philox4x32_10([i as u32, stream, 0, 0], [seed, 0x4b35]);
                u01_f64(w[0], w[1])
            })
            .collect()
    }

    /// Inertial positions for local coordinates (x, y_c, z) at a time with
    /// the angle `alpha` (the reverse of the cell function, in f64).
    fn inertial(g: &BinGeometry, alpha: f64, local: [f64; 3]) -> [f64; 3] {
        let rho = g.a0_km + local[0];
        let phi = local[1] / g.a0_km + alpha;
        [rho * phi.cos(), rho * phi.sin(), local[2]]
    }

    #[test]
    fn key_packing_round_trips_and_orders_like_the_triple() {
        let cs = [
            Cell {
                ix: 0,
                iy: 0,
                iz: 0,
            },
            Cell {
                ix: -MAX_IX,
                iy: MAX_NY - 1,
                iz: MAX_IZ,
            },
            Cell {
                ix: MAX_IX,
                iy: 2047,
                iz: -MAX_IZ,
            },
            Cell {
                ix: -1,
                iy: 2048,
                iz: -1,
            },
            Cell {
                ix: 5,
                iy: 88_099,
                iz: 3,
            },
        ];
        for c in cs {
            assert_eq!(unpack_key(pack_key(c)), Some(c));
            assert_ne!(pack_key(c), NO_KEY);
        }
        for a in cs {
            for b in cs {
                assert_eq!(
                    a.cmp(&b),
                    (pack_key(a)[0], pack_key(a)[1]).cmp(&(pack_key(b)[0], pack_key(b)[1]))
                );
            }
        }
        assert_eq!(unpack_key(NO_KEY), None);
    }

    #[test]
    fn geometry_adjusts_ly_and_checks_its_arguments() {
        let g = BinGeometry::new(A0, N0, [0.05, 10.0, 0.04], TAU * A0, 0.0).unwrap();
        assert_eq!(g.ny, 88_104); // round(881,047.9 / 10)
        assert!((g.ly_km * f64::from(g.ny) - TAU * A0).abs() < 1e-9);
        let arc = BinGeometry::new(A0, N0, [0.05, 10.0, 0.04], 12_237.0, 0.1).unwrap();
        assert_eq!(arc.ny, 1224);
        assert!((arc.ly_km - 12_237.0 / 1224.0).abs() < 1e-12);
        // A domain smaller than one cell still has one cell.
        assert_eq!(
            BinGeometry::new(A0, N0, [1.0, 1000.0, 1.0], 100.0, 0.0)
                .unwrap()
                .ny,
            1
        );
        for bad in [
            BinGeometry::new(A0, N0, [0.0, 10.0, 0.04], 100.0, 0.0),
            BinGeometry::new(A0, N0, [0.05, f64::NAN, 0.04], 100.0, 0.0),
            BinGeometry::new(A0, N0, [0.05, 10.0, -1.0], 100.0, 0.0),
            BinGeometry::new(A0, N0, [0.05, 10.0, 0.04], 2.0 * TAU * A0, 0.0),
            BinGeometry::new(A0, N0, [0.05, 0.1, 0.04], TAU * A0, 0.0), // 8.8e6 cells
            BinGeometry::new(A0, N0, [0.05, 10.0, 0.04], 100.0, f64::INFINITY),
        ] {
            assert!(matches!(bad, Err(Error::Config(_))), "{bad:?}");
        }
    }

    #[test]
    fn geometry_from_the_config_uses_the_domain() {
        let text = include_str!("../../fring-model/tests/data/example-run.toml");
        let c = |d: Domain| {
            let mut cfg = RunConfig::from_toml_str(text).unwrap();
            cfg.collisions = Collisions::Stochastic {
                particle_radius_km: 1e-3,
                weight: 1.0,
                cell_radial_km: 0.05,
                cell_azimuthal_km: 10.0,
                cell_vertical_km: 0.04,
                every_steps: 1,
                restitution: fring_model::config::Restitution::Constant { epsilon: 0.5 },
            };
            cfg.domain = d;
            BinGeometry::from_config(&cfg, N0)
        };
        let full = c(Domain::FullRing).unwrap();
        assert_eq!(full.phi0_rad, 0.0);
        assert_eq!(full.a0_km, 140_220.0);
        assert_eq!(full.ny, (TAU * 140_220.0 / 10.0_f64).round() as u32);
        let arc = c(Domain::Arc {
            width_deg: 5.0,
            centre_longitude_deg: 30.0,
        })
        .unwrap();
        assert!((arc.phi0_rad - 27.5f64.to_radians()).abs() < 1e-15);
        assert!((arc.domain_km - 140_220.0 * 5.0f64.to_radians()).abs() < 1e-9);
        let mut cfg = RunConfig::from_toml_str(text).unwrap();
        cfg.collisions = Collisions::None;
        assert!(BinGeometry::from_config(&cfg, N0).is_err());
    }

    #[test]
    fn hash_has_known_values_and_spreads_neighbouring_cells() {
        // Values computed once from the definition; they pin the function that
        // the shader must reproduce.
        assert_eq!(hash_key([0, 0]), 0);
        let h = |a, b| hash_key([a, b]);
        assert_ne!(h(1, 0), h(0, 1));
        // Neighbouring cells along y fall into many different buckets.
        let m = 1usize << 12;
        let mut seen = std::collections::HashSet::new();
        for iy in 0..2048u32 {
            seen.insert((hash_key(pack_key(Cell { ix: 3, iy, iz: -2 })) as usize) & (m - 1));
        }
        assert!(seen.len() > 1500, "{} buckets for 2048 cells", seen.len());
    }

    /// Rule that links the DS twin (and so the GPU) to the f64 reference
    /// (A0 decision, K5-BIN):
    ///
    /// - A particle may get a different cell in the twin and in f64 only if
    ///   its f64 coordinate is within `delta` of a cell edge on an axis in
    ///   which the two cells differ (the edges of y include 0 and D).
    /// - `delta` = 10 × the largest |Δx|, |Δy|, |Δz| between the twin and f64
    ///   over the random cases, and `delta` ≤ 1e-6 km.
    ///
    /// The test prints the largest error, `delta` and the number of mismatches.
    #[test]
    fn twin_agrees_with_f64_up_to_the_edge_rule() {
        let n = 200_000;
        let mut worst = 0f64;
        let mut results = Vec::new();
        for (domain, phi0, t_s) in [
            (TAU * A0, 0.0, 0.0),
            (TAU * A0, 0.0, 1.0e8),
            (12_237.0, 0.3, 3.3e7),
        ] {
            let g = geometry(domain, phi0);
            let alpha = g.alpha_rad(t_s);
            let prm = params(&g);
            let (u0, u1, u2) = (uniform(1, 1, n), uniform(1, 2, n), uniform(1, 3, n));
            for i in 0..n {
                // x within +-100 km, y over the domain and beyond (arc: out of
                // range particles too), z within +-1 km: many cells at all
                // phases. Inertial position from the f64 local position.
                let local = [
                    200.0 * u0[i] - 100.0,
                    1.05 * domain * u1[i],
                    2.0 * u2[i] - 1.0,
                ];
                let p = inertial(&g, alpha, local);
                let (bf, lf) = cell_f64(&g, alpha, p);
                let ps = [Dsm::split(p[0]), Dsm::split(p[1]), Dsm::split(p[2])];
                let (bd, ld) = cell_ds(&prm, g.ny, Dsm::split(alpha), ps);
                if bf == Binned::OutOfRange || bd == Binned::OutOfRange {
                    // At the ends of the arc only: the same rule on y.
                    if bf != bd {
                        let dy = edge_distance_km(&g, lf)[1].min((lf[1] - g.domain_km).abs());
                        results.push((dy, 0.0));
                    }
                    continue;
                }
                assert!(matches!((bf, bd), (Binned::Cell(_), Binned::Cell(_))));
                for a in 0..3 {
                    let mut e = (ld[a] - lf[a]).abs();
                    if a == 1 {
                        // The longitude of the twin may differ by a whole
                        // turn at the wrap (y near 0 or D).
                        e = e.min((e - g.domain_km).abs());
                    }
                    worst = worst.max(e);
                }
                if bf != bd {
                    let (Binned::Cell(cf), Binned::Cell(cd)) = (bf, bd) else {
                        unreachable!()
                    };
                    let dist = edge_distance_km(&g, lf);
                    let mut near = f64::INFINITY;
                    if cf.ix != cd.ix {
                        near = near.min(dist[0]);
                    }
                    if cf.iy != cd.iy {
                        near = near.min(dist[1]);
                    }
                    if cf.iz != cd.iz {
                        near = near.min(dist[2]);
                    }
                    results.push((near, 0.0));
                }
            }
        }
        let delta = 10.0 * worst;
        let mismatches = results.len();
        let far = results.iter().filter(|r| r.0 > delta).count();
        eprintln!(
            "K5-BIN twin vs f64: largest |dx|,|dy|,|dz| = {worst:.3e} km, delta = {delta:.3e} km, \
             cell mismatches = {mismatches} of {}, outside delta = {far}",
            3 * n
        );
        assert!(delta <= 1.0e-6, "delta {delta}");
        assert_eq!(far, 0, "a mismatch farther than delta from an edge");
    }

    /// Particles placed on a cell edge plus an offset of 1e-9 to 1e-5 km, on
    /// each axis, at several phases: the twin may differ from f64 only for an
    /// offset within `delta` (the rule of the test above, with delta = 1e-7 km,
    /// 10 times the largest error measured there), and must agree for 1e-6 km
    /// and more.
    #[test]
    fn particles_next_to_an_edge_obey_the_rule() {
        let delta = 1.0e-7;
        let offsets = [
            -1e-5, -1e-6, -1e-7, -1e-8, -1e-9, 0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
        ];
        let (mut mismatches, mut total) = (0usize, 0usize);
        for (domain, phi0, t_s) in [
            (TAU * A0, 0.0, 0.0),
            (TAU * A0, 0.0, 7.0e7),
            (12_237.0, 0.3, 1e6),
        ] {
            let g = geometry(domain, phi0);
            let alpha = g.alpha_rad(t_s);
            let prm = params(&g);
            let u = uniform(5, 1, 3 * 200);
            for axis in 0..3 {
                for (j, &off) in offsets.iter().enumerate() {
                    for r in 0..200 {
                        let (ix, iy, iz) = (
                            (u[3 * r] * 400.0) as i32 - 200,
                            (u[3 * r + 1] * f64::from(g.ny - 2)) as u32 + 1,
                            (u[3 * r + 2] * 200.0) as i32 - 100,
                        );
                        let mut local = [
                            (f64::from(ix) + 0.5) * g.lx_km,
                            (f64::from(iy) + 0.5) * g.ly_km,
                            (f64::from(iz) + 0.5) * g.lz_km,
                        ];
                        let size = [g.lx_km, g.ly_km, g.lz_km][axis];
                        local[axis] -= 0.5 * size;
                        local[axis] += off;
                        let p = inertial(&g, alpha, local);
                        let (bf, lf) = cell_f64(&g, alpha, p);
                        let ps = [Dsm::split(p[0]), Dsm::split(p[1]), Dsm::split(p[2])];
                        let (bd, _) = cell_ds(&prm, g.ny, Dsm::split(alpha), ps);
                        total += 1;
                        if bf != bd {
                            mismatches += 1;
                            let near = edge_distance_km(&g, lf)[axis];
                            assert!(
                                near <= delta,
                                "axis {axis} offset {off}: edge distance {near}"
                            );
                        }
                        if off.abs() >= 1e-6 {
                            assert_eq!(bf, bd, "axis {axis} offset {off} case {j}");
                        }
                    }
                }
            }
        }
        eprintln!("K5-BIN edge cases: {mismatches} cell mismatches of {total}, all within delta");
    }

    /// Cases where the cell is easy to get wrong.
    #[test]
    fn edge_cases_of_the_cell() {
        let g = geometry(TAU * A0, 0.0);
        let prm = params(&g);
        let ny = g.ny;
        let check = |alpha: f64, p: [f64; 3]| {
            let (bf, _) = cell_f64(&g, alpha, p);
            let ps = [Dsm::split(p[0]), Dsm::split(p[1]), Dsm::split(p[2])];
            let (bd, _) = cell_ds(&prm, ny, Dsm::split(alpha), ps);
            (bf, bd)
        };
        // Mid-cell points at the four sides of the ring and at the wrap: the
        // two functions agree, with the expected cell.
        let ly = g.ly_km;
        for (phi, want_iy) in [
            (0.5 * ly / A0, 0u32),
            (-0.5 * ly / A0, ny - 1),
            (PI / 2.0 + 0.5 * ly / A0, ny / 4),
            (PI - 0.5 * ly / A0, ny / 2 - 1),
            (-PI + 0.5 * ly / A0, ny / 2),
            (3.0 * PI / 2.0 + 0.5 * ly / A0, (3 * ny) / 4),
        ] {
            let rho = A0 + 0.0123;
            let p = [rho * phi.cos(), rho * phi.sin(), -0.0123];
            let (bf, bd) = check(0.0, p);
            assert_eq!(bf, bd, "phi {phi}");
            let Binned::Cell(c) = bf else { panic!() };
            assert_eq!(c.iy, want_iy, "phi {phi}");
            assert_eq!((c.ix, c.iz), (0, -1)); // x = 12.3 m, z = -12.3 m
        }
        // Negative x and z: indices -1 and below.
        let (bf, bd) = check(0.0, [A0 - 0.025, 0.0 + 1.0, -0.0001]);
        assert_eq!(bf, bd);
        let Binned::Cell(c) = bf else { panic!() };
        assert_eq!((c.ix, c.iz), (-1, -1));
        // The rotating frame: at alpha the particle at inertial longitude alpha
        // is at y = 0 (cell 0) and one just below alpha is in cell n_y - 1.
        for alpha in [0.7, 3.0, 6.0, 1.0e-9] {
            for (dphi, want) in [(0.5 * ly / A0, 0), (-0.5 * ly / A0, ny - 1)] {
                let phi: f64 = alpha + dphi;
                let rho = A0 + 0.0123;
                let p = [rho * phi.cos(), rho * phi.sin(), 0.0123];
                let (bf, bd) = check(alpha, p);
                assert_eq!(bf, bd);
                assert_eq!(
                    bf,
                    Binned::Cell(Cell {
                        ix: 0,
                        iy: want,
                        iz: 0
                    })
                );
            }
        }
        // A bad state: the origin; a huge distance.
        assert_eq!(
            check(0.0, [0.0, 0.0, 0.0]),
            (Binned::NotBinned, Binned::NotBinned)
        );
        // Out of range: x beyond the key.
        let far = A0 + 1.0e5 * 0.05 * 41.0; // ix = 4.1e6 > 2^21
        let (bf, bd) = check(0.0, [far, 0.0, 0.0]);
        assert_eq!((bf, bd), (Binned::OutOfRange, Binned::OutOfRange));
        // The arc: a particle behind the end of the arc is out of range.
        let arc = geometry(12_237.0, 0.3);
        let arc_prm = params(&arc);
        let alpha = arc.alpha_rad(0.0);
        for (dy_km, want_in) in [(5_000.0, true), (11_000.0, true), (13_000.0, false)] {
            let phi = alpha + dy_km / A0;
            let rho = A0 + 0.0123;
            let p = [rho * phi.cos(), rho * phi.sin(), 0.0123];
            let (bf, _) = cell_f64(&arc, alpha, p);
            let ps = [Dsm::split(p[0]), Dsm::split(p[1]), Dsm::split(p[2])];
            let (bd, _) = cell_ds(&arc_prm, arc.ny, Dsm::split(alpha), ps);
            assert_eq!(bf, bd);
            assert_eq!(matches!(bf, Binned::Cell(_)), want_in, "dy {dy_km}");
        }
    }

    fn random_cells(n: usize, seed: u32, spread: u32) -> Vec<Binned> {
        let (a, b, c) = (
            uniform(seed, 1, n),
            uniform(seed, 2, n),
            uniform(seed, 3, n),
        );
        (0..n)
            .map(|i| {
                if i % 97 == 5 {
                    return Binned::NotBinned;
                }
                if i % 89 == 7 {
                    return Binned::OutOfRange;
                }
                Binned::Cell(Cell {
                    ix: (a[i] * f64::from(spread)) as i32 - (spread as i32) / 2,
                    iy: (b[i] * f64::from(spread)) as u32,
                    iz: (c[i] * 3.0) as i32 - 1,
                })
            })
            .collect()
    }

    #[test]
    fn counting_sort_twin_agrees_with_the_std_sort() {
        // Few cells (many per cell), many cells, and a tiny number of buckets
        // that forces different cells into one bucket.
        for (n, spread, m) in [
            (10_000usize, 6u32, 16384usize),
            (10_000, 400, 16384),
            (10_000, 400, 4),
            (10_000, 400, 1),
            (1, 3, 1),
            (7, 3, 8),
        ] {
            let cells = random_cells(n, 9, spread);
            let out = bin_twin(&cells, m);
            let want = cells_reference(&cells);
            let mut got = runs(&out);
            got.sort_by_key(|r| pack_key(r.0));
            assert_eq!(got, want, "n {n} spread {spread} m {m}");
            assert_eq!(
                out.n_binned as usize,
                cells
                    .iter()
                    .filter(|b| matches!(b, Binned::Cell(_)))
                    .count()
            );
            assert_eq!(out.bucket_start[m], out.n_binned);
            assert_eq!(
                out.n_not_binned as usize,
                cells.iter().filter(|b| **b == Binned::NotBinned).count()
            );
            assert!(
                out.sorted[out.n_binned as usize..]
                    .iter()
                    .all(|&i| i == NO_ID)
            );
            // Inside a bucket the order is (key, id).
            for b in 0..m {
                let (s, e) = (
                    out.bucket_start[b] as usize,
                    out.bucket_start[b + 1] as usize,
                );
                let w: Vec<_> = (s..e)
                    .map(|p| (out.sorted_keys[p], out.sorted[p]))
                    .collect();
                assert!(w.windows(2).all(|x| x[0] < x[1]));
                assert!(w.iter().all(|&(k, _)| hash_key(k) as usize & (m - 1) == b));
            }
        }
    }

    #[test]
    fn counting_sort_edge_cases() {
        // No particle binned; none at all; all in one cell (ids in order).
        let none = vec![Binned::NotBinned; 5];
        let o = bin_twin(&none, 8);
        assert_eq!((o.n_binned, o.n_not_binned), (0, 5));
        assert!(runs(&o).is_empty());
        let o = bin_twin(&[], 1);
        assert_eq!(o.n_binned, 0);
        let one = vec![
            Binned::Cell(Cell {
                ix: -2,
                iy: 9,
                iz: 4
            });
            5000
        ];
        let o = bin_twin(&one, 8192);
        let r = runs(&o);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].1, (0..5000).collect::<Vec<u32>>());
    }
}
