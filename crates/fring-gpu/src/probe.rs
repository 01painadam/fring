//! Precision probe of the GPU (task G1): does double-single arithmetic survive
//! the WGSL to Metal compilation, is `fma` fused, and how large are the errors
//! of division, square root, sine and cosine?
//!
//! The kernels are in `shaders/probe.wgsl` (with `shaders/ds.wgsl`). Every
//! check compares the GPU result with an exact or f64 value on the host.

use crate::gpu::{Gpu, from_words, to_words};
use fring_model::rng::{Counter, Philox, Stream};

/// The WGSL source of the probe: the DS library and the probe kernels.
pub const PROBE_WGSL: &str = concat!(
    include_str!("../shaders/ds.wgsl"),
    include_str!("../shaders/probe.wgsl")
);

/// Number of test values per check.
pub const N_PROBE: usize = 1 << 20;

/// Results of the probe. Counts are out of [`N_PROBE`] values.
#[derive(Clone, Debug, Default)]
pub struct ProbeReport {
    /// Adapter name and backend.
    pub adapter: String,
    /// `a + b` differs from the correctly rounded sum.
    pub add_not_rounded: usize,
    /// TwoSum: `s + e` differs from the exact `a + b`.
    pub two_sum_fail: usize,
    /// Fast2Sum (ordered inputs): `s + e` differs from the exact sum.
    pub fast_two_sum_fail: usize,
    /// TwoSum with the guard work-around: `s + e` differs from the exact sum.
    pub two_sum_guarded_fail: usize,
    /// Fast2Sum with the guard work-around: `s + e` differs from the exact sum.
    pub fast_two_sum_guarded_fail: usize,
    /// TwoSum as a guarded Fast2Sum after ordering by magnitude: failures.
    pub two_sum_select_fail: usize,
    /// Guarded DS addition: maximum relative error in units of 2^-48.
    pub ds_add_guarded_max_rel_u2: f64,
    /// Guarded DS multiplication: maximum relative error in units of 2^-48.
    pub ds_mul_guarded_max_rel_u2: f64,
    /// `a * b` differs from the correctly rounded product.
    pub mul_not_rounded: usize,
    /// `fma(a, b, -p)` differs from the exact error `a b - p` (so not fused).
    pub fma_not_exact: usize,
    /// A plain `a * b - p` equals the exact error and is not zero: the
    /// compiler fused it into an fma.
    pub plain_contracted: usize,
    /// Values with a non-zero exact product error (the cases that count).
    pub nonzero_product_error: usize,
    /// Dekker's product: error term differs from the exact error.
    pub dekker_fail: usize,
    /// Maximum error of `a / b` in units in the last place (ULP).
    pub div_max_ulp: f64,
    /// Fraction of quotients that are not correctly rounded.
    pub div_not_rounded: f64,
    /// Maximum error of `1 / b` in ULP.
    pub recip_max_ulp: f64,
    /// Maximum error of `sqrt` in ULP.
    pub sqrt_max_ulp: f64,
    /// Fraction of square roots that are not correctly rounded.
    pub sqrt_not_rounded: f64,
    /// Maximum error of `inverseSqrt` in ULP.
    pub inverse_sqrt_max_ulp: f64,
    /// Maximum absolute error of built-in `sin` and `cos` on [-pi, pi].
    pub trig_max_abs: f64,
    /// Maximum relative error of built-in `sin` on [-0.1, 0.1].
    pub sin_small_max_rel: f64,
    /// Maximum relative error of built-in `cos` on [-0.1, 0.1].
    pub cos_small_max_rel: f64,
    /// Maximum relative error of DS addition, in units of 2^-48.
    pub ds_add_max_rel_u2: f64,
    /// Maximum relative error of DS multiplication, in units of 2^-48.
    pub ds_mul_max_rel_u2: f64,
    /// Maximum relative error of DS division, in units of 2^-48.
    pub ds_div_max_rel_u2: f64,
}

impl ProbeReport {
    /// True if double-single arithmetic works on this GPU: TwoSum, Fast2Sum
    /// and the fma product error are exact for every test value, and + and *
    /// are correctly rounded.
    pub fn double_single_works(&self) -> bool {
        self.add_not_rounded == 0
            && self.mul_not_rounded == 0
            && self.two_sum_fail == 0
            && self.fast_two_sum_fail == 0
            && self.fma_not_exact == 0
    }

    /// True if the guarded DS primitives (sums multiplied by a runtime 1.0
    /// from a uniform buffer, so that the compiler cannot re-associate them)
    /// are exact for every test value.
    pub fn guarded_double_single_works(&self) -> bool {
        self.add_not_rounded == 0
            && self.mul_not_rounded == 0
            && self.two_sum_guarded_fail == 0
            && self.fast_two_sum_guarded_fail == 0
            && self.two_sum_select_fail == 0
            && self.fma_not_exact == 0
    }
}

/// One unit in the last place of the f32 nearest to `x` (normal range).
pub fn ulp_f32(x: f64) -> f64 {
    let e = (x.abs() as f32).to_bits() >> 23 & 0xff;
    2f64.powi(e as i32 - 127 - 23)
}

/// Error of `got` against the exact value `exact`, in ULP of f32.
pub fn ulp_error(got: f32, exact: f64) -> f64 {
    (got as f64 - exact).abs() / ulp_f32(exact)
}

/// A random f32 with a random sign, random mantissa and exponent in
/// `emin..=emax`.
fn rand_f32(w0: u32, w1: u32, emin: i32, emax: i32) -> f32 {
    let span = (emax - emin + 1) as u32;
    let e = emin + (w1 % span) as i32;
    let sign = w1 >> 31;
    f32::from_bits(sign << 31 | ((e + 127) as u32) << 23 | (w0 & 0x7f_ffff))
}

/// The inputs: four f32 per value, from the test stream of the Philox RNG.
/// `x`, `y` are random f32 with exponents in [-5, 18]; `z` is uniform in
/// [-0.1, 0.1]; `w` is unused (0).
fn inputs_basic() -> Vec<f32> {
    let rng = Philox::from_seed(0x6731);
    let mut v = Vec::with_capacity(4 * N_PROBE);
    for i in 0..N_PROBE {
        let c = |draw| Counter {
            index: i as u32,
            step: 0,
            stream: Stream::TEST,
            draw,
        };
        let w = rng.block(c(0));
        let u = rng.uniform_f32(c(1));
        v.push(rand_f32(w[0], w[1], -5, 18));
        v.push(rand_f32(w[2], w[3], -5, 18));
        v.push(0.2 * u[0] - 0.1);
        v.push(0.0);
    }
    v
}

/// Angles for the built-in sine and cosine: x uniform in [-pi, pi], z uniform
/// in [-0.1, 0.1].
fn inputs_trig() -> Vec<f32> {
    let rng = Philox::from_seed(0x6732);
    let mut v = Vec::with_capacity(4 * N_PROBE);
    for i in 0..N_PROBE {
        let u = rng.uniform_f64(Counter {
            index: i as u32,
            step: 0,
            stream: Stream::TEST,
            draw: 0,
        });
        v.push(((2.0 * u[0] - 1.0) * std::f64::consts::PI) as f32);
        v.push(0.0);
        v.push((0.2 * u[1] - 0.1) as f32);
        v.push(0.0);
    }
    v
}

/// Two DS numbers per value: (x, y) and (z, w), each a normalised hi + lo of
/// a random f64 with exponent in [-5, 18].
fn inputs_ds() -> Vec<f32> {
    let rng = Philox::from_seed(0x6733);
    let mut v = Vec::with_capacity(4 * N_PROBE);
    for i in 0..N_PROBE {
        let u = rng.uniform_f64(Counter {
            index: i as u32,
            step: 0,
            stream: Stream::TEST,
            draw: 0,
        });
        let w = rng.block(Counter {
            index: i as u32,
            step: 0,
            stream: Stream::TEST,
            draw: 1,
        });
        for (k, uu) in u.iter().enumerate() {
            let e = -5 + (w[k] % 24) as i32;
            let sign = if w[k + 2] & 1 == 1 { -1.0 } else { 1.0 };
            let x = sign * (1.0 + uu) * 2f64.powi(e);
            let hi = x as f32;
            let lo = (x - hi as f64) as f32;
            v.push(hi);
            v.push(lo);
        }
    }
    v
}

fn run(gpu: &Gpu, entry: &str, input: &[f32]) -> Vec<f32> {
    let n = input.len() / 4;
    let mut bufs = vec![to_words(input), vec![0u32; input.len()]];
    gpu.run(PROBE_WGSL, entry, &mut bufs, None, n.div_ceil(64) as u32, 1);
    from_words(&bufs[1])
}

/// Runs every probe kernel on `gpu` and compares with the host.
pub fn run_probe(gpu: &Gpu) -> ProbeReport {
    let mut r = ProbeReport {
        adapter: format!(
            "{} ({:?}, driver {} {})",
            gpu.info.name, gpu.info.backend, gpu.info.driver, gpu.info.driver_info
        ),
        ..Default::default()
    };
    let basic = inputs_basic();

    // Sums.
    let out = run(gpu, "probe_sum", &basic);
    for i in 0..N_PROBE {
        let (a, b) = (basic[4 * i] as f64, basic[4 * i + 1] as f64);
        let exact = a + b; // exact: exponents differ by at most 23
        let o = &out[4 * i..4 * i + 4];
        if o[0] != exact as f32 {
            r.add_not_rounded += 1;
        }
        if o[0] as f64 + o[1] as f64 != exact {
            r.two_sum_fail += 1;
        }
        if o[2] as f64 + o[3] as f64 != exact {
            r.fast_two_sum_fail += 1;
        }
    }

    // Sums with the guard work-around (a runtime 1.0 from a uniform buffer).
    let n = basic.len() / 4;
    let mut bufs = vec![to_words(&basic), vec![0u32; basic.len()]];
    let guard = [1f32.to_bits(), 0, 0, 0];
    gpu.run(
        PROBE_WGSL,
        "probe_sum_g",
        &mut bufs,
        Some(&guard),
        n.div_ceil(64) as u32,
        1,
    );
    let out = from_words(&bufs[1]);
    for i in 0..N_PROBE {
        let (a, b) = (basic[4 * i] as f64, basic[4 * i + 1] as f64);
        let exact = a + b;
        let o = &out[4 * i..4 * i + 4];
        if o[0] as f64 + o[1] as f64 != exact {
            r.two_sum_guarded_fail += 1;
        }
        if o[2] as f64 + o[3] as f64 != exact {
            r.fast_two_sum_guarded_fail += 1;
        }
    }

    let mut bufs = vec![to_words(&basic), vec![0u32; basic.len()]];
    gpu.run(
        PROBE_WGSL,
        "probe_sum_sel",
        &mut bufs,
        Some(&guard),
        n.div_ceil(64) as u32,
        1,
    );
    let out = from_words(&bufs[1]);
    for i in 0..N_PROBE {
        let exact = basic[4 * i] as f64 + basic[4 * i + 1] as f64;
        if out[4 * i] as f64 + out[4 * i + 1] as f64 != exact {
            r.two_sum_select_fail += 1;
        }
    }

    // Products.
    let out = run(gpu, "probe_prod", &basic);
    for i in 0..N_PROBE {
        let (a, b) = (basic[4 * i] as f64, basic[4 * i + 1] as f64);
        let exact = a * b; // exact: 48 bits
        let o = &out[4 * i..4 * i + 4];
        let p = o[0] as f64;
        if o[0] != exact as f32 {
            r.mul_not_rounded += 1;
        }
        let err = exact - p;
        if err != 0.0 {
            r.nonzero_product_error += 1;
            if o[2] as f64 == err {
                r.plain_contracted += 1;
            }
        }
        if o[1] as f64 != err {
            r.fma_not_exact += 1;
        }
        if o[3] as f64 != err {
            r.dekker_fail += 1;
        }
    }

    // Division and square roots.
    let out = run(gpu, "probe_divsqrt", &basic);
    let (mut div_nr, mut sqrt_nr) = (0usize, 0usize);
    for i in 0..N_PROBE {
        let (a, b) = (basic[4 * i] as f64, basic[4 * i + 1] as f64);
        let o = &out[4 * i..4 * i + 4];
        let q = a / b;
        r.div_max_ulp = r.div_max_ulp.max(ulp_error(o[0], q));
        if o[0] != q as f32 {
            div_nr += 1;
        }
        let s = a.abs().sqrt();
        r.sqrt_max_ulp = r.sqrt_max_ulp.max(ulp_error(o[1], s));
        if o[1] != s as f32 {
            sqrt_nr += 1;
        }
        r.inverse_sqrt_max_ulp = r.inverse_sqrt_max_ulp.max(ulp_error(o[2], 1.0 / s));
        r.recip_max_ulp = r.recip_max_ulp.max(ulp_error(o[3], 1.0 / b));
    }
    r.div_not_rounded = div_nr as f64 / N_PROBE as f64;
    r.sqrt_not_rounded = sqrt_nr as f64 / N_PROBE as f64;

    // Built-in sine and cosine.
    let trig = inputs_trig();
    let out = run(gpu, "probe_trig", &trig);
    for i in 0..N_PROBE {
        let (x, z) = (trig[4 * i] as f64, trig[4 * i + 2] as f64);
        let o = &out[4 * i..4 * i + 4];
        r.trig_max_abs = r
            .trig_max_abs
            .max((o[0] as f64 - x.sin()).abs())
            .max((o[1] as f64 - x.cos()).abs());
        if z != 0.0 {
            r.sin_small_max_rel = r
                .sin_small_max_rel
                .max((o[2] as f64 - z.sin()).abs() / z.sin().abs());
        }
        r.cos_small_max_rel = r
            .cos_small_max_rel
            .max((o[3] as f64 - z.cos()).abs() / z.cos());
    }

    // Double-single arithmetic, against f64 and against the Rust twin.
    let dsin = inputs_ds();
    let out = run(gpu, "probe_ds", &dsin);
    let outq = run(gpu, "probe_dsdiv", &dsin);
    let mut bufs = vec![to_words(&dsin), vec![0u32; dsin.len()]];
    let guard = [1f32.to_bits(), 0, 0, 0];
    gpu.run(
        PROBE_WGSL,
        "probe_ds_g",
        &mut bufs,
        Some(&guard),
        (dsin.len() / 4).div_ceil(64) as u32,
        1,
    );
    let outg = from_words(&bufs[1]);
    let u2 = 2f64.powi(-48);
    for i in 0..N_PROBE {
        let d = &dsin[4 * i..4 * i + 4];
        let a = d[0] as f64 + d[1] as f64;
        let b = d[2] as f64 + d[3] as f64;
        let o = &out[4 * i..4 * i + 4];
        let rel = |got: f64, exact: f64| (got - exact).abs() / exact.abs() / u2;
        let s = a + b;
        if s != 0.0 {
            r.ds_add_max_rel_u2 = r.ds_add_max_rel_u2.max(rel(o[0] as f64 + o[1] as f64, s));
        }
        r.ds_mul_max_rel_u2 = r
            .ds_mul_max_rel_u2
            .max(rel(o[2] as f64 + o[3] as f64, a * b));
        let g = &outg[4 * i..4 * i + 4];
        if s != 0.0 {
            r.ds_add_guarded_max_rel_u2 = r
                .ds_add_guarded_max_rel_u2
                .max(rel(g[0] as f64 + g[1] as f64, s));
        }
        r.ds_mul_guarded_max_rel_u2 = r
            .ds_mul_guarded_max_rel_u2
            .max(rel(g[2] as f64 + g[3] as f64, a * b));
        let q = &outq[4 * i..4 * i + 2];
        r.ds_div_max_rel_u2 = r
            .ds_div_max_rel_u2
            .max(rel(q[0] as f64 + q[1] as f64, a / b));
    }
    r
}
