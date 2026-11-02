//! Precision probe of the GPU (task G1, ADR 0006): does double-single (DS)
//! arithmetic survive the WGSL to Metal compilation, is `fma` fused, how
//! large are the errors of division, square root, sine and cosine, and do
//! the shaders give the same bits as their Rust twins?
//!
//! - [`run_probe`]: the full report (`examples/g1_probe.rs`).
//! - [`self_test`]: the short check that every GPU code path with DS must
//!   run at start-up. It fails closed with [`Error::Backend`].
//!
//! The kernels are in `shaders/probe.wgsl` (with `ds.wgsl` and `trig.wgsl`).

use crate::ds::Ds;
use crate::gpu::Gpu;
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::{from_words, to_words};
#[cfg(not(target_arch = "wasm32"))]
use crate::wf32::{model, sin_cos_f32};
use fring_model::rng::{Counter, Philox, Stream};
use fring_model::{Error, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// The WGSL source of the probe: the DS and trig libraries and the kernels.
pub const PROBE_WGSL: &str = concat!(
    include_str!("../shaders/ds.wgsl"),
    include_str!("../shaders/trig.wgsl"),
    include_str!("../shaders/probe.wgsl")
);

/// Number of test values per check in [`run_probe`].
pub const N_PROBE: usize = 1 << 20;
/// Number of test values per check in [`self_test`].
pub const N_SELF_TEST: usize = 1 << 16;

/// u² = 2⁻⁴⁸, the unit of the DS error fields.
const U2: f64 = 3.552_713_678_800_501e-15;

/// Results of the probe. Counts are out of the number of test values.
#[derive(Clone, Debug, Default)]
pub struct ProbeReport {
    /// Adapter name and backend.
    pub adapter: String,
    /// Test values per check.
    pub n: usize,
    /// `a + b` differs from the correctly rounded sum.
    pub add_not_rounded: usize,
    /// `a * b` differs from the correctly rounded product.
    pub mul_not_rounded: usize,
    /// Plain TwoSum (no guard): `s + e` differs from the exact sum.
    pub plain_two_sum_fail: usize,
    /// Plain Fast2Sum (no guard): `s + e` differs from the exact sum.
    pub plain_fast_two_sum_fail: usize,
    /// Plain Dekker product (Veltkamp split, no fma): error term wrong.
    pub plain_dekker_fail: usize,
    /// Plain DS addition: maximum relative error in units of u² = 2⁻⁴⁸.
    pub plain_ds_add_max_u2: f64,
    /// Plain DS multiplication: maximum relative error in units of u².
    pub plain_ds_mul_max_u2: f64,
    /// Guarded TwoSum: `s + e` differs from the exact sum.
    pub two_sum_fail: usize,
    /// Guarded Fast2Sum (after a magnitude sort): failures.
    pub fast_two_sum_fail: usize,
    /// `fma(a, b, -p)` differs from the exact product error (so not fused).
    pub fma_not_exact: usize,
    /// Values with a non-zero product error.
    pub nonzero_product_error: usize,
    /// A plain `a * b - p` equals the exact (non-zero) error: the compiler
    /// fused it into an fma.
    pub plain_contracted: usize,
    /// Guarded DS addition: maximum relative error in u² (bound 3).
    pub ds_add_max_u2: f64,
    /// Guarded DS multiplication: maximum relative error in u² (bound 5).
    pub ds_mul_max_u2: f64,
    /// DS division: maximum relative error in u².
    pub ds_div_max_u2: f64,
    /// DS square root: maximum relative error in u².
    pub ds_sqrt_max_u2: f64,
    /// DS small-angle sine: maximum relative error in u².
    pub ds_sin_max_u2: f64,
    /// DS small-angle 1 - cosine: maximum relative error in u².
    pub ds_omc_max_u2: f64,
    /// DS add and mul results whose bits differ from the Rust twin [`Ds`].
    pub ds_twin_bit_differences: usize,
    /// DS sine and 1 - cosine results whose bits differ from the Rust twin.
    pub ds_trig_twin_bit_differences: usize,
    /// Maximum error of `a / b` in ULP.
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
    /// Built-in `sin` and `cos` on [-π, π]: maximum absolute error.
    pub builtin_trig_max_abs: f64,
    /// Own `sin_cos_f32` on [-π, π]: maximum absolute error.
    pub own_trig_max_abs: f64,
    /// Own `sin_cos_f32`: results whose bits differ from the Rust twin.
    pub own_trig_twin_bit_differences: usize,
}

impl ProbeReport {
    /// True if the guarded DS primitives work on this GPU: + and * correctly
    /// rounded, fma fused, guarded TwoSum and Fast2Sum exact, DS add and mul
    /// within their bounds (3u² and 5u²).
    pub fn double_single_works(&self) -> bool {
        self.failures().is_empty()
    }

    /// The failed conditions of [`Self::double_single_works`], in words.
    pub fn failures(&self) -> Vec<String> {
        let mut f = Vec::new();
        let mut need = |ok: bool, what: String| {
            if !ok {
                f.push(what);
            }
        };
        need(
            self.add_not_rounded == 0,
            format!("{} sums not correctly rounded", self.add_not_rounded),
        );
        need(
            self.mul_not_rounded == 0,
            format!("{} products not correctly rounded", self.mul_not_rounded),
        );
        need(
            self.fma_not_exact == 0,
            format!("fma not fused for {} values", self.fma_not_exact),
        );
        need(
            self.two_sum_fail == 0,
            format!("guarded TwoSum wrong for {} values", self.two_sum_fail),
        );
        need(
            self.fast_two_sum_fail == 0,
            format!(
                "guarded Fast2Sum wrong for {} values",
                self.fast_two_sum_fail
            ),
        );
        need(
            self.ds_add_max_u2 <= 3.0,
            format!("DS add error {:.2} u² > 3 u²", self.ds_add_max_u2),
        );
        need(
            self.ds_mul_max_u2 <= 5.0,
            format!("DS mul error {:.2} u² > 5 u²", self.ds_mul_max_u2),
        );
        f
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

fn counter(i: usize, draw: u16) -> Counter {
    Counter {
        index: i as u32,
        step: 0,
        stream: Stream::TEST,
        draw,
    }
}

/// A random f32 with a random sign, mantissa and exponent in [-5, 18].
fn rand_f32(w0: u32, w1: u32) -> f32 {
    let e = -5 + (w1 % 24) as i32;
    f32::from_bits((w1 >> 31) << 31 | ((e + 127) as u32) << 23 | (w0 & 0x7f_ffff))
}

/// Pairs of random f32 (x, y); z and w are 0. The exact sum and product of
/// two such values fit in f64.
fn inputs_f32(n: usize) -> Vec<f32> {
    let rng = Philox::from_seed(0x6731);
    let mut v = Vec::with_capacity(4 * n);
    for i in 0..n {
        let w = rng.block(counter(i, 0));
        v.extend([rand_f32(w[0], w[1]), rand_f32(w[2], w[3]), 0.0, 0.0]);
    }
    v
}

/// Two DS numbers per value, (x, y) and (z, w), each the nearest DS of a
/// random f64 with exponent in [-5, 18].
fn inputs_ds(n: usize) -> Vec<f32> {
    let rng = Philox::from_seed(0x6733);
    let mut v = Vec::with_capacity(4 * n);
    for i in 0..n {
        let u = rng.uniform_f64(counter(i, 0));
        let w = rng.block(counter(i, 1));
        for k in 0..2 {
            let e = -5 + (w[k] % 24) as i32;
            let sign = if w[k + 2] & 1 == 1 { -1.0 } else { 1.0 };
            let d = Ds::<0>::split(sign * (1.0 + u[k]) * 2f64.powi(e));
            v.extend([d.hi, d.lo]);
        }
    }
    v
}

/// Angles: x uniform in [-π, π] (f32), and a DS angle (z, w) uniform in
/// [-0.25, 0.25]; y is 0.
#[cfg(not(target_arch = "wasm32"))]
fn inputs_angles(n: usize) -> Vec<f32> {
    let rng = Philox::from_seed(0x6732);
    let mut v = Vec::with_capacity(4 * n);
    for i in 0..n {
        let u = rng.uniform_f64(counter(i, 0));
        let d = Ds::<0>::split(0.5 * u[1] - 0.25);
        v.extend([
            ((2.0 * u[0] - 1.0) * std::f64::consts::PI) as f32,
            0.0,
            d.hi,
            d.lo,
        ]);
    }
    v
}

/// Runs `entry` on `input` (four f32 per value); the guard uniform is
/// bound when `guarded`. Blocks (native only).
#[cfg(not(target_arch = "wasm32"))]
fn run(gpu: &Gpu, entry: &str, input: &[f32], guarded: bool) -> Vec<f32> {
    let n = input.len() / 4;
    let mut bufs = vec![to_words(input), vec![0u32; input.len()]];
    let guard = [1f32.to_bits(), 0, 0, 0];
    gpu.run(
        PROBE_WGSL,
        entry,
        &mut bufs,
        guarded.then_some(&guard[..]),
        n.div_ceil(64) as u32,
        1,
    );
    from_words(&bufs[1])
}

fn rel_u2(got: (f32, f32), exact: f64) -> f64 {
    (got.0 as f64 + got.1 as f64 - exact).abs() / exact.abs() / U2
}

/// The checks of [`self_test`] (and part of [`run_probe`]) on `n` values.
#[cfg(not(target_arch = "wasm32"))]
fn core_checks(gpu: &Gpu, n: usize, r: &mut ProbeReport) {
    let basic = inputs_f32(n);
    let out_sum = run(gpu, "probe_sum", &basic, true);
    let out_prod = run(gpu, "probe_prod", &basic, false);
    let dsin = inputs_ds(n);
    let out_ds = run(gpu, "probe_ds", &dsin, true);
    check_outputs(n, &basic, &out_sum, &out_prod, &dsin, &out_ds, r);
}

/// The comparisons of the core checks: GPU outputs against exact values and
/// the Rust twins. `basic` and `dsin` are the inputs (four f32 per value);
/// `out_*` are the outputs of `probe_sum`, `probe_prod` and `probe_ds`. Shared
/// by the blocking and the two-phase self-test, so there is one set of checks.
fn check_outputs(
    n: usize,
    basic: &[f32],
    out_sum: &[f32],
    out_prod: &[f32],
    dsin: &[f32],
    out_ds: &[f32],
    r: &mut ProbeReport,
) {
    r.n = n;
    let out = out_sum;
    for i in 0..n {
        let exact = basic[4 * i] as f64 + basic[4 * i + 1] as f64;
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
    let out = out_prod;
    for i in 0..n {
        let exact = basic[4 * i] as f64 * basic[4 * i + 1] as f64;
        let o = &out[4 * i..4 * i + 4];
        if o[0] != exact as f32 {
            r.mul_not_rounded += 1;
        }
        let err = exact - o[0] as f64;
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
            r.plain_dekker_fail += 1;
        }
    }
    let out = out_ds;
    for i in 0..n {
        let d = &dsin[4 * i..4 * i + 4];
        let (a, b) = (
            Ds::<0> { hi: d[0], lo: d[1] },
            Ds::<0> { hi: d[2], lo: d[3] },
        );
        let (x, y) = (d[0] as f64 + d[1] as f64, d[2] as f64 + d[3] as f64);
        let o = &out[4 * i..4 * i + 4];
        if x + y != 0.0 {
            r.ds_add_max_u2 = r.ds_add_max_u2.max(rel_u2((o[0], o[1]), x + y));
        }
        r.ds_mul_max_u2 = r.ds_mul_max_u2.max(rel_u2((o[2], o[3]), x * y));
        let (s, p) = (a + b, a * b);
        if [s.hi, s.lo, p.hi, p.lo] != [o[0], o[1], o[2], o[3]] {
            r.ds_twin_bit_differences += 1;
        }
    }
}

/// The start-up check for every GPU code path that uses double-single
/// arithmetic (A0 decision for ADR 0006: fail closed). Runs the guarded
/// TwoSum, Fast2Sum, fma product error and DS add and mul on
/// [`N_SELF_TEST`] values (about 0.1 s) and returns [`Error::Backend`] with
/// the failed conditions if DS does not work on this GPU and compiler.
#[cfg(not(target_arch = "wasm32"))]
pub fn self_test(gpu: &Gpu) -> Result<()> {
    let mut t = SelfTest::submit(gpu);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU poll failed");
    t.poll().expect("the map has finished after the wait")
}

/// The error of a failed self-test (one text for both forms of the test).
fn self_test_error(info: &wgpu::AdapterInfo, failures: &[String]) -> Error {
    Error::Backend(format!(
        "double-single self-test failed on {} {:?}: {}. The shader compiler \
         may have re-associated the guarded sums (ADR 0006); do not run \
         the double-single kernels on this GPU.",
        info.name,
        info.backend,
        failures.join("; ")
    ))
}

const MAP_PENDING: u8 = 0;
const MAP_OK: u8 = 1;
const MAP_FAILED: u8 = 2;

/// The self-test in two phases, for callers that must not block (a browser).
///
/// [`SelfTest::submit`] writes the inputs, records the three core kernels in
/// one submission, copies the outputs to one `MAP_READ` buffer and starts the
/// map. [`SelfTest::poll`] returns `None` until the map has finished, then the
/// result of the same checks as [`self_test`], with the same error text.
pub struct SelfTest {
    gpu: Gpu,
    stage: wgpu::Buffer,
    state: Arc<AtomicU8>,
    basic: Vec<f32>,
    dsin: Vec<f32>,
    n: usize,
    done: bool,
}

impl SelfTest {
    /// Starts the test: one submission and one `map_async`. Does not block.
    pub fn submit(gpu: &Gpu) -> Self {
        let n = N_SELF_TEST;
        let d = gpu.device();
        let q = gpu.queue();
        let basic = inputs_f32(n);
        let dsin = inputs_ds(n);
        let size = (4 * n * 4) as u64;
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("self-test"),
            source: wgpu::ShaderSource::Wgsl(PROBE_WGSL.into()),
        });
        let buf = |usage| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let bytes = |w: &[f32]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
        // The guard uniform of `run`: 1.0 (ADR 0006, M16).
        let guard = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        q.write_buffer(&guard, 0, &bytes(&[1.0, 0.0, 0.0, 0.0]));
        let stage = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self-test stage"),
            size: 3 * size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = d.create_command_encoder(&Default::default());
        let mut outs = Vec::new();
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (entry, input, guarded) in [
                ("probe_sum", &basic, true),
                ("probe_prod", &basic, false),
                ("probe_ds", &dsin, true),
            ] {
                let inb = buf(rw);
                q.write_buffer(&inb, 0, &bytes(input));
                let outb = buf(rw);
                let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let mut entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inb.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: outb.as_entire_binding(),
                    },
                ];
                if guarded {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2,
                        resource: guard.as_entire_binding(),
                    });
                }
                let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
                outs.push((inb, outb));
            }
        }
        for (k, (_, outb)) in outs.iter().enumerate() {
            enc.copy_buffer_to_buffer(outb, 0, &stage, k as u64 * size, size);
        }
        q.submit([enc.finish()]);
        let state = Arc::new(AtomicU8::new(MAP_PENDING));
        let cb = Arc::clone(&state);
        stage.map_async(wgpu::MapMode::Read, .., move |r| {
            cb.store(
                if r.is_ok() { MAP_OK } else { MAP_FAILED },
                Ordering::Release,
            );
        });
        Self {
            gpu: gpu.clone(),
            stage,
            state,
            basic,
            dsin,
            n,
            done: false,
        }
    }

    /// Returns `None` while the GPU work or the map is pending. Then it
    /// returns `Some(result)`, once. Later calls return
    /// `Some(Err(Error::Backend(..)))` ("already taken"), not `None`.
    ///
    /// On a native device this call also drives the map callback with a
    /// non-blocking `device.poll`; in a browser the callback runs when the
    /// page returns to the event loop.
    pub fn poll(&mut self) -> Option<Result<()>> {
        if self.done {
            return Some(Err(Error::Backend(
                "the self-test result was already taken".to_string(),
            )));
        }
        // Not an error if the poll has nothing to do (always Ok on the web).
        let _ = self.gpu.device().poll(wgpu::PollType::Poll);
        match self.state.load(Ordering::Acquire) {
            MAP_PENDING => return None,
            MAP_FAILED => {
                self.done = true;
                return Some(Err(Error::Backend(
                    "double-single self-test: the read-back of the GPU result failed".to_string(),
                )));
            }
            _ => {}
        }
        self.done = true;
        let n = self.n;
        let words: Vec<f32> = {
            let view = self.stage.get_mapped_range(..).expect("mapped range");
            view.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        self.stage.unmap();
        let per = 4 * n;
        let mut r = ProbeReport::default();
        check_outputs(
            n,
            &self.basic,
            &words[..per],
            &words[per..2 * per],
            &self.dsin,
            &words[2 * per..3 * per],
            &mut r,
        );
        let f = r.failures();
        Some(if f.is_empty() {
            Ok(())
        } else {
            Err(self_test_error(&self.gpu.info, &f))
        })
    }
}

/// Runs every probe kernel on `gpu` with [`N_PROBE`] values per check and
/// compares with exact values and the Rust twins.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_probe(gpu: &Gpu) -> ProbeReport {
    let mut r = ProbeReport {
        adapter: format!("{} ({:?})", gpu.info.name, gpu.info.backend),
        ..Default::default()
    };
    let n = N_PROBE;
    core_checks(gpu, n, &mut r);

    // Plain (unguarded) algorithms.
    let basic = inputs_f32(n);
    let out = run(gpu, "probe_sum_plain", &basic, false);
    for i in 0..n {
        let exact = basic[4 * i] as f64 + basic[4 * i + 1] as f64;
        let o = &out[4 * i..4 * i + 4];
        if o[0] as f64 + o[1] as f64 != exact {
            r.plain_two_sum_fail += 1;
        }
        if o[2] as f64 + o[3] as f64 != exact {
            r.plain_fast_two_sum_fail += 1;
        }
    }
    let dsin = inputs_ds(n);
    let out = run(gpu, "probe_ds_plain", &dsin, false);
    for i in 0..n {
        let d = &dsin[4 * i..4 * i + 4];
        let (x, y) = (d[0] as f64 + d[1] as f64, d[2] as f64 + d[3] as f64);
        let o = &out[4 * i..4 * i + 4];
        if x + y != 0.0 {
            r.plain_ds_add_max_u2 = r.plain_ds_add_max_u2.max(rel_u2((o[0], o[1]), x + y));
        }
        r.plain_ds_mul_max_u2 = r.plain_ds_mul_max_u2.max(rel_u2((o[2], o[3]), x * y));
    }

    // DS division and square root (seeds from the GPU division and sqrt).
    let out = run(gpu, "probe_ds_div_sqrt", &dsin, true);
    for i in 0..n {
        let d = &dsin[4 * i..4 * i + 4];
        let (x, y) = (d[0] as f64 + d[1] as f64, d[2] as f64 + d[3] as f64);
        let o = &out[4 * i..4 * i + 4];
        r.ds_div_max_u2 = r.ds_div_max_u2.max(rel_u2((o[0], o[1]), x / y));
        r.ds_sqrt_max_u2 = r.ds_sqrt_max_u2.max(rel_u2((o[2], o[3]), x.abs().sqrt()));
    }

    // Division and square roots.
    let out = run(gpu, "probe_divsqrt", &basic, false);
    let (mut div_nr, mut sqrt_nr) = (0usize, 0usize);
    for i in 0..n {
        let (a, b) = (basic[4 * i] as f64, basic[4 * i + 1] as f64);
        let o = &out[4 * i..4 * i + 4];
        let q = a / b;
        r.div_max_ulp = r.div_max_ulp.max(ulp_error(o[0], q));
        div_nr += usize::from(o[0] != q as f32);
        let s = a.abs().sqrt();
        r.sqrt_max_ulp = r.sqrt_max_ulp.max(ulp_error(o[1], s));
        sqrt_nr += usize::from(o[1] != s as f32);
        r.inverse_sqrt_max_ulp = r.inverse_sqrt_max_ulp.max(ulp_error(o[2], 1.0 / s));
        r.recip_max_ulp = r.recip_max_ulp.max(ulp_error(o[3], 1.0 / b));
    }
    r.div_not_rounded = div_nr as f64 / n as f64;
    r.sqrt_not_rounded = sqrt_nr as f64 / n as f64;

    // Sine and cosine: built-in, own f32, own DS.
    let ang = inputs_angles(n);
    let out = run(gpu, "probe_trig", &ang, true);
    for i in 0..n {
        let x = ang[4 * i];
        let xd = x as f64;
        let o = &out[4 * i..4 * i + 4];
        r.builtin_trig_max_abs = r
            .builtin_trig_max_abs
            .max((o[0] as f64 - xd.sin()).abs())
            .max((o[1] as f64 - xd.cos()).abs());
        r.own_trig_max_abs = r
            .own_trig_max_abs
            .max((o[2] as f64 - xd.sin()).abs())
            .max((o[3] as f64 - xd.cos()).abs());
        let (s, c) = sin_cos_f32(x);
        if (s, c) != (o[2], o[3]) {
            r.own_trig_twin_bit_differences += 1;
        }
    }
    let out = run(gpu, "probe_ds_trig", &ang, true);
    for i in 0..n {
        let x = Ds::<{ model::M4 }> {
            hi: ang[4 * i + 2],
            lo: ang[4 * i + 3],
        };
        let xd = x.hi as f64 + x.lo as f64;
        let o = &out[4 * i..4 * i + 4];
        if xd != 0.0 {
            r.ds_sin_max_u2 = r.ds_sin_max_u2.max(rel_u2((o[0], o[1]), xd.sin()));
            let omc = 2.0 * (0.5 * xd).sin().powi(2);
            r.ds_omc_max_u2 = r.ds_omc_max_u2.max(rel_u2((o[2], o[3]), omc));
        }
        let (s, _, omc) = x.sin_cos_small();
        if [s.hi, s.lo, omc.hi, omc.lo] != [o[0], o[1], o[2], o[3]] {
            r.ds_trig_twin_bit_differences += 1;
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_outputs_fail_the_checks() {
        let n = 64;
        let (basic, dsin) = (inputs_f32(n), inputs_ds(n));
        let zero = vec![0.0f32; 4 * n];
        let mut r = ProbeReport::default();
        check_outputs(n, &basic, &zero, &zero, &dsin, &zero, &mut r);
        assert_eq!(r.n, n);
        assert!(!r.failures().is_empty());
    }
}
