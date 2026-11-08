//! Stochastic collisions on the GPU (task K5): the host side of the WGSL in
//! `shaders/collide/` and the Rust twins.
//!
//! Owner: A4 (AJ and A0 decision, 2026-11-08; the rest of `fring-gpu` is A3's).
//!
//! The kernel follows `crates/fring-collide/SPEC.md` 5.2 to 5.5 and 12 on the
//! cells of [`crate::bin::GpuBinner`]: one thread per cell, the members in id
//! order. What each WGSL file has as a Rust twin:
//!
//! | WGSL | Twin |
//! |---|---|
//! | `rng.wgsl` (`philox4x32_10`, `mulhilo`, `u01_f32`) | `fring_model::rng`, word for word |
//! | `pair.wgsl` (`impact_normal`, `collide_pair`, `restitution`) | `fring_collide::pair` run with `f32` |
//! | `peculiar.wgsl` (`vcirc_ds`, `peculiar_ds`) | `fring_collide::frame::peculiar_velocity` run with [`Dsm`] |
//! | `cell.wgsl` (`col_cells`) | [`cell_twin`] |
//!
//! Exact between GPU and twin: the Philox words, the cell runs, N_c, the smallest
//! id, the candidate indices i and j, and the number of candidates (unless
//! frac(M) is within about 1e-6 of the draw). Not exact (f32 against f32 with
//! another sine, power and square root, or f32 against f64): the acceptance
//! test, the impact normal and the velocity changes. The tests compare those
//! with tolerances.
//!
//! Overfull cells: a cell that needs 32,767 or more candidates is skipped on
//! the GPU and counted; [`GpuCollider::read_tally`] returns the error. The CPU
//! stops at the first such cell; the GPU has changed the other cells by then.
//! The run stops either way.

use crate::bin::Dsm;
use crate::gpu::Gpu;
use crate::wh_gpu::grid;
use fring_collide::pair;
use fring_model::config::Restitution;
use fring_model::physics::SaturnField;
use fring_model::rng::{Counter, Philox, Stream, u01_f32};
use fring_model::{Error, Result, Vec3};

/// The WGSL of a collision pass: `ds.wgsl` and the files of `shaders/collide/`.
pub const COLLIDE_WGSL: &str = concat!(
    include_str!("../shaders/ds.wgsl"),
    include_str!("../shaders/collide/rng.wgsl"),
    include_str!("../shaders/collide/pair.wgsl"),
    include_str!("../shaders/collide/peculiar.wgsl"),
    include_str!("../shaders/collide/cell.wgsl")
);

/// The WGSL of the test kernels (functions of the files above, one by one).
pub const COLLIDE_TEST_WGSL: &str = concat!(
    include_str!("../shaders/ds.wgsl"),
    include_str!("../shaders/collide/rng.wgsl"),
    include_str!("../shaders/collide/pair.wgsl"),
    include_str!("../shaders/collide/peculiar.wgsl"),
    include_str!("../shaders/collide/test.wgsl")
);

/// Candidates per cell and step: `draw` has 16 bits (SPEC 5.5).
pub const MAX_CANDIDATES: u32 = 32_767;
const CHUNK: usize = 256;
const COL_WG: u32 = 64;
const TOT_WORDS: usize = 16;

/// The model parameters (SPEC 3, 6) and the run seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CollideParams {
    /// Radius s of one real particle, km.
    pub particle_radius_km: f64,
    /// Real particles per superparticle W.
    pub weight: f64,
    /// Restitution law.
    pub restitution: Restitution,
    /// Run seed (key of the Philox generator).
    pub seed: u64,
}

impl CollideParams {
    /// σ = π (2s)² in km².
    pub fn cross_section_km2(&self) -> f64 {
        std::f64::consts::PI * (2.0 * self.particle_radius_km).powi(2)
    }

    /// k = W σ Δt_c / V_c in s km⁻¹ (the factor of g_max in M, SPEC 5.3), as the
    /// shader gets it.
    pub fn rate_factor(&self, dt_s: f64, volume_km3: f64) -> f32 {
        (self.weight * self.cross_section_km2() * dt_s / volume_km3) as f32
    }
}

fn le_bytes_f32(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_bytes_u32(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The 5 `vec4<f32>` of the `cp` uniform of `cell.wgsl` (and the restitution
/// part of the test uniform): `[0]` one, 0, ε, v_c; `[1]` p; `[2]` GM, R;
/// `[3]` J2, J4; `[4]` J6 (DS pairs).
pub fn collide_uniform(field: &SaturnField<f64>, law: &Restitution) -> Vec<f32> {
    let (eps, vc, p) = match *law {
        Restitution::Constant { epsilon } => (epsilon, 0.0, 0.0),
        Restitution::PowerLaw {
            reference_speed_km_s,
            exponent,
        } => (1.0, reference_speed_km_s, exponent),
    };
    let ds = |x: f64| {
        let d = Dsm::split(x);
        [d.hi, d.lo]
    };
    let mut v = vec![1.0, 0.0, eps as f32, vc as f32, p as f32, 0.0, 0.0, 0.0];
    for x in [
        field.gm_km3_s2,
        field.reference_radius_km,
        field.j2,
        field.j4,
        field.j6,
    ] {
        v.extend(ds(x));
    }
    v.resize(20, 0.0);
    v
}

/// The sums of one cell, as the shader gets them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CellTally {
    /// Candidates tested.
    pub candidates: u32,
    /// Accepted collisions.
    pub collisions: u32,
    /// Candidates with |g| > g_max.
    pub gmax_exceeded: u32,
    /// Σ ΔK of the cell, km² s⁻² (f32).
    pub loss_km2_s2: f32,
    /// Σ ΔK² of the cell (f32).
    pub loss_sq_km4_s4: f32,
    /// W σ g_max Δt / V > 1.
    pub overfull: bool,
    /// M ≥ 32,767: the cell is skipped.
    pub too_full: bool,
}

/// Rust twin of one thread of `col_cells` after the mean is subtracted: `w`
/// holds u − ū in f32 for the members in id order and is changed in place;
/// `first_id` is the smallest id; `k_rate` = W σ Δt / V. The pair operator and
/// the impact geometry are `fring_collide::pair` in f32, the generator is
/// `fring_model::rng`.
pub fn cell_twin(
    rng: &Philox,
    law: &Restitution,
    k_rate: f32,
    first_id: u32,
    step: u64,
    w: &mut [Vec3<f32>],
) -> CellTally {
    let mut t = CellTally::default();
    let nc = w.len();
    if nc < 2 {
        return t;
    }
    let wmax2 = w.iter().map(|v| v.dot(*v)).fold(0.0f32, f32::max);
    let g_max = 2.0 * wmax2.sqrt();
    if g_max.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return t;
    }
    let per_pair = k_rate * g_max;
    t.overfull = per_pair > 1.0;
    let m = 0.5 * (nc as f32) * ((nc - 1) as f32) * per_pair;
    if m >= MAX_CANDIDATES as f32 {
        t.too_full = true;
        return t;
    }
    let block = |draw: u32| {
        rng.block(Counter {
            index: first_id,
            step,
            stream: Stream::COLLISIONS,
            draw: draw as u16,
        })
    };
    let w0 = block(0);
    let whole = m.floor();
    let mut cands = whole as u32;
    if m - whole > u01_f32(w0[0]) {
        cands += 1;
    }
    for k in 0..cands {
        let b = block(1 + 2 * k);
        let i = fring_model::rng::mulhilo(b[0], nc as u32).0 as usize;
        let j0 = fring_model::rng::mulhilo(b[1], (nc - 1) as u32).0 as usize;
        let j = if j0 >= i { j0 + 1 } else { j0 };
        let g = w[i] - w[j];
        let g_norm = g.dot(g).sqrt();
        if g_norm.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            continue;
        }
        if g_norm > g_max {
            t.gmax_exceeded += 1;
        } else if u01_f32(b[2]) >= g_norm / g_max {
            continue;
        }
        let v = block(2 + 2 * k);
        let n_hat = pair::impact_normal(g, u01_f32(v[0]), u01_f32(v[2]));
        if let Some(imp) = pair::collide(g, n_hat, law) {
            w[i] += imp.delta_ui_km_s;
            w[j] -= imp.delta_ui_km_s;
            t.collisions += 1;
            t.loss_km2_s2 += imp.energy_loss_km2_s2;
            t.loss_sq_km4_s4 += imp.energy_loss_km2_s2 * imp.energy_loss_km2_s2;
        }
    }
    t.candidates = cands;
    t
}

/// The collision pass on the GPU: pipelines, scratch buffers and the tally.
///
/// It reads the cells from the `sorted` and `skeys` buffers of
/// [`crate::bin::GpuBinner`] (or buffers of the same layout) and changes the
/// double-single velocity planes of the state buffer in place.
pub struct GpuCollider {
    gpu: Gpu,
    n: usize,
    chunks: usize,
    params: CollideParams,
    volume_km3: f64,
    pipelines: Vec<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    cp_buf: wgpu::Buffer,
    ctl_buf: wgpu::Buffer,
    state: wgpu::Buffer,
    sorted: wgpu::Buffer,
    skeys: wgpu::Buffer,
    uw: wgpu::Buffer,
    en: wgpu::Buffer,
    tot: wgpu::Buffer,
}

const ENTRIES: [&str; 4] = ["col_clear", "col_cells", "col_sum1", "col_sum2"];

impl GpuCollider {
    /// A collision pass for the `n` binned particles of `state` (6 planes of
    /// `cap` DS values), cells from `sorted` (`n` ids) and `skeys` (`n` keys of
    /// two `u32`), with cell volume `volume_km3`. `box_mode`: the velocities of
    /// the state are peculiar velocities (K1 box tests); else the shader
    /// subtracts v_circ(ρ) φ̂ with `field` (SPEC 5.2).
    ///
    /// The caller must have run the DS self-test ([`crate::probe::SelfTest`]).
    /// The state buffer needs the usage `STORAGE`.
    ///
    /// # Errors
    /// [`Error::Backend`] if a buffer is too small or the device allows fewer
    /// than 6 storage buffers per shader stage.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        state: &wgpu::Buffer,
        sorted: &wgpu::Buffer,
        skeys: &wgpu::Buffer,
        n: usize,
        cap: usize,
        field: &SaturnField<f64>,
        params: CollideParams,
        volume_km3: f64,
        box_mode: bool,
    ) -> Result<Self> {
        if cap < n
            || state.size() < (6 * cap * 8) as u64
            || sorted.size() < (4 * n) as u64
            || skeys.size() < (8 * n) as u64
        {
            return Err(Error::Backend(format!(
                "collisions: a buffer is too small for {n} particles (capacity {cap})"
            )));
        }
        if gpu.limits.max_storage_buffers_per_shader_stage < 6 {
            return Err(Error::Backend(format!(
                "collisions need 6 storage buffers per shader stage; {} allows {}",
                gpu.info.name, gpu.limits.max_storage_buffers_per_shader_stage
            )));
        }
        let chunks = n.div_ceil(CHUNK);
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
        let cp_buf = mk(80, uni);
        q.write_buffer(
            &cp_buf,
            0,
            &le_bytes_f32(&collide_uniform(field, &params.restitution)),
        );
        let flags = u32::from(box_mode)
            | if matches!(params.restitution, Restitution::PowerLaw { .. }) {
                2
            } else {
                0
            };
        let ctl_buf = mk(16, uni);
        q.write_buffer(
            &ctl_buf,
            0,
            &le_bytes_u32(&[n as u32, cap as u32, flags, chunks as u32]),
        );
        let uw = mk((16 * n) as u64, rw);
        let en = mk((4 * (2 * n + 4 * chunks)) as u64, rw);
        let tot = mk((4 * TOT_WORDS) as u64, rw);
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collide"),
            source: wgpu::ShaderSource::Wgsl(COLLIDE_WGSL.into()),
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
            label: Some("collide"),
            entries: &[
                entry(0, st(false)),
                entry(1, st(true)),
                entry(2, st(true)),
                entry(3, st(false)),
                entry(4, st(false)),
                entry(5, st(false)),
                entry(6, u),
                entry(7, u),
                entry(8, u),
            ],
        });
        let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("collide"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ENTRIES
            .iter()
            .map(|e| {
                d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        Ok(Self {
            gpu: gpu.clone(),
            n,
            chunks,
            params,
            volume_km3,
            pipelines,
            layout,
            cp_buf,
            ctl_buf,
            state: state.clone(),
            sorted: sorted.clone(),
            skeys: skeys.clone(),
            uw,
            en,
            tot,
        })
    }

    /// The model parameters.
    pub fn params(&self) -> &CollideParams {
        &self.params
    }

    /// Records one collision pass into `encoder`: the tally is cleared, the
    /// cells are processed, the energy sums are made. `step` is the step count
    /// of the run after the integrator step (SPEC 5.5); `dt_s` is Δt_c. The
    /// caller submits. Each call has its own small uniform buffer.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, step: u64, dt_s: f64) {
        let d = self.gpu.device();
        let key = Philox::from_seed(self.params.seed).key();
        let k_rate = self.params.rate_factor(dt_s, self.volume_km3);
        let dynb = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("collide dyn"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        {
            let mut v = dynb.get_mapped_range_mut(..).expect("range");
            v.copy_from_slice(&le_bytes_u32(&[
                key[0],
                key[1],
                step as u32,
                (step >> 32) as u32,
                k_rate.to_bits(),
                0,
                0,
                0,
            ]));
        }
        dynb.unmap();
        fn e(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }
        }
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("collide"),
            layout: &self.layout,
            entries: &[
                e(0, &self.state),
                e(1, &self.sorted),
                e(2, &self.skeys),
                e(3, &self.uw),
                e(4, &self.en),
                e(5, &self.tot),
                e(6, &self.cp_buf),
                e(7, &self.ctl_buf),
                e(8, &dynb),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("collide"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind, &[]);
        let (gx, gy) = grid(self.n.max(1), COL_WG);
        let (cx, cy) = grid(self.chunks.max(1), COL_WG);
        let dispatch = [(1, 1), (gx, gy), (cx, cy), (1, 1)];
        for (k, (x, y)) in dispatch.into_iter().enumerate() {
            pass.set_pipeline(&self.pipelines[k]);
            pass.dispatch_workgroups(x, y, 1);
        }
    }

    /// Reads the tally of the last pass (blocking; for tests and tools). The
    /// energy sums are double-single, so they are exact in the f64 sum.
    ///
    /// # Errors
    /// [`Error::Backend`] if a cell needed 32,767 or more candidates (the CPU
    /// returns `CollideError` for the same cell).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_tally(&self) -> Result<fring_collide::tally::Tally> {
        use fring_collide::tally::Tally;
        use fring_model::CollisionStats;
        let w = self.read_words(&self.tot, 0, TOT_WORDS);
        let f = |hi: u32, lo: u32| f64::from(f32::from_bits(hi)) + f64::from(f32::from_bits(lo));
        if w[9] > 0 {
            return Err(Error::Backend(format!(
                "{} cells need more than {MAX_CANDIDATES} candidates in one step (the cells are too \
                 large; the CPU returns CollideError for the same cell)",
                w[9]
            )));
        }
        let stats = CollisionStats {
            energy_loss_km2_s2: f(w[10], w[11]),
            candidates: u64::from(w[0]),
            occupied_cells: u64::from(w[3]),
            occupancy_sum: u64::from(w[4]),
            occupancy_sq_sum: u64::from(w[5]) | (u64::from(w[6]) << 32),
            lonely_particles: u64::from(w[7]),
            overfull_cells: u64::from(w[8]),
            gmax_exceeded: u64::from(w[2]),
            deep_collisions: 0,
        };
        Ok(Tally {
            collisions: u64::from(w[1]),
            stats,
            energy_loss_sq_km4_s4: f(w[12], w[13]),
        })
    }

    /// Reads `words` `u32` of `buf` from word `offset` (blocking).
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

    /// The scratch buffer `uw` (u − ū per sorted position, `vec4<f32>`), for tests.
    pub fn scratch_buffer(&self) -> &wgpu::Buffer {
        &self.uw
    }

    /// The state buffer.
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state
    }
}
