//! [`GpuBackend`]: the wgpu implementation of [`fring_model::Backend`]
//! (task G2, ADR 0006 and 0007).
//!
//! The particle state lives on the GPU between calls: Saturn-centred,
//! inertial position and velocity in double-single (DS), 6 planes of DS
//! values (structure of arrays). [`Backend::step`] works in blocks of steps.
//! For each block the host fills the moon kick table in f64 (exactly as
//! `CpuBackend` does), rounds it to DS, uploads it, and starts one dispatch;
//! every thread then advances one particle through the whole block. The
//! host fills block k+1 while the GPU runs block k. Nothing is read back
//! except 4 bytes at the end of each `step` call (the failure flag).
//!
//! # Collisions (task K5-WIRE)
//!
//! With `[collisions] model = "stochastic"`, blocks end at multiples of
//! `every_steps`. After such a block the state is binned into the cells
//! (`GpuBinner`) and `GpuCollider` changes the velocities, in the same
//! submission; then the 16 tally words are read (one wait per collision step)
//! and added on the host. [`Backend::stats`] returns the sum since its last
//! call. Frozen or dropped particles are not binned, so they do not collide.
//! The shader takes the smallest *index* of a cell for the random numbers;
//! it equals the id when the ids are 0..n in order, as
//! `generate_particles` makes them. Arc, GPU mode C, chaos and hard_sphere
//! are refused (see `check_collisions`).
//! [`Backend::particles`] and [`Backend::stats`] read the whole state back,
//! so do not call them at every step.
//!
//! # Start-up checks (ADR 0006, M16)
//!
//! [`GpuBackend::new`] fails with [`Error::Backend`] unless (1) the
//! double-single self-test passes and (2) one block of the real kernel gives
//! the result of its Rust twin to a relative 10⁻⁶ on a few particles.
//!
//! # Mode
//!
//! Option D+ (default, science runs), option C (reference mode) or option D
//! (faster, but it does not meet PLAN section 10 in the tail; ADR 0006
//! addendum). The run config has no field for it, so [`Backend::new`] reads
//! the environment variable `FRING_GPU_MODE` (`dp`, `c` or `d`); use
//! [`GpuBackend::with_options`] in code.

use crate::bin::{BinGeometry, GpuBinner};
use crate::collide::{CollideParams, GpuCollider};
use crate::gpu::Gpu;
use crate::spike::{Consts, State, kick_scale, state_from_f64, state_to_f64};
use crate::stepper::{GpuStepper, StepperOptions, plan_capacity};
use crate::wf32::model;
use crate::wh::{self, HostTable, Mode};
use crate::wh_gpu::{DEFAULT_WORKGROUP, NO_FAILURE, WhKernels, unbound};
use fring_collide::tally::Tally;
use fring_model::config::{Collisions, Domain};
use fring_model::physics::frames::reference_mean_motion;
use fring_model::physics::{
    ArcGeometry, EpicycleElements, KickTable, SaturnField, epicycle_state, interpolate_moons,
};
use fring_model::{
    Backend, ChaosReport, DropReport, Error, InitialState, MoonState, ParticleView, Particles,
    Result, RunConfig, StepStats, System, Vec3,
};
use rayon::prelude::*;

pub use crate::stepper::MAX_BLOCK_STEPS;
/// Particle-steps per dispatch that keep one command buffer near 0.2 s on the
/// M4 (macOS stops GPU work that runs for several seconds).
const TARGET_PARTICLE_STEPS: usize = 200_000_000;

/// Options of [`GpuBackend::with_options`].
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Number format of the kernels.
    pub mode: Mode,
    /// Workgroup size (a power of two; 64 by default).
    pub workgroup: u32,
    /// Steps per dispatch; `None` chooses it from the number of particles.
    pub block_steps: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            mode: Mode::Dp,
            workgroup: DEFAULT_WORKGROUP,
            block_steps: None,
        }
    }
}

impl Options {
    /// Defaults, with the mode from `FRING_GPU_MODE` (`dp`, `d` or `c`).
    pub fn from_env() -> Result<Self> {
        let mut o = Self::default();
        if let Ok(m) = std::env::var("FRING_GPU_MODE") {
            o.mode = match m.trim().to_ascii_lowercase().as_str() {
                "dp" | "d+" | "" => Mode::Dp,
                "d" => Mode::D,
                "c" => Mode::C,
                other => {
                    return Err(Error::Config(format!(
                        "FRING_GPU_MODE must be \"dp\", \"d\" or \"c\", not {other:?}"
                    )));
                }
            };
        }
        Ok(o)
    }
}

/// The wgpu backend (see the module documentation). It keeps a
/// [`GpuStepper`] and adds the blocking parts: the ring of slots with its
/// poll, the read-back of the snapshot, the failure flag and the start-up
/// check.
pub struct GpuBackend {
    kernels: WhKernels,
    mode: Mode,
    stepper: GpuStepper,
    field: SaturnField<f64>,
    ids: Vec<u32>,
    n: usize,
    cap: usize,
    status_stage: wgpu::Buffer,
    stage: wgpu::Buffer,
    acc_stage: Option<wgpu::Buffer>,
    snapshot: Particles,
    snapshot_stale: bool,
    energy0: Vec<f64>,
    max_rel_energy_error: f64,
    /// `[loss] on_unbound = "freeze"` (ICR-004): a lost particle is frozen and
    /// `step` does not return an error for it.
    freeze: bool,
    /// `[loss] moon_radius = "params"`: the radii must be set before the first step.
    rule: bool,
    radii_set: bool,
    /// Staging buffer of [`Backend::dropped`], made on first use.
    drop_stage: Option<wgpu::Buffer>,
    /// `[collisions] model = "stochastic"` (task K5-WIRE).
    collide: Option<GpuCollision>,
}

/// The bin pass and the collision pass (task K5-WIRE), and the sum of the
/// tallies of the passes since the last [`Backend::stats`] call.
struct GpuCollision {
    binner: GpuBinner,
    collider: GpuCollider,
    /// One collision step every this many integrator steps.
    every: u64,
    /// Sum of the pass tallies, added on the host in pass order (so the
    /// result is the same in every run).
    tally: Tally,
}

/// The refusals of `[collisions]` on the GPU (as on the CPU, task K2-CPU, and
/// the reasons in `docs/handoffs/K5-WIRE.md`). No GPU needed.
fn check_collisions(config: &RunConfig, mode: Mode) -> Result<()> {
    match config.collisions {
        Collisions::None => return Ok(()),
        Collisions::HardSphere { .. } => {
            return Err(Error::Config(
                "collisions.model = \"hard_sphere\" is not implemented on the GPU backend \
                 (task K3 is CPU only)"
                    .to_string(),
            ));
        }
        Collisions::Stochastic { .. } => {}
    }
    if matches!(config.domain, Domain::Arc { .. }) {
        return Err(Error::Config(
            "collisions with domain.kind = \"arc\" are not available on the GPU backend: \
             a test of the collider on the arc must come first (tasks ARC-CELLS-GPU, D1)"
                .to_string(),
        ));
    }
    if config
        .diagnostics
        .as_ref()
        .is_some_and(|d| d.chaos.is_some())
    {
        return Err(Error::Config(
            "[diagnostics.chaos] cannot be used with collisions.model = \"stochastic\": \
             a shadow particle has no meaning under random velocity kicks, because the \
             main particle and its shadow would get different kicks and the separation \
             (and so MEGNO) would measure the noise (A0 decision, 2026-11-08)"
                .to_string(),
        ));
    }
    if mode == Mode::C {
        return Err(Error::Config(
            "collisions are not available in GPU mode C (not tested); use mode dp or d".to_string(),
        ));
    }
    Ok(())
}

impl GpuBackend {
    /// As [`Backend::new`] with explicit options.
    pub fn with_options(
        config: &RunConfig,
        system: &System,
        init: &InitialState,
        opts: Options,
    ) -> Result<Self> {
        config.validate()?;
        system.validate()?;
        init.particles.validate()?;
        // ICR-004 amendment 2: no sub-stepping on the GPU.
        if config.substep.is_some() {
            return Err(Error::Config(
                "[substep] is not available on the GPU backend (ICR-004 amendment 2); \
                 remove it or use the CPU backend"
                    .to_string(),
            ));
        }
        check_collisions(config, opts.mode)?;
        let freeze = config.freeze_unbound();
        let rule = config.moon_radius_rule();
        if (freeze || rule) && opts.mode == Mode::C {
            return Err(Error::Config(
                "[loss] (freeze or the moon-radius rule) is not available in GPU mode C; \
                 use mode dp or d"
                    .to_string(),
            ));
        }
        let eph = &system.moons;
        if !eph.tracks.is_empty() {
            let end = eph.end_s();
            let run_end = init.time_s + config.time.duration_s();
            if init.time_s < eph.start_s || run_end > end {
                return Err(Error::Input(format!(
                    "the moon ephemeris covers [{}, {end}] s, the run needs [{}, {run_end}] s",
                    eph.start_s, init.time_s
                )));
            }
        }
        wh::check_moons(eph.tracks.len())?;
        let step_s = config.time.step_s;
        let p = &init.particles;
        let n = p.len();
        if n > u32::MAX as usize / 2 {
            return Err(Error::Backend(format!("{n} particles are too many")));
        }
        let Some(gpu) = Gpu::new() else {
            return Err(Error::Backend("no GPU adapter found".to_string()));
        };
        let wg = opts.workgroup;
        // Before the self-test, so a too large run fails early.
        let cap = plan_capacity(&gpu, n, wg)?;
        let state_bytes = (6 * cap * 8) as u64;
        let kernels = WhKernels::new(&gpu, wg)?;
        startup_check(
            &kernels,
            opts.mode,
            system,
            init.time_s,
            step_s,
            config.frame.reference_semi_major_axis_km,
        )?;

        // Task D1: a co-moving arc domain. Particles made all round the ring
        // are folded into the arc first (a rotation by whole widths).
        let field = SaturnField::<f64>::from_params(&system.saturn);
        let arc = ArcGeometry::from_domain_in_field(
            &config.domain,
            &field,
            config.frame.reference_semi_major_axis_km,
        )?;
        let folded;
        let init = if let Some(g) = &arc {
            folded = fold_into_arc(g, init)?;
            &folded
        } else {
            init
        };
        let p = &init.particles;

        let block_steps = opts
            .block_steps
            .unwrap_or_else(|| (TARGET_PARTICLE_STEPS / n.max(1)).clamp(8, MAX_BLOCK_STEPS))
            .clamp(1, MAX_BLOCK_STEPS);
        let chaos_cfg = config.diagnostics.as_ref().and_then(|d| d.chaos);
        let mut stepper = GpuStepper::new_with_drift(
            &gpu,
            &kernels,
            system,
            init,
            StepperOptions {
                step_s,
                block_steps,
                mode: opts.mode,
                chaos: chaos_cfg,
                drift_radius_km: Some(config.frame.reference_semi_major_axis_km),
            },
        )?;

        if rule && arc.is_some() {
            return Err(Error::Config(
                "[loss] moon_radius = \"params\" cannot be combined with the arc (ICR-004)"
                    .to_string(),
            ));
        }
        if let Some(geom) = arc {
            stepper.set_arc(geom)?;
        }
        // After `set_arc`, which replaces the status buffer.
        let collide = match config.collisions {
            Collisions::Stochastic {
                particle_radius_km,
                weight,
                restitution,
                every_steps,
                ..
            } => {
                let n0 = reference_mean_motion(&field, config.frame.reference_semi_major_axis_km);
                let geom = BinGeometry::from_config(config, n0)?;
                let binner = GpuBinner::for_stepper(&gpu, &stepper, geom)?;
                let volume = geom.lx_km * geom.ly_km * geom.lz_km;
                let collider = GpuCollider::new(
                    &gpu,
                    stepper.state_buffer(),
                    binner.sorted_buffer(),
                    binner.sorted_keys_buffer(),
                    n,
                    stepper.capacity(),
                    &field,
                    CollideParams {
                        particle_radius_km,
                        weight,
                        restitution,
                        seed: config.run.seed,
                    },
                    volume,
                    false,
                )?;
                Some(GpuCollision {
                    binner,
                    collider,
                    every: every_steps,
                    tally: Tally::default(),
                })
            }
            _ => None,
        };

        let d = gpu.device();
        let mk = |size: u64, usage: wgpu::BufferUsages| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size.max(16),
                usage,
                mapped_at_creation: false,
            })
        };
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let acc_stage = chaos_cfg.map(|_| mk((crate::chaos::ACC_PLANES * cap * 8) as u64, read));

        let energy0 = if eph.tracks.is_empty() {
            (0..n)
                .map(|i| field.specific_energy(p.position_km(i), p.velocity_km_s(i)))
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            mode: opts.mode,
            stepper,
            field,
            ids: p.id.clone(),
            n,
            cap,
            status_stage: mk(16, read),
            stage: mk(state_bytes, read),
            acc_stage,
            snapshot: p.clone(),
            snapshot_stale: false,
            energy0,
            max_rel_energy_error: 0.0,
            freeze,
            rule,
            radii_set: false,
            drop_stage: None,
            collide,
            kernels,
        })
    }

    /// Sets the radius of each moon in km for `moon_radius = "params"`
    /// (ICR-004), in the moon order of the `System`, as
    /// `CpuBackend::set_moon_radii_km` does. The caller reads
    /// `<params_dir>/moons.toml` (`moon_radii_from_toml_str`) and matches the
    /// names. The backend reads no files. With the rule off in the config the
    /// radii are checked and ignored.
    ///
    /// # Errors
    /// [`Error::Config`] if the number of radii differs from the number of
    /// moons, a radius is not finite and > 0, or steps were already recorded.
    pub fn set_moon_radii_km(&mut self, radii_km: &[f64]) -> Result<()> {
        let n = self.stepper.n_moons();
        if radii_km.len() != n {
            return Err(Error::Config(format!(
                "set_moon_radii_km: {} radii for {n} moons",
                radii_km.len()
            )));
        }
        if let Some(r) = radii_km.iter().find(|r| !(r.is_finite() && **r > 0.0)) {
            return Err(Error::Config(format!(
                "set_moon_radii_km: radius {r} is not finite and > 0"
            )));
        }
        if !self.rule {
            return Ok(());
        }
        self.stepper.set_moon_radii_km(radii_km)?;
        self.radii_set = true;
        Ok(())
    }

    /// The non-blocking core (read-only), for example to compare its buffers.
    pub fn stepper(&self) -> &GpuStepper {
        &self.stepper
    }

    /// The GPU (device and queue) that the backend uses, for example to read
    /// the buffers of [`Self::stepper`].
    pub fn gpu(&self) -> &Gpu {
        self.kernels.gpu()
    }

    /// The number format of the kernels.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Name of the adapter, for example "Apple M4".
    pub fn adapter_name(&self) -> String {
        format!(
            "{} ({:?})",
            self.kernels.gpu().info.name,
            self.kernels.gpu().info.backend
        )
    }

    /// The limits of the device.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.kernels.gpu().limits
    }

    /// Steps per dispatch.
    pub fn block_steps(&self) -> usize {
        self.stepper.block_steps()
    }

    fn has_moons(&self) -> bool {
        self.stepper.n_moons() > 0
    }

    /// Waits for the GPU and returns the failure flag.
    fn failure_flag(&mut self) -> u32 {
        let gpu = self.kernels.gpu();
        let d = gpu.device();
        let mut enc = d.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(self.stepper.status_buffer(), 0, &self.status_stage, 0, 4);
        gpu.queue().submit([enc.finish()]);
        self.status_stage
            .map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
        d.poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let flag = {
            let v = self.status_stage.get_mapped_range(..).expect("range");
            u32::from_le_bytes([v[0], v[1], v[2], v[3]])
        };
        self.status_stage.unmap();
        flag
    }

    /// Reads the state back and converts it to inertial f64.
    fn sync_snapshot(&mut self) -> Result<()> {
        if !self.snapshot_stale {
            return Ok(());
        }
        let gpu = self.kernels.gpu();
        let d = gpu.device();
        let size = (6 * self.cap * 8) as u64;
        let mut enc = d.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(self.stepper.state_buffer(), 0, &self.stage, 0, size);
        gpu.queue().submit([enc.finish()]);
        self.stage
            .map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
        d.poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let (cap, n) = (self.cap, self.n);
        let bad = {
            let view = self.stage.get_mapped_range(..).expect("range");
            let bytes: &[u8] = &view;
            let word = |w: usize| {
                let o = 4 * w;
                f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as f64
            };
            let p = &mut self.snapshot;
            let planes: [&mut Vec<f64>; 6] = [
                &mut p.x_km,
                &mut p.y_km,
                &mut p.z_km,
                &mut p.vx_km_s,
                &mut p.vy_km_s,
                &mut p.vz_km_s,
            ];
            let mut bad = false;
            for (c, plane) in planes.into_iter().enumerate() {
                let any_bad = plane[..n]
                    .par_iter_mut()
                    .enumerate()
                    .map(|(i, out)| {
                        let w = 2 * (c * cap + i);
                        *out = word(w) + word(w + 1);
                        !out.is_finite()
                    })
                    .reduce(|| false, |a, b| a || b);
                bad |= any_bad;
            }
            bad
        };
        self.stage.unmap();
        if bad {
            return Err(Error::Backend(
                "a particle state is not finite after the GPU step".to_string(),
            ));
        }
        self.snapshot_stale = false;
        Ok(())
    }
}

impl Backend for GpuBackend {
    fn new(config: &RunConfig, system: &System, init: &InitialState) -> Result<Self> {
        Self::with_options(config, system, init, Options::from_env()?)
    }

    /// Advances all particles by `n_steps` steps. If a particle has an unbound
    /// or invalid orbit, the call returns `Error::Backend` with the lowest
    /// such index; the state of the backend is then unspecified and the run
    /// must stop. The call waits for the GPU once, at its end.
    fn step(&mut self, n_steps: u64) -> Result<()> {
        if self.rule && !self.radii_set {
            return Err(Error::Config(
                "moon_radius = \"params\" needs the moon radii: call \
                 GpuBackend::set_moon_radii_km before the first step"
                    .to_string(),
            ));
        }
        let gpu = self.kernels.gpu().clone();
        let (d, q) = (gpu.device(), gpu.queue());
        // `encode_block` counts each block in the stepper at once. A failed
        // call must leave the count as it was (the state is unspecified, but
        // `time_s` and `steps_total` must not move), so it is set back.
        let base = self.stepper.steps_total;
        let mut done = 0u64;
        while done < n_steps {
            let mut enc = d.create_command_encoder(&Default::default());
            // Blocks end at multiples of `every_steps` (as on the CPU), so the
            // collision steps do not depend on how the caller splits the run.
            let mut max_steps = n_steps - done;
            if let Some(c) = &self.collide {
                max_steps = max_steps.min(c.every - self.stepper.steps_total % c.every);
            }
            let r = self.stepper.encode_block(&mut enc, max_steps, 0, |last| {
                if let Some(idx) = last {
                    d.poll(wgpu::PollType::Wait {
                        submission_index: Some(idx),
                        timeout: None,
                    })
                    .expect("GPU poll failed");
                }
            });
            let (block, slot) = match r {
                Ok(v) => v,
                Err(e) => {
                    self.stepper.steps_total = base;
                    return Err(e);
                }
            };
            // The collision step on the synchronised state at the end of a
            // block that ends on a multiple of `every_steps`: bin, then
            // collide, in the same submission.
            let steps_now = self.stepper.steps_total;
            let collide_now = self
                .collide
                .as_ref()
                .is_some_and(|c| steps_now.is_multiple_of(c.every));
            if collide_now && let Some(c) = &self.collide {
                c.binner.encode(&mut enc, self.stepper.time_s());
                c.collider
                    .encode(&mut enc, steps_now, c.every as f64 * self.stepper.step_s());
            }
            let idx = q.submit([enc.finish()]);
            self.stepper.set_last(slot, idx);
            done += block;
            if collide_now && let Some(c) = &mut self.collide {
                // The pass clears its counters, so each pass is read here.
                match c.collider.read_tally() {
                    Ok(t) => c.tally += t,
                    Err(e) => {
                        self.stepper.steps_total = base;
                        return Err(e);
                    }
                }
            }
        }
        self.snapshot_stale = true;
        let flag = self.failure_flag();
        // With `freeze` a lost particle is flagged and frozen, not an error.
        if flag != NO_FAILURE && !self.freeze {
            self.stepper.steps_total = base;
            let idx = flag as usize;
            let id = self.ids.get(idx).copied().unwrap_or(u32::MAX);
            return Err(unbound(id, flag));
        }
        Ok(())
    }

    fn time_s(&self) -> f64 {
        self.stepper.time_s()
    }

    fn moons(&self) -> MoonState {
        let t = self.time_s();
        interpolate_moons(self.stepper.ephemeris(), t).unwrap_or(MoonState {
            time_s: t,
            moons: Vec::new(),
        })
    }

    fn particles(&mut self) -> Result<ParticleView<'_>> {
        self.sync_snapshot()?;
        Ok(ParticleView {
            time_s: self.time_s(),
            particles: &self.snapshot,
        })
    }

    fn chaos(&mut self) -> Result<Option<ChaosReport>> {
        let (Some((cfg, acc_buf)), Some(acc_stage)) =
            (self.stepper.chaos_parts(), self.acc_stage.as_ref())
        else {
            return Ok(None);
        };
        let gpu = self.kernels.gpu();
        let d = gpu.device();
        let size = (crate::chaos::ACC_PLANES * self.cap * 8) as u64;
        let mut enc = d.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(acc_buf, 0, acc_stage, 0, size);
        gpu.queue().submit([enc.finish()]);
        acc_stage.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
        d.poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let words: Vec<f32> = {
            let view = acc_stage.get_mapped_range(..).expect("range");
            view.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        acc_stage.unmap();
        let interval_s = f64::from(cfg.renorm_steps) * self.stepper.step_s();
        Ok(Some(crate::chaos::report(
            &words,
            self.cap,
            self.n,
            interval_s,
            self.time_s(),
        )))
    }

    /// Lost particles (ICR-004), with `on_unbound = "freeze"`: reads the
    /// status buffer back (`8 + 2 cap` words). `drop_step` is exact in the GPU
    /// convention (see [`GpuStepper::status_buffer`]); the frozen state is the
    /// state at the start of the block of the loss.
    fn dropped(&mut self) -> Result<Option<DropReport>> {
        if !self.freeze {
            return Ok(None);
        }
        let gpu = self.kernels.gpu();
        let d = gpu.device();
        let words = 8 + 2 * self.cap;
        let size = (4 * words) as u64;
        let stage = self.drop_stage.get_or_insert_with(|| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dropped"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let mut enc = d.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(self.stepper.status_buffer(), 0, stage, 0, size);
        gpu.queue().submit([enc.finish()]);
        stage.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
        d.poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let w: Vec<u32> = {
            let view = stage.get_mapped_range(..).expect("range");
            view.as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect()
        };
        stage.unmap();
        let (n, cap) = (self.n, self.cap);
        let reason: Vec<u8> = (0..n).map(|i| w[8 + i] as u8).collect();
        let drop_step: Vec<u64> = (0..n)
            .map(|i| {
                if reason[i] == 0 {
                    u64::MAX
                } else {
                    u64::from(w[8 + cap + i])
                }
            })
            .collect();
        let n_moons = self.stepper.n_moons();
        Ok(Some(DropReport {
            time_s: self.time_s(),
            reason,
            drop_step,
            n_unbound: u64::from(w[1]),
            n_inside_moon: (0..n_moons).map(|k| u64::from(w[2 + k])).collect(),
        }))
    }

    fn stats(&mut self) -> Result<StepStats> {
        let max_rel_energy_error = if self.has_moons() || self.collide.is_some() {
            None
        } else {
            self.sync_snapshot()?;
            let s = &self.snapshot;
            for (i, &e0) in self.energy0.iter().enumerate() {
                let e = self
                    .field
                    .specific_energy(s.position_km(i), s.velocity_km_s(i));
                self.max_rel_energy_error = self.max_rel_energy_error.max(((e - e0) / e0).abs());
            }
            Some(self.max_rel_energy_error)
        };
        // Collision sums since the previous call (ICR-009).
        let tally = self
            .collide
            .as_mut()
            .map(|c| std::mem::take(&mut c.tally))
            .unwrap_or_default();
        Ok(StepStats {
            time_s: self.time_s(),
            steps_total: self.stepper.steps_total,
            max_rel_energy_error,
            collisions_since_last: tally.collisions,
            collision: tally.stats,
        })
    }
}

/// The initial state with every particle folded into the arc (task D1).
fn fold_into_arc(g: &ArcGeometry, init: &InitialState) -> Result<InitialState> {
    let mut particles = Particles::with_capacity(init.particles.len());
    for i in 0..init.particles.len() {
        let w = g
            .fold(
                init.particles.position_km(i),
                init.particles.velocity_km_s(i),
                init.time_s,
            )
            .ok_or_else(|| Error::Input(format!("particle {i} has no finite position")))?;
        particles.push(init.particles.id[i], w.pos_km, w.vel_km_s);
    }
    Ok(InitialState {
        time_s: init.time_s,
        particles,
    })
}

/// The second start-up check of ADR 0006: one short block of the real kernel
/// against its Rust twin on a few particles (relative < 1e-6, PLAN section
/// 10), because a primitive can pass alone and fail after inlining.
fn startup_check(
    k: &WhKernels,
    mode: Mode,
    system: &System,
    t0_s: f64,
    h_s: f64,
    a0_km: f64,
) -> Result<()> {
    const M: u8 = model::M4;
    const STEPS: usize = 4;
    let fail = |msg: String| {
        Err(Error::Backend(format!(
            "start-up check of the GPU kernel {} failed on {}: {msg}. The shader compiler may \
             have re-associated the guarded double-single sums (ADR 0006, M16).",
            mode.entry(),
            k.gpu().info.name
        )))
    };
    let eph = &system.moons;
    let gms: Vec<f64> = eph.tracks.iter().map(|t| t.gm_km3_s2).collect();
    let c = Consts::<M>::for_step_drift(&system.saturn, &gms, h_s, Some(a0_km));
    let scale = kick_scale(h_s);
    let mut table = KickTable::<f64>::for_ephemeris(eph);
    table.fill(eph, t0_s, h_s, 0, STEPS)?;
    let host = HostTable::<M>::from_kick_table(&table, scale);
    let field = SaturnField::<f64>::from_params(&system.saturn);
    let states: Vec<State<M>> = (0..32)
        .map(|j| {
            let el = EpicycleElements {
                a_km: 139_000.0 + 3_000.0 * (j % 8) as f64 / 8.0,
                e: [0.0, 0.0024, 0.01, 0.002][j % 4],
                i_rad: [0.0, 1.1e-4, 1e-3, 5e-4][j % 4],
                mean_anomaly_rad: 0.37 * j as f64,
                z_phase_rad: 0.11 * j as f64,
                mean_longitude_rad: 0.197 * j as f64,
            };
            let (p, v) = epicycle_state(&field, &el);
            state_from_f64(p, v)
        })
        .collect();
    let mut gpu_states = states.clone();
    let (status, _) = k.apply(
        mode.entry(),
        &wh::params(&c, &gms),
        &wh::pack_table(&table, scale),
        STEPS,
        eph.tracks.len(),
        &mut gpu_states,
    );
    if status != NO_FAILURE {
        return fail(format!("the kernel flagged particle {status}"));
    }
    let rel = |a: Vec3<f64>, b: Vec3<f64>| (a - b).norm() / b.norm();
    let mut worst = 0.0f64;
    for (s0, g) in states.iter().zip(&gpu_states) {
        let mut t = *s0;
        if wh::advance(mode, &c, &host, &mut t).is_none() {
            return fail("the Rust twin found an unbound orbit".to_string());
        }
        let ((tp, tv), (gp, gv)) = (state_to_f64(&t), state_to_f64(g));
        worst = worst.max(rel(gp, tp)).max(rel(gv, tv));
    }
    if worst.is_nan() || worst >= 1e-6 {
        return fail(format!(
            "GPU against Rust twin: relative difference {worst:.2e}, limit 1e-6"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_names() {
        assert_eq!(Mode::D.entry(), "step_d");
        assert_eq!(Mode::C.entry(), "step_c");
        assert_eq!(wh::MAX_MOONS, 4);
    }
}
