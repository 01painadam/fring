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

use crate::gpu::Gpu;
use crate::spike::{Consts, State, kick_scale, state_from_f64, state_to_f64};
use crate::wf32::model;
use crate::wh::{self, HostTable, Mode, ROW};
use crate::wh_gpu::{DEFAULT_WORKGROUP, NO_FAILURE, WhKernels, grid, unbound};
use fring_model::config::ChaosConfig;
use fring_model::physics::chaos::shadow_offset_km;
use fring_model::physics::{
    EpicycleElements, KickTable, SaturnField, epicycle_state, interpolate_moons,
};
use fring_model::{
    Backend, ChaosReport, Error, InitialState, MoonEphemeris, MoonState, ParticleView, Particles,
    Result, RunConfig, StepStats, System, Vec3,
};
use rayon::prelude::*;

/// Number of table buffers in flight.
const RING: usize = 3;
/// Most steps in one dispatch.
pub const MAX_BLOCK_STEPS: usize = 4096;
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

/// Buffers and pipeline of the chaos indicator (ICR-003, ADR 0008); only
/// made with `[diagnostics.chaos]`.
struct GpuChaos {
    cfg: ChaosConfig,
    acc_buf: wgpu::Buffer,
    acc_stage: wgpu::Buffer,
    renorm_pipeline: wgpu::ComputePipeline,
    renorm_bind: wgpu::BindGroup,
}

struct Slot {
    table: wgpu::Buffer,
    ctl: wgpu::Buffer,
    bind: wgpu::BindGroup,
    /// As `bind`, with the shadow state (chaos only).
    bind_shadow: Option<wgpu::BindGroup>,
    last: Option<wgpu::SubmissionIndex>,
}

/// The wgpu backend (see the module documentation).
pub struct GpuBackend {
    kernels: WhKernels,
    mode: Mode,
    pipeline: wgpu::ComputePipeline,
    field: SaturnField<f64>,
    eph: MoonEphemeris,
    table: KickTable<f64>,
    scale: f64,
    step_s: f64,
    start_s: f64,
    steps_total: u64,
    ids: Vec<u32>,
    n: usize,
    cap: usize,
    block_steps: usize,
    n_moons: usize,
    state_buf: wgpu::Buffer,
    status_buf: wgpu::Buffer,
    status_stage: wgpu::Buffer,
    stage: wgpu::Buffer,
    slots: Vec<Slot>,
    next_slot: usize,
    snapshot: Particles,
    snapshot_stale: bool,
    energy0: Vec<f64>,
    max_rel_energy_error: f64,
    chaos: Option<GpuChaos>,
}

fn le_bytes(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
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
        if !wg.is_power_of_two() || wg > gpu.limits.max_compute_workgroup_size_x {
            return Err(Error::Config(format!(
                "workgroup size {wg} is not a power of two within the device limit"
            )));
        }
        let cap = n.next_multiple_of(wg as usize).max(wg as usize);
        let state_bytes = (6 * cap * 8) as u64;
        let (max_bind, max_buf) = (
            gpu.limits.max_storage_buffer_binding_size,
            gpu.limits.max_buffer_size,
        );
        if state_bytes > max_bind || state_bytes > max_buf {
            return Err(Error::Backend(format!(
                "{n} particles need a state buffer of {state_bytes} bytes; {} allows \
                 max_storage_buffer_binding_size {max_bind} and max_buffer_size {max_buf} bytes",
                gpu.info.name
            )));
        }
        let kernels = WhKernels::new(&gpu, wg)?;
        startup_check(&kernels, opts.mode, system, init.time_s, step_s)?;

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
        let rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let uni = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        // Constants.
        let gms: Vec<f64> = eph.tracks.iter().map(|t| t.gm_km3_s2).collect();
        let c = Consts::<{ model::M4 }>::for_step(&system.saturn, &gms, step_s);
        let prm_buf = mk((wh::PARAM_WORDS * 4) as u64, uni);
        q.write_buffer(&prm_buf, 0, &le_bytes(&wh::params(&c, &gms)));

        // State: 6 planes of DS values.
        let state_buf = mk(state_bytes, rw);
        let states: Vec<State<{ model::M4 }>> = (0..n)
            .map(|i| state_from_f64(p.position_km(i), p.velocity_km_s(i)))
            .collect();
        q.write_buffer(
            &state_buf,
            0,
            &le_bytes(&crate::wh_gpu::pack_states(&states, cap)),
        );
        let status_buf = mk(4, rw);
        q.write_buffer(&status_buf, 0, &NO_FAILURE.to_le_bytes());
        let block_steps = opts
            .block_steps
            .unwrap_or_else(|| (TARGET_PARTICLE_STEPS / n.max(1)).clamp(8, MAX_BLOCK_STEPS))
            .clamp(1, MAX_BLOCK_STEPS);
        let chaos_cfg = config.diagnostics.as_ref().and_then(|d| d.chaos);
        let shadow_buf = chaos_cfg.map(|cfg| {
            // The shadow starts d0 away in the radial direction (position only).
            let shadows: Vec<State<{ model::M4 }>> = (0..n)
                .map(|i| {
                    let dir = shadow_offset_km(p.position_km(i), cfg.offset_km);
                    state_from_f64(p.position_km(i) + dir, p.velocity_km_s(i))
                })
                .collect();
            let buf = mk(state_bytes, rw);
            q.write_buffer(
                &buf,
                0,
                &le_bytes(&crate::wh_gpu::pack_states(&shadows, cap)),
            );
            buf
        });
        let slots = (0..RING)
            .map(|_| {
                let table = mk((block_steps * ROW * 4) as u64, rw);
                let ctl = mk(16, uni);
                let bind = kernels.bind_group(&state_buf, &table, &prm_buf, &ctl, &status_buf);
                let bind_shadow = shadow_buf
                    .as_ref()
                    .map(|sb| kernels.bind_group(sb, &table, &prm_buf, &ctl, &status_buf));
                Slot {
                    table,
                    ctl,
                    bind,
                    bind_shadow,
                    last: None,
                }
            })
            .collect();
        let chaos = match (chaos_cfg, shadow_buf) {
            (Some(cfg), Some(shadow_buf)) => {
                let acc_bytes = (crate::chaos::ACC_PLANES * cap * 8) as u64;
                let acc_buf = mk(acc_bytes, rw); // zero: j = 0, sums 0
                let cprm = mk(16, uni);
                q.write_buffer(&cprm, 0, &le_bytes(&[cfg.offset_km as f32, 0.0, 0.0, 0.0]));
                let ctl = mk(16, uni);
                let ctl_words = wh::control(1, n, cap, eph.tracks.len());
                let ctl_bytes: Vec<u8> = ctl_words.iter().flat_map(|x| x.to_le_bytes()).collect();
                q.write_buffer(&ctl, 0, &ctl_bytes);
                let renorm_bind = kernels.chaos_bind_group(
                    &state_buf,
                    &prm_buf,
                    &ctl,
                    &shadow_buf,
                    &acc_buf,
                    &cprm,
                );
                Some(GpuChaos {
                    cfg,
                    acc_buf,
                    acc_stage: mk(acc_bytes, read),
                    renorm_pipeline: kernels.chaos_pipeline(),
                    renorm_bind,
                })
            }
            _ => None,
        };

        let field = SaturnField::<f64>::from_params(&system.saturn);
        let energy0 = if eph.tracks.is_empty() {
            (0..n)
                .map(|i| field.specific_energy(p.position_km(i), p.velocity_km_s(i)))
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            pipeline: kernels.pipeline(opts.mode.entry()),
            mode: opts.mode,
            field,
            eph: eph.clone(),
            table: KickTable::for_ephemeris(eph),
            scale: kick_scale(step_s),
            step_s,
            start_s: init.time_s,
            steps_total: 0,
            ids: p.id.clone(),
            n,
            cap,
            block_steps,
            n_moons: eph.tracks.len(),
            state_buf,
            status_buf,
            status_stage: mk(16, read),
            stage: mk(state_bytes, read),
            slots,
            next_slot: 0,
            snapshot: p.clone(),
            snapshot_stale: false,
            energy0,
            max_rel_energy_error: 0.0,
            chaos,
            kernels,
        })
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
        self.block_steps
    }

    fn has_moons(&self) -> bool {
        self.n_moons > 0
    }

    /// Waits for the GPU and returns the failure flag.
    fn failure_flag(&mut self) -> u32 {
        let gpu = self.kernels.gpu();
        let d = gpu.device();
        let mut enc = d.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.status_buf, 0, &self.status_stage, 0, 4);
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
        enc.copy_buffer_to_buffer(&self.state_buf, 0, &self.stage, 0, size);
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
        let gpu = self.kernels.gpu().clone();
        let (d, q) = (gpu.device(), gpu.queue());
        let (gx, gy) = grid(self.cap, self.kernels.workgroup);
        let mut done = 0u64;
        while done < n_steps {
            let first = self.steps_total + done;
            let mut block = (n_steps - done).min(self.block_steps as u64);
            if let Some(c) = &self.chaos {
                // Blocks end at multiples of K (as in CpuBackend), so the
                // renormalisation steps do not depend on how the caller
                // splits the run.
                let k = u64::from(c.cfg.renorm_steps);
                block = block.min(k - first % k);
            }
            let renorm_now = self
                .chaos
                .as_ref()
                .is_some_and(|c| (first + block).is_multiple_of(u64::from(c.cfg.renorm_steps)));
            let block = block as usize;
            // The host works while the GPU runs the previous blocks.
            self.table
                .fill(&self.eph, self.start_s, self.step_s, first, block)?;
            let packed = wh::pack_table(&self.table, self.scale);
            let slot = &mut self.slots[self.next_slot];
            if let Some(idx) = slot.last.take() {
                d.poll(wgpu::PollType::Wait {
                    submission_index: Some(idx),
                    timeout: None,
                })
                .expect("GPU poll failed");
            }
            q.write_buffer(&slot.table, 0, &le_bytes(&packed));
            let ctl = wh::control(block, self.n, self.cap, self.n_moons);
            let ctl_bytes: Vec<u8> = ctl.iter().flat_map(|x| x.to_le_bytes()).collect();
            q.write_buffer(&slot.ctl, 0, &ctl_bytes);
            let mut enc = d.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &slot.bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
                if let (Some(c), Some(bs)) = (&self.chaos, &slot.bind_shadow) {
                    pass.set_bind_group(0, bs, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                    if renorm_now {
                        pass.set_pipeline(&c.renorm_pipeline);
                        pass.set_bind_group(0, &c.renorm_bind, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
            }
            slot.last = Some(q.submit([enc.finish()]));
            self.next_slot = (self.next_slot + 1) % RING;
            done += block as u64;
        }
        self.snapshot_stale = true;
        let flag = self.failure_flag();
        if flag != NO_FAILURE {
            let idx = flag as usize;
            let id = self.ids.get(idx).copied().unwrap_or(u32::MAX);
            return Err(unbound(id, flag));
        }
        self.steps_total += n_steps;
        Ok(())
    }

    fn time_s(&self) -> f64 {
        self.start_s + self.steps_total as f64 * self.step_s
    }

    fn moons(&self) -> MoonState {
        let t = self.time_s();
        interpolate_moons(&self.eph, t).unwrap_or(MoonState {
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
        let Some(c) = &self.chaos else {
            return Ok(None);
        };
        let gpu = self.kernels.gpu();
        let d = gpu.device();
        let size = (crate::chaos::ACC_PLANES * self.cap * 8) as u64;
        let mut enc = d.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&c.acc_buf, 0, &c.acc_stage, 0, size);
        gpu.queue().submit([enc.finish()]);
        c.acc_stage
            .map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
        d.poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let words: Vec<f32> = {
            let view = c.acc_stage.get_mapped_range(..).expect("range");
            view.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        c.acc_stage.unmap();
        let interval_s = f64::from(c.cfg.renorm_steps) * self.step_s;
        Ok(Some(crate::chaos::report(
            &words,
            self.cap,
            self.n,
            interval_s,
            self.time_s(),
        )))
    }

    fn stats(&mut self) -> Result<StepStats> {
        let max_rel_energy_error = if self.has_moons() {
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
        Ok(StepStats {
            time_s: self.time_s(),
            steps_total: self.steps_total,
            max_rel_energy_error,
            ..Default::default()
        })
    }
}

/// The second start-up check of ADR 0006: one short block of the real kernel
/// against its Rust twin on a few particles (relative < 1e-6, PLAN section
/// 10), because a primitive can pass alone and fail after inlining.
fn startup_check(k: &WhKernels, mode: Mode, system: &System, t0_s: f64, h_s: f64) -> Result<()> {
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
    let c = Consts::<M>::for_step(&system.saturn, &gms, h_s);
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
