//! [`GpuStepper`]: the non-blocking core of [`crate::backend::GpuBackend`]
//! (ICR-005 item 5).
//!
//! The stepper owns the buffers and the pipelines and records the work of
//! `n` steps into a command encoder that the caller submits. It never waits
//! for the GPU and never reads the clock, so it also builds for wasm32 and
//! runs in a browser. [`crate::backend::GpuBackend`] adds the blocking parts
//! (the ring of slots with its poll, the read-back, the failure flag).
//!
//! # State buffer layout (part of the interface)
//!
//! [`GpuStepper::state_buffer`] has usage `STORAGE | COPY_SRC | COPY_DST`
//! and holds 6 planes in this order: x, y, z, vx, vy, vz (Saturn-centred,
//! inertial; km and km/s). Each plane has [`GpuStepper::capacity`] entries
//! (the particle count rounded up to a multiple of the workgroup size). The
//! value of particle `i` in plane `c` is a double-single pair stored as two
//! f32 words: word `2 (c cap + i)` is the high part and word
//! `2 (c cap + i) + 1` is the low part. The value is `high + low`.

use crate::spike::{Consts, State, kick_scale, state_from_f64};
use crate::wf32::model;
use crate::wh::{self, ArcRole, Mode, ROW};
use crate::wh_gpu::{WhKernels, grid, status_init_bytes, status_words};
use fring_model::config::ChaosConfig;
use fring_model::physics::chaos::shadow_offset_km;
use fring_model::physics::{ArcGeometry, KickTable};
use fring_model::{Error, InitialState, MoonEphemeris, Result, System};

/// Number of table buffers in flight.
pub(crate) const RING: usize = 3;
/// Most steps in one dispatch.
pub const MAX_BLOCK_STEPS: usize = 4096;

/// Buffers and pipeline of the chaos indicator (ICR-003, ADR 0008); only
/// made with a chaos config.
struct GpuChaos {
    cfg: ChaosConfig,
    acc_buf: wgpu::Buffer,
    renorm_pipeline: wgpu::ComputePipeline,
    renorm_bind: wgpu::BindGroup,
}

/// The arc domain of task D1 (see [`crate::arc`]): the geometry and the
/// pipeline of the shadow (the pipeline of the particles is `pipeline`).
struct GpuArc {
    geom: ArcGeometry,
    shadow_pipeline: Option<wgpu::ComputePipeline>,
}

/// The buffers of one block: kick table and control words.
struct Slot {
    table: wgpu::Buffer,
    ctl: wgpu::Buffer,
    bind: wgpu::BindGroup,
    /// As `bind`, with the shadow state (chaos only).
    bind_shadow: Option<wgpu::BindGroup>,
    /// Last submission that used this slot (set by the blocking backend only).
    last: Option<wgpu::SubmissionIndex>,
}

pub(crate) fn le_bytes(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The padded particle count for `n` particles, after the checks of the
/// workgroup size and of the buffer limits of the device.
pub(crate) fn plan_capacity(gpu: &crate::gpu::Gpu, n: usize, workgroup: u32) -> Result<usize> {
    if !workgroup.is_power_of_two() || workgroup > gpu.limits.max_compute_workgroup_size_x {
        return Err(Error::Config(format!(
            "workgroup size {workgroup} is not a power of two within the device limit"
        )));
    }
    let cap = n
        .next_multiple_of(workgroup as usize)
        .max(workgroup as usize);
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
    Ok(cap)
}

/// Records the steps of the Wisdom-Holman kernels into a command encoder
/// (see the module documentation).
pub struct GpuStepper {
    kernels: WhKernels,
    pipeline: wgpu::ComputePipeline,
    eph: MoonEphemeris,
    table: KickTable<f64>,
    scale: f64,
    step_s: f64,
    start_s: f64,
    /// Steps recorded so far. The blocking backend sets it back when a call fails.
    pub(crate) steps_total: u64,
    n: usize,
    cap: usize,
    block_steps: usize,
    n_moons: usize,
    state_buf: wgpu::Buffer,
    status_buf: wgpu::Buffer,
    prm_buf: wgpu::Buffer,
    shadow_buf: Option<wgpu::Buffer>,
    mode: Mode,
    /// The words of the `prm` uniform with the rule off.
    prm_words: Vec<f32>,
    /// The moon radii of the rule (ICR-006), km; `None` = rule off.
    radii_km: Option<Vec<f64>>,
    slots: Vec<Slot>,
    next_slot: usize,
    chaos: Option<GpuChaos>,
    arc: Option<GpuArc>,
}

impl GpuStepper {
    /// Makes the buffers and uploads the constants and the states. Does not
    /// block. `block_steps` is clamped to `1..=MAX_BLOCK_STEPS`.
    ///
    /// # Errors
    /// [`Error::Config`] or [`Error::Backend`] if the workgroup size or the
    /// number of particles does not fit the device; [`Error::Input`] if there
    /// are more moons than the kernels take.
    #[allow(clippy::too_many_arguments)] // the signature of ICR-005 item 5
    pub fn new(
        gpu: &crate::gpu::Gpu,
        kernels: &WhKernels,
        system: &System,
        init: &InitialState,
        step_s: f64,
        block_steps: usize,
        mode: Mode,
        chaos: Option<ChaosConfig>,
    ) -> Result<Self> {
        let eph = &system.moons;
        wh::check_moons(eph.tracks.len())?;
        if chaos.is_some_and(|c| c.renorm_steps == 0) {
            return Err(Error::Config("renorm_steps must be at least 1".to_string()));
        }
        let p = &init.particles;
        let n = p.len();
        let cap = plan_capacity(gpu, n, kernels.workgroup)?;
        let state_bytes = (6 * cap * 8) as u64;

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

        // Constants.
        let gms: Vec<f64> = eph.tracks.iter().map(|t| t.gm_km3_s2).collect();
        let c = Consts::<{ model::M4 }>::for_step(&system.saturn, &gms, step_s);
        let prm_buf = mk((wh::PARAM_WORDS * 4) as u64, uni);
        let prm_words = wh::params(&c, &gms);
        q.write_buffer(&prm_buf, 0, &le_bytes(&prm_words));

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
        let status_buf = mk((4 * status_words(cap)) as u64, rw);
        q.write_buffer(&status_buf, 0, &status_init_bytes(cap));
        let block_steps = block_steps.clamp(1, MAX_BLOCK_STEPS);
        let shadow_buf = chaos.map(|cfg| {
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
        let mut s = Self {
            kernels: kernels.clone(),
            pipeline: kernels.pipeline(mode.entry()),
            eph: eph.clone(),
            table: KickTable::for_ephemeris(eph),
            scale: kick_scale(step_s),
            step_s,
            start_s: init.time_s,
            steps_total: 0,
            n,
            cap,
            block_steps,
            n_moons: eph.tracks.len(),
            state_buf,
            status_buf,
            prm_buf,
            shadow_buf,
            mode,
            prm_words,
            radii_km: None,
            slots: Vec::new(),
            next_slot: 0,
            chaos: None,
            arc: None,
        };
        for _ in 0..RING {
            let slot = s.make_slot();
            s.slots.push(slot);
        }
        s.chaos = match (chaos, s.shadow_buf.as_ref()) {
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
                    &s.state_buf,
                    &s.prm_buf,
                    &ctl,
                    shadow_buf,
                    &acc_buf,
                    &cprm,
                );
                Some(GpuChaos {
                    cfg,
                    acc_buf,
                    renorm_pipeline: kernels.chaos_pipeline(),
                    renorm_bind,
                })
            }
            _ => None,
        };
        Ok(s)
    }

    fn make_slot(&self) -> Slot {
        let d = self.kernels.gpu().device();
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
        // Two more rows after the kick rows: the moon positions at the sync
        // times of the block (the radius rule of ICR-006).
        let table = mk(((self.block_steps + 2) * ROW * 4) as u64, rw);
        let ctl = mk(16, uni);
        let k = &self.kernels;
        let bind = k.bind_group(
            &self.state_buf,
            &table,
            &self.prm_buf,
            &ctl,
            &self.status_buf,
        );
        let bind_shadow = self
            .shadow_buf
            .as_ref()
            .map(|sb| k.bind_group(sb, &table, &self.prm_buf, &ctl, &self.status_buf));
        Slot {
            table,
            ctl,
            bind,
            bind_shadow,
            last: None,
        }
    }

    /// Records the steps of the next block (at most `max_steps`, and at most
    /// `block_steps`; with chaos, up to the next multiple of the
    /// renormalisation interval) and returns the number of steps and the slot
    /// that it used.
    ///
    /// The kick table and the control words are written with
    /// `queue.write_buffer`, which the queue applies at the next submit. So
    /// two blocks in one encoder need two slots: `used_in_call` is the number
    /// of blocks that the same encoder already holds, and the stepper adds a
    /// slot if all slots are taken. `wait` gets the last submission of the
    /// slot, after the host has filled the table and before it overwrites the
    /// slot's buffers.
    pub(crate) fn encode_block(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        max_steps: u64,
        used_in_call: usize,
        wait: impl FnOnce(Option<wgpu::SubmissionIndex>),
    ) -> Result<(u64, usize)> {
        let first = self.steps_total;
        let mut block = max_steps.min(self.block_steps as u64);
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
        let mut packed = wh::pack_table(&self.table, self.scale);
        if let Some(a) = &self.arc {
            crate::arc::set_kick_angles(
                &mut packed,
                &a.geom,
                self.start_s,
                self.step_s,
                first,
                block,
            );
        }
        let idx = if used_in_call >= self.slots.len() {
            let slot = self.make_slot();
            self.slots.push(slot);
            self.slots.len() - 1
        } else {
            self.next_slot
        };
        wait(self.slots[idx].last.take());
        let (gx, gy) = grid(self.cap, self.kernels.workgroup);
        let q = self.kernels.gpu().queue();
        let slot = &self.slots[idx];
        q.write_buffer(&slot.table, 0, &le_bytes(&packed));
        if self.radii_km.is_some() {
            // The moons at the sync times of the block, for the rule (rows n, n + 1).
            let rows = wh::sync_rows(&self.eph, self.start_s, self.step_s, first, block);
            q.write_buffer(&slot.table, (block * ROW * 4) as u64, &le_bytes(&rows));
        }
        if let Some(a) = &self.arc {
            // The arc centre at the sync times and the constants of the wrap.
            let rows = crate::arc::extra_rows(&a.geom, self.start_s, self.step_s, first, block);
            q.write_buffer(&slot.table, (block * ROW * 4) as u64, &le_bytes(&rows));
        }
        let ctl = wh::control(block, self.n, self.cap, self.n_moons);
        let ctl_bytes: Vec<u8> = ctl.iter().flat_map(|x| x.to_le_bytes()).collect();
        q.write_buffer(&slot.ctl, 0, &ctl_bytes);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &slot.bind, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
            if let (Some(c), Some(bs)) = (&self.chaos, &slot.bind_shadow) {
                if let Some(sp) = self.arc.as_ref().and_then(|a| a.shadow_pipeline.as_ref()) {
                    pass.set_pipeline(sp);
                }
                pass.set_bind_group(0, bs, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
                if renorm_now {
                    pass.set_pipeline(&c.renorm_pipeline);
                    pass.set_bind_group(0, &c.renorm_bind, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
        }
        self.next_slot = (idx + 1) % self.slots.len();
        self.steps_total += block as u64;
        Ok((block as u64, idx))
    }

    /// Records `n_steps` steps into `encoder`: the same blocks and the same
    /// chaos renormalisation steps as `GpuBackend::step`. Fills the moon
    /// kick table on the host and writes it with `queue.write_buffer`. The
    /// caller submits the encoder.
    ///
    /// **Submit the encoder before the next call of `encode_steps`.** Calls
    /// reuse a ring of table buffers, and `write_buffer` takes effect at the
    /// next submit, so a second call before the submit would overwrite the
    /// tables of the first. Inside one call every block has its own buffers.
    ///
    /// The call does not check the failure flag; read
    /// [`Self::status_buffer`] when you need it.
    ///
    /// # Errors
    /// [`Error::Input`] if the moon ephemeris does not cover the steps. The
    /// blocks recorded before the error stay in the encoder and are counted
    /// in [`Self::time_s`].
    pub fn encode_steps(&mut self, encoder: &mut wgpu::CommandEncoder, n_steps: u64) -> Result<()> {
        let (mut done, mut used) = (0u64, 0usize);
        while done < n_steps {
            let (b, _) = self.encode_block(encoder, n_steps - done, used, |_| {})?;
            done += b;
            used += 1;
        }
        Ok(())
    }

    /// Switches the moon-radius rule of ICR-006 on, with one radius in km for
    /// each moon (the moon order of the `System`; the mean radii `radius_km`
    /// of `params/moons.toml`). A radius of 0 leaves that moon out. For each
    /// drift between two kicks (and the half drifts at the ends of a block)
    /// the kernel interpolates the position of the particle relative to each
    /// moon linearly between the two ends and finds the minimum distance; if
    /// it is less than the radius of moon `k`, the particle is dropped with
    /// reason `2 + k` (see [`Self::status_buffer`]). A drift whose ends have no
    /// moon position (a time outside the ephemeris at the very end of the
    /// run) is not checked.
    ///
    /// Call it before the first step. The rule is off by default; with it off
    /// the results are bit for bit those of a stepper without this method.
    ///
    /// # Errors
    /// [`Error::Config`] if steps were already recorded, if the mode is C, if
    /// chaos is on, if the number of radii differs from the number of moons
    /// (at most 4: the kernels take 4 moons, and the status words for moons 4
    /// and 5 stay reserved), or if a radius is negative or not finite.
    pub fn set_moon_radii_km(&mut self, radii_km: &[f64]) -> Result<()> {
        let err = |m: String| Err(Error::Config(m));
        if self.steps_total > 0 {
            return err("set_moon_radii_km must be called before the first step".into());
        }
        let Some(entry) = self.mode.entry_rule() else {
            return err("the moon-radius rule is not available in mode C".into());
        };
        if self.chaos.is_some() {
            return err("the moon-radius rule cannot be combined with chaos".into());
        }
        if self.arc.is_some() {
            return err("the moon-radius rule cannot be combined with the arc".into());
        }
        if radii_km.len() != self.n_moons {
            return err(format!(
                "{} moon radii for {} moons",
                radii_km.len(),
                self.n_moons
            ));
        }
        if radii_km.iter().any(|r| !(r.is_finite() && *r >= 0.0)) {
            return err("moon radii must be finite and not negative".into());
        }
        let words = wh::with_radii(&self.prm_words, radii_km);
        let q = self.kernels.gpu().queue();
        q.write_buffer(&self.prm_buf, 0, &le_bytes(&words));
        self.pipeline = self.kernels.pipeline(entry);
        self.radii_km = Some(radii_km.to_vec());
        Ok(())
    }

    /// Switches the co-moving arc domain of task D1 on: after each of the
    /// wrap sites of [`crate::arc`] (start of a block, before each kick, end of
    /// a block) a particle outside the arc is rotated by a multiple of the
    /// width, in position and velocity, so that every particle is inside the
    /// arc at the end of every block. With chaos, the shadow of a particle
    /// takes the same rotations as the particle.
    ///
    /// Call it before the first step. With no arc the kernels are those of a
    /// stepper without this method, bit for bit.
    ///
    /// # Errors
    /// [`Error::Config`] if steps were already recorded, if the mode is C, or
    /// if the moon-radius rule is on.
    pub fn set_arc(&mut self, geom: ArcGeometry) -> Result<()> {
        let err = |m: &str| Err(Error::Config(m.to_string()));
        if self.steps_total > 0 {
            return err("set_arc must be called before the first step");
        }
        if self.radii_km.is_some() {
            return err("the arc cannot be combined with the moon-radius rule");
        }
        let role = if self.chaos.is_some() {
            ArcRole::Log
        } else {
            ArcRole::Plain
        };
        let Some(entry) = self.mode.entry_arc(role) else {
            return err("the arc is not available in mode C");
        };
        self.pipeline = self.kernels.pipeline(entry);
        let shadow_pipeline = if self.chaos.is_some() {
            self.mode
                .entry_arc(ArcRole::Replay)
                .map(|e| self.kernels.pipeline(e))
        } else {
            None
        };
        if self.chaos.is_some() {
            // The wrap log of the chaos shadows lives after the status words.
            let words = status_words(self.cap) + self.cap * crate::arc::ARC_LOG_WORDS;
            let mut init = status_init_bytes(self.cap);
            init.resize(4 * words, 0);
            let d = self.kernels.gpu().device();
            self.status_buf = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (4 * words) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.kernels
                .gpu()
                .queue()
                .write_buffer(&self.status_buf, 0, &init);
            let n = self.slots.len();
            self.slots = (0..n).map(|_| self.make_slot()).collect();
            self.next_slot = 0;
        }
        self.arc = Some(GpuArc {
            geom,
            shadow_pipeline,
        });
        Ok(())
    }

    /// The arc geometry, if the arc is on.
    pub fn arc(&self) -> Option<&ArcGeometry> {
        self.arc.as_ref().map(|a| &a.geom)
    }

    /// The moon radii of the rule, km, if it is on.
    pub fn moon_radii_km(&self) -> Option<&[f64]> {
        self.radii_km.as_deref()
    }

    /// Time of the recorded state, s: `start + steps * step_s`.
    pub fn time_s(&self) -> f64 {
        self.start_s + self.steps_total as f64 * self.step_s
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.n
    }

    /// True if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Entries per plane of the state buffer (at least `len`, a multiple of
    /// the workgroup size).
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// The state buffer (usage `STORAGE | COPY_SRC | COPY_DST`); layout in
    /// the module documentation.
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buf
    }

    /// The status buffer (ICR-006), `8 + capacity` `u32` words, usage
    /// `STORAGE | COPY_SRC | COPY_DST`:
    /// - word 0: the smallest index of a dropped particle, or `u32::MAX` if
    ///   none was dropped (what [`crate::backend::GpuBackend`] reads);
    /// - word 1: the number of particles dropped for an unbound or invalid
    ///   orbit;
    /// - words 2 to 7: the number dropped inside moon 0 to 5 (the kernels take
    ///   4 moons, so words 6 and 7 stay reserved);
    /// - word `8 + i`: the drop reason of particle `i`: 0 live, 1 unbound or
    ///   invalid orbit, `2 + k` inside moon `k`.
    ///
    /// The step kernels of modes D and Dp set the reason of a failed particle
    /// and then skip it. Mode C only sets word 0.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status_buf
    }

    /// Steps per dispatch.
    pub fn block_steps(&self) -> usize {
        self.block_steps
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_last(&mut self, slot: usize, idx: wgpu::SubmissionIndex) {
        self.slots[slot].last = Some(idx);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn ephemeris(&self) -> &MoonEphemeris {
        &self.eph
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn step_s(&self) -> f64 {
        self.step_s
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn n_moons(&self) -> usize {
        self.n_moons
    }

    /// The chaos accumulator buffer (usage `STORAGE | COPY_SRC | COPY_DST`;
    /// [`crate::chaos::ACC_PLANES`] planes of `capacity` double-single
    /// values), if chaos is on.
    pub fn chaos_buffer(&self) -> Option<&wgpu::Buffer> {
        self.chaos.as_ref().map(|c| &c.acc_buf)
    }

    /// The chaos configuration and accumulator buffer, if chaos is on.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn chaos_parts(&self) -> Option<(ChaosConfig, &wgpu::Buffer)> {
        self.chaos.as_ref().map(|c| (c.cfg, &c.acc_buf))
    }
}
