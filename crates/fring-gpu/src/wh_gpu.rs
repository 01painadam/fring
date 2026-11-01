//! The production kernels of `shaders/wh.wgsl` on the GPU: compiled pipelines,
//! the bind group layout and a one-shot [`WhKernels::apply`] for tests and the
//! start-up check. The persistent buffers of a run are in [`crate::backend`].

use crate::ds::Ds;
use crate::gpu::Gpu;
use crate::probe::self_test;
use crate::spike::State;
use crate::wh::ROW;
use fring_model::{Error, Result};

/// The WGSL source: DS library, sine and cosine, production kernels. The
/// text `WG_SIZE` is replaced by the workgroup size.
pub const WH_WGSL: &str = concat!(
    include_str!("../shaders/ds.wgsl"),
    include_str!("../shaders/trig.wgsl"),
    include_str!("../shaders/wh.wgsl")
);

/// Default workgroup size (see the sweep in the G2 handoff).
pub const DEFAULT_WORKGROUP: u32 = 64;

/// `status[0]` when no particle failed.
pub const NO_FAILURE: u32 = u32::MAX;

/// Compiled shader module and layout of the production kernels.
pub struct WhKernels {
    gpu: Gpu,
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    /// Workgroup size of the kernels.
    pub workgroup: u32,
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl WhKernels {
    /// Runs the double-single self-test (ADR 0006) and compiles the kernels.
    ///
    /// # Errors
    /// [`Error::Backend`] if the self-test fails.
    pub fn new(gpu: &Gpu, workgroup: u32) -> Result<Self> {
        self_test(gpu)?;
        let d = gpu.device();
        let src = WH_WGSL.replace("WG_SIZE", &workgroup.to_string());
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wh"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let st = |ro| wgpu::BufferBindingType::Storage { read_only: ro };
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wh"),
            entries: &[
                entry(0, st(false)),
                entry(1, st(true)),
                entry(2, wgpu::BufferBindingType::Uniform),
                entry(3, wgpu::BufferBindingType::Uniform),
                entry(4, st(false)),
            ],
        });
        let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wh"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        Ok(Self {
            gpu: gpu.clone(),
            module,
            layout,
            pipeline_layout,
            workgroup,
        })
    }

    /// The GPU.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The compute pipeline of the entry point `name`.
    pub fn pipeline(&self, name: &str) -> wgpu::ComputePipeline {
        self.gpu
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&self.pipeline_layout),
                module: &self.module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    /// A bind group for the five buffers of the kernels.
    pub fn bind_group(
        &self,
        state: &wgpu::Buffer,
        table: &wgpu::Buffer,
        prm: &wgpu::Buffer,
        ctl: &wgpu::Buffer,
        status: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        fn e(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }
        }
        self.gpu
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wh"),
                layout: &self.layout,
                entries: &[e(0, state), e(1, table), e(2, prm), e(3, ctl), e(4, status)],
            })
    }

    /// Runs the entry point `name` once on `states` with the given constants
    /// and kick table (see [`crate::wh`]) and returns the result and
    /// `status[0]` (the smallest index of a failed particle, or
    /// [`NO_FAILURE`]). Creates its buffers; for tests and the start-up check.
    pub fn apply<const M: u8>(
        &self,
        name: &str,
        prm: &[f32],
        table: &[f32],
        n_steps: usize,
        n_moons: usize,
        states: &mut [State<M>],
    ) -> (u32, std::time::Duration) {
        let n = states.len();
        let cap = n.next_multiple_of(self.workgroup as usize).max(1);
        let d = self.gpu.device();
        let q = self.gpu.queue();
        let bytes = |w: &[f32]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let ubytes = |w: &[u32]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let mk = |size: u64, usage| {
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
        let state_buf = mk((6 * cap * 8) as u64, rw);
        q.write_buffer(&state_buf, 0, &bytes(&pack_states(states, cap)));
        let table_buf = mk((table.len().max(ROW) * 4) as u64, rw);
        q.write_buffer(&table_buf, 0, &bytes(table));
        let uni = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let prm_buf = mk((prm.len() * 4) as u64, uni);
        q.write_buffer(&prm_buf, 0, &bytes(prm));
        let ctl_buf = mk(16, uni);
        q.write_buffer(
            &ctl_buf,
            0,
            &ubytes(&crate::wh::control(n_steps, n, cap, n_moons)),
        );
        let status_buf = mk(4, rw);
        q.write_buffer(&status_buf, 0, &NO_FAILURE.to_le_bytes());
        let bind = self.bind_group(&state_buf, &table_buf, &prm_buf, &ctl_buf, &status_buf);
        let pipe = self.pipeline(name);
        q.submit([]);
        d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bind, &[]);
            let (gx, gy) = grid(cap, self.workgroup);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        let t0 = std::time::Instant::now();
        q.submit([enc.finish()]);
        d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let elapsed = t0.elapsed();

        let stage = mk(
            (6 * cap * 8) as u64 + 16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut enc = d.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&state_buf, 0, &stage, 0, (6 * cap * 8) as u64);
        enc.copy_buffer_to_buffer(&status_buf, 0, &stage, (6 * cap * 8) as u64, 4);
        q.submit([enc.finish()]);
        stage.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
        d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let view = stage.get_mapped_range(..).expect("mapped range");
        let words: Vec<f32> = view[..6 * cap * 8]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let off = 6 * cap * 8;
        let status = u32::from_le_bytes([view[off], view[off + 1], view[off + 2], view[off + 3]]);
        unpack_states(&words, cap, states);
        (status, elapsed)
    }
}

/// The dispatch grid for `cap` particles (a multiple of `workgroup`): at most
/// 65535 workgroups in x, the rest in y. The shaders form the particle index
/// as `y * nw.x * workgroup + x`.
pub fn grid(cap: usize, workgroup: u32) -> (u32, u32) {
    let groups = (cap as u32).div_ceil(workgroup).max(1);
    let gx = groups.min(65535);
    (gx, groups.div_ceil(gx))
}

/// States as the 6 planes of the GPU: word `2 (c cap + i)` is hi, `+1` is lo.
pub fn pack_states<const M: u8>(states: &[State<M>], cap: usize) -> Vec<f32> {
    let mut w = vec![0f32; 12 * cap];
    for (i, s) in states.iter().enumerate() {
        for (c, d) in s.iter().enumerate() {
            w[2 * (c * cap + i)] = d.hi;
            w[2 * (c * cap + i) + 1] = d.lo;
        }
    }
    w
}

/// Inverse of [`pack_states`].
pub fn unpack_states<const M: u8>(w: &[f32], cap: usize, states: &mut [State<M>]) {
    for (i, s) in states.iter_mut().enumerate() {
        for (c, d) in s.iter_mut().enumerate() {
            *d = Ds {
                hi: w[2 * (c * cap + i)],
                lo: w[2 * (c * cap + i) + 1],
            };
        }
    }
}

/// An error for a failed GPU step.
pub fn unbound(id: u32, index: u32) -> Error {
    Error::Backend(format!(
        "particle {id} (index {index}) has an unbound or invalid orbit"
    ))
}
