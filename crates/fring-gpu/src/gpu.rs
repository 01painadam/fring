//! A small wgpu helper: open the GPU, run one compute kernel on storage
//! buffers of 32-bit words, read the buffers back.
//!
//! This is the minimum that the precision probe and the benchmark of task G1
//! need. Task G2 replaces it with the real backend (persistent buffers,
//! moon table, blocks of steps).

use std::time::{Duration, Instant};

/// An open GPU device and its queue (cheap to clone: the handles are shared).
#[derive(Clone)]
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Name, backend and driver of the adapter.
    pub info: wgpu::AdapterInfo,
    /// The limits of the device (those of the adapter).
    pub limits: wgpu::Limits,
}

/// A storage buffer for [`Gpu::run`]: 32-bit words, read and written by the kernel.
pub type Words = Vec<u32>;

/// Converts f32 values to their bit patterns (exact).
pub fn to_words(x: &[f32]) -> Words {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Converts bit patterns back to f32 values (exact).
pub fn from_words(w: &[u32]) -> Vec<f32> {
    w.iter().map(|&b| f32::from_bits(b)).collect()
}

impl Gpu {
    /// Opens the default high-performance adapter. Returns `None` if there is
    /// no GPU (for example on a CI runner), so callers can skip GPU work.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fring-gpu"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .ok()?;
        Some(Self {
            device,
            queue,
            info,
            limits,
        })
    }

    /// The wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Compiles the entry point `entry` of the WGSL source `shader`.
    ///
    /// # Panics
    /// If the shader does not compile.
    pub fn kernel(&self, shader: &str, entry: &str) -> Kernel<'_> {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        Kernel {
            gpu: self,
            pipeline,
        }
    }

    /// Compiles `entry` and runs it once; see [`Kernel::run`].
    pub fn run(
        &self,
        shader: &str,
        entry: &str,
        buffers: &mut [Words],
        uniform: Option<&[u32]>,
        workgroups: u32,
        dispatches: u32,
    ) -> Duration {
        self.kernel(shader, entry)
            .run(buffers, uniform, workgroups, dispatches)
    }
}

/// A compiled compute kernel (one entry point).
pub struct Kernel<'g> {
    gpu: &'g Gpu,
    pipeline: wgpu::ComputePipeline,
}

impl Kernel<'_> {
    /// Runs the kernel `dispatches` times with `workgroups` workgroups in x.
    ///
    /// Binding `i` of group 0 is `buffers[i]` (storage; the shader may
    /// declare it `read` or `read_write`). If `uniform` is given, it is the
    /// next binding (uniform buffer). After the run, every storage buffer is
    /// read back into `buffers`. Returns the wall time from the submit to the
    /// end of the GPU work (without upload and read-back).
    ///
    /// # Panics
    /// If the bindings do not match the shader or the device is lost.
    pub fn run(
        &self,
        buffers: &mut [Words],
        uniform: Option<&[u32]>,
        workgroups: u32,
        dispatches: u32,
    ) -> Duration {
        let d = &self.gpu.device;
        let queue = &self.gpu.queue;
        let bytes = |w: &[u32]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let storage: Vec<wgpu::Buffer> = buffers
            .iter()
            .map(|w| {
                let b = d.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (w.len().max(1) * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&b, 0, &bytes(w));
                b
            })
            .collect();
        let uni = uniform.map(|u| {
            let b = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (u.len().max(4) * 4).next_multiple_of(16) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&b, 0, &bytes(u));
            b
        });
        let mut entries: Vec<wgpu::BindGroupEntry> = storage
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        if let Some(b) = &uni {
            entries.push(wgpu::BindGroupEntry {
                binding: storage.len() as u32,
                resource: b.as_entire_binding(),
            });
        }
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        // Upload first, so the timing below holds only the kernels.
        queue.submit([]);
        d.poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");

        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            for _ in 0..dispatches {
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        let t0 = Instant::now();
        queue.submit([enc.finish()]);
        d.poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        let elapsed = t0.elapsed();

        let staging: Vec<wgpu::Buffer> = storage
            .iter()
            .map(|b| {
                d.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: b.size(),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let mut enc = d.create_command_encoder(&Default::default());
        for (s, b) in staging.iter().zip(&storage) {
            enc.copy_buffer_to_buffer(b, 0, s, 0, b.size());
        }
        queue.submit([enc.finish()]);
        for s in &staging {
            s.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
        }
        d.poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        for (w, s) in buffers.iter_mut().zip(&staging) {
            let view = s.get_mapped_range(..).expect("mapped range");
            let (chunks, _) = view.as_chunks::<4>();
            for (x, c) in w.iter_mut().zip(chunks) {
                *x = u32::from_le_bytes(*c);
            }
        }
        elapsed
    }
}
