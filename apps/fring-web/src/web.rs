//! The web part (wasm32 only): the canvas, the render and projection pipelines,
//! `fring_gpu::GpuStepper` and the functions that the page (`index.html`) calls.
//! The page owns the DOM and the animation loop; this module owns the GPU.
//!
//! The pure parts (set-up, step planning, views, projection maths) are in
//! `live.rs` and `view.rs`, which the host tests cover.
//!
//! Start-up (ICR-005): `App::create` opens the device and starts the GPU
//! self-test. `frame` polls it once per frame. If it fails, the page shows the
//! message and the integrator never starts. If it passes, the next frame
//! compiles the kernels, builds the moon ephemeris and makes the stepper.

use crate::live::{
    BLOCK_STEPS, Band, END_MARGIN_S, FLOATS_PER_INSTANCE, Governor, MAX_IN_FLIGHT, PANDORA,
    PARTICLE_COUNT, ProjectionUniform, SPAN_S, STEP_S, Setup, StepPlanner, ViewFrame, count_fits,
    head_instances, instance_bytes, max_hours_for, view_transform,
};
use crate::view::Camera;
use fring_gpu::gpu::Gpu;
use fring_gpu::probe::SelfTest;
use fring_gpu::stepper::GpuStepper;
use fring_gpu::wh::Mode;
use fring_gpu::wh_gpu::{DEFAULT_WORKGROUP, NO_FAILURE, WhKernels};
use fring_model::MoonEphemeris;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use wasm_bindgen::prelude::*;

const SHADER: &str = include_str!("../shaders/points.wgsl");
const PROJECT_SHADER: &str = include_str!("../shaders/project.wgsl");

/// How often the failure flag of the stepper is read back, ms.
const STATUS_PERIOD_MS: f64 = 1000.0;

// States of the read-back of the failure flag.
const READ_IDLE: u8 = 0;
const READ_PENDING: u8 = 1;
const READ_READY: u8 = 2;
const READ_FAILED: u8 = 3;

enum Phase {
    /// The GPU self-test runs; polled once per frame.
    SelfTesting(Box<SelfTest>),
    /// The self-test passed; the next frame builds the kernels and the stepper.
    Preparing,
    Running,
    /// A message for the page; nothing runs.
    Failed(String),
}

/// A small `MAP_READ` buffer for `status_buffer()` and the state of its map.
struct StatusReader {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    last_ms: f64,
}

#[wasm_bindgen]
pub struct App {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    adapter_name: String,
    // Render.
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instances: wgpu::Buffer,
    /// Particle count that `instances` was sized for.
    instances_n: u32,
    // Projection of the double-single state into `instances`.
    project_pipeline: wgpu::ComputePipeline,
    project_uniform: wgpu::Buffer,
    project_bind: Option<wgpu::BindGroup>,
    // Simulation.
    phase: Phase,
    setup: Setup,
    full: Option<MoonEphemeris>,
    kernels: Option<WhKernels>,
    stepper: Option<GpuStepper>,
    pandora_on: bool,
    /// Number of particles of the running (or next) stepper.
    count: u32,
    band: Band,
    view: ViewFrame,
    camera: Camera,
    hours_per_s: f64,
    paused: bool,
    last_ms: Option<f64>,
    planner: StepPlanner,
    governor: Governor,
    in_flight: Arc<AtomicU32>,
    status: StatusReader,
    failed_index: Option<u32>,
    steps_last: u64,
    limited: bool,
    note: String,
    scratch: Vec<f32>,
    bytes: Vec<u8>,
}

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// The camera that a view starts with (and goes back to on "Reset view").
fn initial_camera(setup: &Setup, view: ViewFrame) -> Camera {
    match view {
        ViewFrame::Inertial | ViewFrame::Rotating => Camera::whole_ring(setup.a0_km()),
        ViewFrame::Moon(k) => Camera::moon_view(setup.moon_gap_km(k)),
    }
}

fn floats_to_le_bytes(data: &[f32], out: &mut Vec<u8>) {
    out.clear();
    out.extend(data.iter().flat_map(|x| x.to_le_bytes()));
}

#[wasm_bindgen]
impl App {
    /// Opens WebGPU on `canvas` and starts the GPU self-test. The promise fails
    /// with a message if the browser has no usable WebGPU adapter.
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<App, JsValue> {
        console_error_panic_hook::set_once();
        let (width, height) = (canvas.width().max(1), canvas.height().max(1));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(err)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(err)?;
        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fring-web"),
                ..Default::default()
            })
            .await
            .map_err(err)?;
        let config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| err("the canvas cannot use this adapter"))?;
        surface.configure(&device, &config);

        // The limits that the device was given (not the adapter's), because
        // the stepper checks its buffers against them.
        let gpu = Gpu::from_parts(device.clone(), queue.clone(), info, device.limits());
        let self_test = SelfTest::submit(&gpu);

        // Render pipeline (instanced discs).
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("points"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("points"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("points"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: (FLOATS_PER_INSTANCE * 4) as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Instances: Saturn and the moons (written by the host), then the
        // particles (written by the projection pass). The buffer is both a
        // vertex buffer and a storage buffer.
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: instance_bytes(PARTICLE_COUNT),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Projection pipeline.
        let project_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("project"),
            source: wgpu::ShaderSource::Wgsl(PROJECT_SHADER.into()),
        });
        let project_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("project"),
            layout: None,
            module: &project_module,
            entry_point: Some("project"),
            compilation_options: Default::default(),
            cache: None,
        });
        let project_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("project"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let status_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("status read-back"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let setup = Setup::builtin().map_err(err)?;
        let band = setup.preset_band(0);
        let view = ViewFrame::Inertial;
        let camera = initial_camera(&setup, view);
        Ok(App {
            surface,
            device,
            queue,
            config,
            gpu,
            adapter_name,
            pipeline,
            uniform,
            bind_group,
            instances,
            instances_n: PARTICLE_COUNT,
            project_pipeline,
            project_uniform,
            project_bind: None,
            phase: Phase::SelfTesting(Box::new(self_test)),
            setup,
            full: None,
            kernels: None,
            stepper: None,
            pandora_on: true,
            count: PARTICLE_COUNT,
            band,
            view,
            camera,
            hours_per_s: 6.0,
            paused: false,
            last_ms: None,
            planner: StepPlanner::new(),
            governor: Governor::default(),
            in_flight: Arc::new(AtomicU32::new(0)),
            status: StatusReader {
                buffer: status_buffer,
                state: Arc::new(AtomicU8::new(READ_IDLE)),
                last_ms: 0.0,
            },
            failed_index: None,
            steps_last: 0,
            limited: false,
            note: String::new(),
            scratch: Vec::new(),
            bytes: Vec::new(),
        })
    }

    /// Name of the GPU adapter, for the page footer.
    pub fn adapter_name(&self) -> String {
        self.adapter_name.clone()
    }

    /// `"checking"` (self-test), `"preparing"` (ephemeris and kernels),
    /// `"running"` or `"failed"`.
    pub fn phase(&self) -> String {
        match self.phase {
            Phase::SelfTesting(_) => "checking",
            Phase::Preparing => "preparing",
            Phase::Running => "running",
            Phase::Failed(_) => "failed",
        }
        .to_string()
    }

    /// A message for the readout: the reason of a failure, a particle that
    /// failed, or a note (such as a restart at the end of the ephemeris). Empty
    /// if there is nothing to say.
    pub fn message(&self) -> String {
        if let Phase::Failed(m) = &self.phase {
            return m.clone();
        }
        if let Some(i) = self.failed_index {
            return format!("a particle failed (index {i}); reset to continue");
        }
        self.note.clone()
    }

    /// Sets the canvas size in pixels (after a window resize).
    pub fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        if (w, h) != (self.config.width, self.config.height) {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Speed in simulated hours per real second.
    pub fn set_speed_hours_per_s(&mut self, hours_per_s: f64) {
        self.hours_per_s = hours_per_s;
    }

    /// Pauses or continues the motion.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Pandora on or off. Time goes back to 0.
    pub fn set_pandora(&mut self, on: bool) -> Result<(), JsValue> {
        self.pandora_on = on;
        self.restart()
    }

    /// Initial band by preset: 0 core, 1 circular a0 +- a0 e, 2 circular
    /// between the moons. Time goes back to 0.
    pub fn set_preset(&mut self, index: u32) -> Result<(), JsValue> {
        self.band = self.setup.preset_band(index);
        self.restart()
    }

    /// Smallest and largest semi-major axis (km) of preset `index` as a pair
    /// `[min, max]`; for the core preset, a0 +- 3 sigma.
    pub fn preset_range_km(&self, index: u32) -> Vec<f64> {
        match self.setup.preset_band(index) {
            Band::Circular { min_km, max_km } => vec![min_km, max_km],
            Band::Core => {
                let w = 3.0 * self.setup.ring.dust_strand_width_min_km;
                vec![self.setup.a0_km() - w, self.setup.a0_km() + w]
            }
        }
    }

    /// Circular band from `min_km` to `max_km`. Time goes back to 0. An empty
    /// band is an error and changes nothing.
    pub fn set_band_km(&mut self, min_km: f64, max_km: f64) -> Result<(), JsValue> {
        let band = Band::Circular { min_km, max_km };
        self.setup.initial(band, 1).map_err(err)?;
        self.band = band;
        self.restart()
    }

    /// Starts the run again from t = 0 with the same settings (also clears a
    /// failed particle).
    pub fn restart(&mut self) -> Result<(), JsValue> {
        if self.stepper.is_none() {
            return Ok(()); // The settings are used when the stepper is made.
        }
        self.make_stepper().map_err(err)
    }

    /// 0 = inertial (fixed camera, the start view); 1 = rotating (the camera
    /// turns with the core); 2 = centred on Prometheus; 3 = centred on Pandora.
    /// The camera goes to the start position of that view.
    pub fn set_view(&mut self, index: u32) -> Result<(), JsValue> {
        let view = match index {
            0 => ViewFrame::Inertial,
            1 => ViewFrame::Rotating,
            2 | 3 => {
                let name = if index == 2 { "prometheus" } else { PANDORA };
                let k = self
                    .setup
                    .moon_index(name)
                    .ok_or_else(|| err(format!("no moon {name}")))?;
                ViewFrame::Moon(k)
            }
            _ => return Err(err(format!("no view {index}"))),
        };
        self.view = view;
        self.camera = initial_camera(&self.setup, view);
        Ok(())
    }

    /// Puts the camera back to the start position of the current view.
    pub fn reset_view(&mut self) {
        self.camera = initial_camera(&self.setup, self.view);
    }

    /// A mouse drag of `(dx, dy)` canvas pixels.
    pub fn pan(&mut self, dx_px: f64, dy_px: f64) {
        let size = self.size();
        self.camera.pan_pixels(dx_px, dy_px, size);
    }

    /// Zoom by `factor` (below 1 zooms in) about the canvas pixel `(x_px, y_px)`.
    pub fn zoom(&mut self, x_px: f64, y_px: f64, factor: f64) {
        let size = self.size();
        self.camera.zoom_about(x_px, y_px, factor, size);
    }

    /// Simulated time in days (the time of the particle state).
    pub fn time_days(&self) -> f64 {
        self.stepper.as_ref().map_or(0.0, |s| s.time_s()) / 86_400.0
    }

    /// End of the moon ephemeris in days: the run restarts shortly before it.
    pub fn span_days(&self) -> f64 {
        self.full
            .as_ref()
            .map_or(SPAN_S, |e| e.end_s() - END_MARGIN_S)
            / 86_400.0
    }

    /// Steps run in the last frame.
    pub fn steps_per_frame(&self) -> u32 {
        self.steps_last as u32
    }

    /// True if the simulation runs slower than the slider asks (the GPU or the
    /// frame rate cannot keep up).
    pub fn speed_limited(&self) -> bool {
        self.limited
    }

    /// Number of particles.
    pub fn particle_count(&self) -> u32 {
        self.count
    }

    /// The fastest speed (simulated hours per real second) that the compute
    /// budget allows for the current particle count.
    pub fn max_speed_hours_per_s(&self) -> f64 {
        max_hours_for(self.count)
    }

    /// Sets the number of particles and starts again from t = 0. A count that
    /// does not fit the device is an error and changes nothing. Blocks for the
    /// time that the initial state takes (the page shows a message first).
    pub fn set_particle_count(&mut self, count: u32) -> Result<(), JsValue> {
        let l = &self.gpu.limits;
        count_fits(
            count,
            DEFAULT_WORKGROUP,
            l.max_storage_buffer_binding_size,
            l.max_buffer_size,
            l.max_compute_workgroups_per_dimension,
        )
        .map_err(err)?;
        self.setup.initial(self.band, count).map_err(err)?;
        let old = self.count;
        self.count = count;
        if let Err(e) = self.restart() {
            // Go back to the old count (best effort) and say what went wrong.
            self.count = old;
            let _ = self.restart();
            return Err(e);
        }
        Ok(())
    }

    /// Width of the view in km.
    pub fn view_width_km(&self) -> f64 {
        let size = self.size();
        2.0 * self.camera.half_height_km * size[0] / size[1]
    }

    /// Draws one frame. `now_ms` is the time stamp of `requestAnimationFrame`.
    pub fn frame(&mut self, now_ms: f64) -> Result<(), JsValue> {
        let dt_s = self
            .last_ms
            .map_or(0.0, |last| ((now_ms - last) / 1000.0).clamp(0.0, 0.25));
        self.last_ms = Some(now_ms);
        match &mut self.phase {
            Phase::SelfTesting(test) => {
                match test.poll() {
                    None => {}
                    Some(Ok(())) => self.phase = Phase::Preparing,
                    Some(Err(e)) => self.phase = Phase::Failed(e.to_string()),
                }
                self.draw(0)
            }
            Phase::Preparing => {
                if let Err(e) = self.prepare() {
                    self.phase = Phase::Failed(e);
                }
                self.draw(0)
            }
            Phase::Running => self.run_frame(now_ms, dt_s),
            Phase::Failed(_) => self.draw(0),
        }
    }

    fn size(&self) -> [f64; 2] {
        [self.config.width as f64, self.config.height as f64]
    }

    /// Compiles the kernels, builds the ephemeris and makes the stepper. This
    /// blocks for a second or two; the page has shown "preparing" first.
    fn prepare(&mut self) -> Result<(), String> {
        self.kernels = Some(WhKernels::compile(&self.gpu, DEFAULT_WORKGROUP));
        self.full = Some(
            self.setup
                .build_ephemeris(SPAN_S)
                .map_err(|e| e.to_string())?,
        );
        self.make_stepper().map_err(|e| e.to_string())?;
        self.phase = Phase::Running;
        Ok(())
    }

    /// Makes a new stepper for the current settings (Pandora, band), at t = 0.
    fn make_stepper(&mut self) -> fring_model::Result<()> {
        let (Some(full), Some(kernels)) = (&self.full, &self.kernels) else {
            return Ok(());
        };
        let system = self.setup.system(full, self.pandora_on);
        let init = self.setup.initial(self.band, self.count)?;
        // Free the old buffers first, so that the peak memory is one run.
        self.stepper = None;
        self.project_bind = None;
        if self.instances_n != self.count {
            self.instances = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size: instance_bytes(self.count),
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instances_n = self.count;
        }
        let stepper = GpuStepper::new(
            &self.gpu,
            kernels,
            &system,
            &init,
            STEP_S,
            BLOCK_STEPS,
            Mode::Dp,
            None,
        )?;
        self.project_bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("project"),
            layout: &self.project_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: stepper.state_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.project_uniform.as_entire_binding(),
                },
            ],
        }));
        self.stepper = Some(stepper);
        self.failed_index = None;
        self.planner = StepPlanner::new();
        self.note.clear();
        Ok(())
    }

    /// Reads the failure flag if its map has finished (never blocks).
    fn poll_status(&mut self) {
        let state = self.status.state.load(Ordering::Acquire);
        if state == READ_READY {
            let value = {
                let view = self.status.buffer.get_mapped_range(..);
                view.ok()
                    .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            };
            self.status.buffer.unmap();
            self.status.state.store(READ_IDLE, Ordering::Release);
            if let Some(v) = value
                && v != NO_FAILURE
                && self.failed_index.is_none()
            {
                self.failed_index = Some(v);
            }
        } else if state == READ_FAILED {
            self.status.state.store(READ_IDLE, Ordering::Release);
        }
    }

    /// One frame of the running simulation: steps, projection, draw.
    fn run_frame(&mut self, now_ms: f64, dt_s: f64) -> Result<(), JsValue> {
        self.poll_status();
        self.governor.observe(dt_s);

        // 1. How many steps (C: the moons and the particles share one time).
        let (mut restart, mut steps) = (false, 0u64);
        let in_flight = self.in_flight.load(Ordering::Acquire);
        if let (Some(stepper), Some(full)) = (&self.stepper, &self.full) {
            let (planned, limited) = self.planner.plan(
                dt_s,
                self.hours_per_s.min(max_hours_for(self.count)),
                self.governor.cap(),
            );
            let running = !self.paused && self.failed_index.is_none();
            let busy = in_flight >= MAX_IN_FLIGHT;
            steps = if running && !busy { planned } else { 0 };
            self.limited = running && (limited || busy);
            if running {
                let left_s = full.end_s() - END_MARGIN_S - stepper.time_s();
                if left_s <= STEP_S {
                    restart = true;
                } else {
                    steps = steps.min((left_s / STEP_S) as u64);
                }
            }
        }
        if restart {
            self.make_stepper().map_err(err)?;
            self.note = "Reached the end of the 10-year moon ephemeris: started again from t = 0."
                .to_string();
            steps = 0;
        }
        self.steps_last = steps;

        // 2. Steps and projection, in one submission that always happens.
        let size = self.size();
        let px_per_km = self.camera.px_per_km(size[1]);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute"),
            });
        let mut head_n = 0usize;
        let mut want_map = false;
        if let (Some(stepper), Some(full), Some(bind)) =
            (&mut self.stepper, &self.full, &self.project_bind)
        {
            if steps > 0
                && let Err(e) = stepper.encode_steps(&mut encoder, steps)
            {
                self.phase = Phase::Failed(e.to_string());
                return Ok(());
            }
            let t = stepper.time_s();
            let moons = self.setup.moon_positions(full, t).map_err(err)?;
            let view = view_transform(self.view, self.setup.n0_rad_s(), t, &moons);
            head_n = head_instances(
                &self.setup,
                &view,
                self.camera.centre_km,
                px_per_km,
                &moons,
                self.pandora_on,
                &mut self.scratch,
            );
            floats_to_le_bytes(&self.scratch, &mut self.bytes);
            self.queue.write_buffer(&self.instances, 0, &self.bytes);
            let u = ProjectionUniform::new(
                &view,
                self.camera.centre_km,
                stepper.len() as u32,
                stepper.capacity() as u32,
                head_n as u32,
            );
            self.queue
                .write_buffer(&self.project_uniform, 0, &u.to_bytes());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.project_pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups((stepper.len() as u32).div_ceil(64), 1, 1);
            }
            // B: about once a second, copy the failure flag for a later frame.
            if self.status.state.load(Ordering::Acquire) == READ_IDLE
                && now_ms - self.status.last_ms >= STATUS_PERIOD_MS
            {
                encoder.copy_buffer_to_buffer(
                    stepper.status_buffer(),
                    0,
                    &self.status.buffer,
                    0,
                    4,
                );
                self.status.state.store(READ_PENDING, Ordering::Release);
                self.status.last_ms = now_ms;
                want_map = true;
            }
        }
        self.queue.submit([encoder.finish()]);
        if want_map {
            let state = Arc::clone(&self.status.state);
            self.status
                .buffer
                .map_async(wgpu::MapMode::Read, .., move |r| {
                    state.store(
                        if r.is_ok() { READ_READY } else { READ_FAILED },
                        Ordering::Release,
                    );
                });
        }
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        let counter = Arc::clone(&self.in_flight);
        self.queue.on_submitted_work_done(move || {
            counter.fetch_sub(1, Ordering::AcqRel);
        });

        // 3. Draw.
        let n = self.stepper.as_ref().map_or(0, GpuStepper::len);
        self.draw((head_n + n) as u32)
    }

    /// Renders `count` instances (0 clears the canvas only).
    fn draw(&mut self, count: u32) -> Result<(), JsValue> {
        let size = self.size();
        let px_per_km = self.camera.px_per_km(size[1]);
        let frame_uniform = [
            (2.0 * px_per_km / size[0]) as f32,
            (2.0 * px_per_km / size[1]) as f32,
            size[0] as f32,
            size[1] as f32,
        ];
        floats_to_le_bytes(&frame_uniform, &mut self.bytes);
        self.queue.write_buffer(&self.uniform, 0, &self.bytes);

        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err(err("the canvas surface is lost; reload the page"));
            }
            // Timeout, Occluded and others: skip this frame.
            _ => return Ok(()),
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.025,
                            b: 0.04,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if count > 0 {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.instances.slice(..));
                pass.draw(0..6, 0..count);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(texture);
        Ok(())
    }
}
