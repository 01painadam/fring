//! The web part (wasm32 only): wgpu render pipeline and the functions that the
//! page (`index.html`) calls. The page owns the DOM and the animation loop; this
//! module owns the GPU and the scene.
//!
//! The scene and the camera are in `scene.rs` and `view.rs`, which the host tests cover.

use crate::scene::{FLOATS_PER_INSTANCE, Preset, Scene, floats_to_le_bytes};
use crate::view::Camera;
use wasm_bindgen::prelude::*;

const SHADER: &str = include_str!("../shaders/points.wgsl");

/// Number of particles. W2 raises this to about 10^5.
const PARTICLE_COUNT: u32 = 20_000;

/// Simulated seconds per real second at slider position 0 (one hour per second).
/// The page sends the speed as a number of simulated hours per real second.
const SECONDS_PER_HOUR: f64 = 3600.0;

#[wasm_bindgen]
pub struct App {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instances: wgpu::Buffer,
    scene: Scene,
    camera: Camera,
    t_s: f64,
    hours_per_s: f64,
    paused: bool,
    last_ms: Option<f64>,
    scratch: Vec<f32>,
    bytes: Vec<u8>,
    adapter_name: String,
}

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn preset_from_index(i: u32) -> Preset {
    if i == 1 { Preset::Band } else { Preset::Core }
}

#[wasm_bindgen]
impl App {
    /// Opens WebGPU on `canvas`. The promise fails with a message if the browser
    /// has no usable WebGPU adapter.
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
        let adapter_name = adapter.get_info().name;
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

        let scene = Scene::builtin(Preset::Core, PARTICLE_COUNT).map_err(err)?;
        let n_instances = 1 + scene.moons().len() + scene.len();
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: (n_instances * FLOATS_PER_INSTANCE * 4) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera = Camera::whole_ring(scene.a0_km());
        Ok(App {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform,
            bind_group,
            instances,
            scene,
            camera,
            t_s: 0.0,
            hours_per_s: 6.0,
            paused: false,
            last_ms: None,
            scratch: Vec::new(),
            bytes: Vec::new(),
            adapter_name,
        })
    }

    /// Name of the GPU adapter, for the page footer.
    pub fn adapter_name(&self) -> String {
        self.adapter_name.clone()
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

    /// 0 = narrow core, 1 = band. Time goes back to 0.
    pub fn set_preset(&mut self, index: u32) -> Result<(), JsValue> {
        self.scene = Scene::builtin(preset_from_index(index), PARTICLE_COUNT).map_err(err)?;
        self.t_s = 0.0;
        Ok(())
    }

    /// Shows the whole ring again.
    pub fn reset_view(&mut self) {
        self.camera = Camera::whole_ring(self.scene.a0_km());
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

    /// Simulated time in days.
    pub fn time_days(&self) -> f64 {
        self.t_s / 86_400.0
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
        if !self.paused {
            self.t_s += dt_s * self.hours_per_s * SECONDS_PER_HOUR;
        }
        let size = self.size();
        let px_per_km = self.camera.px_per_km(size[1]);
        self.scene.write_instances(
            self.t_s,
            self.camera.centre_km,
            px_per_km,
            &mut self.scratch,
        );
        floats_to_le_bytes(&self.scratch, &mut self.bytes);
        self.queue.write_buffer(&self.instances, 0, &self.bytes);
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.instances.slice(..));
            pass.draw(0..6, 0..(self.scratch.len() / FLOATS_PER_INSTANCE) as u32);
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(texture);
        Ok(())
    }

    fn size(&self) -> [f64; 2] {
        [self.config.width as f64, self.config.height as f64]
    }
}
