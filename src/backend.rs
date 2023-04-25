use bytemuck::NoUninit;
use wgpu::{util::DeviceExt, *};
use winit::window::Window;

pub struct Backend {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    depth_texture: Texture,
    depth_view: TextureView,
    depth_sampler: Sampler,
    frame_count: u32,
}

impl Backend {
    pub async fn new(window: &Window) -> Self {
        // The instance is a handle to create surfaces and adapters for our GPU.
        // Backends include Vulkan, Metal, DX12, and WebGPU.
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        // The surface is the area we draw to. The surface needs to live as long
        // as the window that created it. The window stays valid for the entire
        // application runtime, so the following code should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        // The adapter is a handle to our graphics cards. We will use this to
        // create the device and queue.
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();

        // The device will specify which graphics card we want to use. The queue
        // will take instructions we provide and send them to the GPU.
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    limits: if cfg!(target_arch = "wasm32") {
                        Limits::downlevel_webgl2_defaults()
                    } else {
                        Limits::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        // We will use the capabilities as a set of options from which we can
        // select our surface configuration options.
        let capabilities = surface.get_capabilities(&adapter);

        // We assume an sRGB color for the format configuration.
        let surface_format = capabilities
            .formats
            .iter()
            .copied()
            .filter(|f| f.describe().srgb)
            .next()
            .unwrap_or(capabilities.formats[0]);

        // We create the configuration struct, using the capabilities.
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: capabilities.present_modes[0],
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
        };

        // We send the configuration to the surface.
        surface.configure(&device, &config);

        // Create a depth texture, for rendering occluding surfaces properly
        let (depth_texture, depth_view, depth_sampler) =
            Self::create_depth_texture(&device, config.width, config.height);

        Self {
            surface,
            device,
            queue,
            config,
            depth_texture,
            depth_view,
            depth_sampler,
            frame_count: 0,
        }
    }

    pub fn create_depth_texture(
        device: &Device,
        width: u32,
        height: u32,
    ) -> (Texture, TextureView, Sampler) {
        let size = Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        };

        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

        let depth_sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            compare: Some(CompareFunction::LessEqual),
            lod_min_clamp: 0.,
            lod_max_clamp: 100.,
            ..Default::default()
        });

        (depth_texture, depth_view, depth_sampler)
    }

    pub fn submit_render(
        &mut self,
        render_pipeline: &RenderPipeline,
        draw_calls: Vec<DrawCall>,
    ) -> Result<(), SurfaceError> {
        // Get a new texture view from our surface, which we will draw onto
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        // Create an encoder, which is responsible for converting render pass
        // descriptions into commands which can be executed by the GPU
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        // Create a unique name for this render pass
        self.frame_count += 1;
        let label = format!("Render Pass {}", self.frame_count);

        // Create the render pass using the encoder. A render pass will always
        // clear the screen to a black color, and will always have a depth
        // buffer.
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label.as_str()),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });

        // We add the pipeline and data to draw to the render pass.
        render_pass.set_pipeline(render_pipeline);
        for draw_call in draw_calls {
            render_pass.set_vertex_buffer(0, draw_call.vertex_buffer);
            render_pass.set_index_buffer(draw_call.index_buffer, IndexFormat::Uint32);
            for i in 0..draw_call.bind_groups.len() {
                render_pass.set_bind_group(i as u32, draw_call.bind_groups[i], &[]);
            }
            render_pass.draw_indexed(0..draw_call.index_count, 0, 0..draw_call.instance_count);
        }

        // Before we can use the encoder again, we must drop the render pass,
        // which "unborrows" the encoder.
        drop(render_pass);

        // Encoder stores the render pass. We just need to tell it to send the
        // command.
        self.queue.submit(std::iter::once(encoder.finish()));

        // Display the rendered frame
        output.present();

        Ok(())
    }

    pub fn submit_write<A>(&self, buffer: &wgpu::Buffer, slice: &[A])
    where
        A: NoUninit,
    {
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(slice));
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.reconfigure();
        }
    }

    pub fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.config);
        let (depth_texture, depth_view, depth_sampler) =
            Self::create_depth_texture(&self.device, self.config.width, self.config.height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.depth_sampler = depth_sampler;
    }

    // Accessors

    pub fn width(&self) -> u32 {
        self.config.width
    }

    pub fn height(&self) -> u32 {
        self.config.height
    }

    pub fn format(&self) -> TextureFormat {
        self.config.format
    }

    pub fn create_bind_group(&self, descriptor: &wgpu::BindGroupDescriptor) -> wgpu::BindGroup {
        self.device.create_bind_group(descriptor)
    }

    pub fn create_bind_group_layout(
        &self,
        descriptor: &wgpu::BindGroupLayoutDescriptor,
    ) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(descriptor)
    }

    pub fn create_buffer(&self, descriptor: &wgpu::util::BufferInitDescriptor) -> wgpu::Buffer {
        self.device.create_buffer_init(descriptor)
    }

    pub fn create_shader(&self, descriptor: wgpu::ShaderModuleDescriptor) -> wgpu::ShaderModule {
        self.device.create_shader_module(descriptor)
    }

    pub fn create_pipeline_layout(&self, descriptor: &PipelineLayoutDescriptor) -> PipelineLayout {
        self.device.create_pipeline_layout(descriptor)
    }

    pub fn create_pipeline(&self, descriptor: &RenderPipelineDescriptor) -> RenderPipeline {
        self.device.create_render_pipeline(descriptor)
    }
}

pub struct DrawCall<'a> {
    pub vertex_buffer: BufferSlice<'a>,
    pub index_buffer: BufferSlice<'a>,
    pub index_count: u32,
    pub instance_count: u32,
    pub bind_groups: Vec<&'a BindGroup>,
}
