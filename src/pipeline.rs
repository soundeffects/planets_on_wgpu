use cgmath::SquareMatrix;
use wgpu::{util::BufferInitDescriptor, *};

use crate::{
    backend::{Backend, DrawCall},
    camera::{Camera, CameraController},
    planet::{Planet, Vertex},
};

pub struct Pipeline {
    backend: Backend,
    camera: Camera,
    uniform_data: UniformData,
    uniform_buffer: Buffer,
    uniform_bind_group: BindGroup,
    render_pipeline: RenderPipeline,
    planet: Planet,
}

impl Pipeline {
    pub fn new(backend: Backend) -> Self {
        // Create our camera object, for tracking the data needed to create the
        // camera space transforms.
        let camera = Camera::new(
            (0., 2., 3.).into(),
            (0., 0., 0.).into(),
            cgmath::Vector3::unit_y(),
            backend.width() as f32 / backend.height() as f32,
            50.,
            0.1,
            100.,
        );

        let planet = Planet::new(&backend);

        // Camera matrix is the camera space transform.
        let mut uniform_data = UniformData::new();
        uniform_data.update_camera(&camera);
        uniform_data.update_planet(&planet);

        // Create a buffer on the GPU to store the camera matrix.
        let uniform_buffer = backend.create_buffer(&BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Create a bind group layout to access this camera matrix from the
        // vertex shader.
        let uniform_bind_group_layout =
            backend.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("Uniform Binding Layout"),
            });

        // There's only a single version of the camera matrix, so we can just
        // create a single camera binding and use it everywhere.
        let uniform_bind_group = backend.create_bind_group(&BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("Uniform Binding"),
        });

        // Importing our shader code
        let shader = backend.create_shader(ShaderModuleDescriptor {
            label: Some("Shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Creating a default render pipeline layout
        let render_pipeline_layout = backend.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            ..Default::default()
        });

        // Define our render pipeline using the shaders and the layout
        let render_pipeline = backend.create_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            // Our format is sRGB, we replace instead of blending, and we write
            // to all values (RGBA values).
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: backend.format(),
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            // No multisample anti-aliasing.
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            backend,
            camera,
            uniform_data,
            uniform_buffer,
            uniform_bind_group,
            render_pipeline,
            planet,
        }
    }

    pub fn update(&mut self, controller: &mut CameraController) {
        // Compute new camera matrix.
        controller.update_camera(&mut self.camera);
        self.uniform_data.update_camera(&self.camera);

        controller.update_planet(&mut self.planet);
        self.planet.increment_rotation();
        self.planet.increment_clouds();
        self.uniform_data.update_planet(&self.planet);

        // Send a write command to the GPU for the new camera matrix.
        self.backend
            .submit_write(&self.uniform_buffer, &[self.uniform_data]);
    }

    pub fn render(&mut self) -> Result<(), SurfaceError> {
        let draw_calls = vec![DrawCall {
            vertex_buffer: self.planet.vertex_buffer().slice(..),
            index_buffer: self.planet.index_buffer().slice(..),
            index_count: self.planet.index_count(),
            instance_count: 1,
            bind_groups: vec![&self.uniform_bind_group],
        }];

        self.backend
            .submit_render(&self.render_pipeline, draw_calls)
    }

    // Accessors

    pub fn resize(&mut self, width: u32, height: u32) {
        self.backend.resize(width, height);
        self.camera.update_aspect_ratio(width, height);
    }

    pub fn reconfigure(&mut self) {
        self.backend.reconfigure();
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformData {
    view_position: [f32; 4],
    view_projection: [[f32; 4]; 4],
    planet_rotation: [[f32; 4]; 4],
    noise_offset: [f32; 4],
    cloud_offset: [f32; 4],
    clouds_disabled: [i32; 4],
}

impl UniformData {
    pub fn new() -> Self {
        Self {
            view_position: [0.; 4],
            view_projection: cgmath::Matrix4::identity().into(),
            planet_rotation: cgmath::Matrix4::identity().into(),
            noise_offset: [0.; 4],
            cloud_offset: [0.; 4],
            clouds_disabled: [0; 4],
        }
    }

    pub fn update_camera(&mut self, camera: &Camera) {
        self.view_position = camera.position();
        self.view_projection = camera.projection();
    }

    pub fn update_planet(&mut self, planet: &Planet) {
        self.planet_rotation = planet.rotation();
        self.cloud_offset = [planet.cloud_offset() as f32; 4];
        let offsets = planet.noise_offset();
        self.noise_offset = [offsets[0], offsets[1], offsets[2], 0.];
        self.clouds_disabled = [planet.clouds_disabled() as i32; 4];
    }
}
