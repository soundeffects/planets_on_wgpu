use cgmath::{Deg, Matrix4};
use hexasphere::shapes::IcoSphere;
use std::mem::size_of;

use wgpu::{util::BufferInitDescriptor, *};

use crate::backend::Backend;

pub struct Planet {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    index_count: u32,
    rotation: u32,
    noise_offset: [f32; 3],
}

impl Planet {
    pub fn new(backend: &Backend) -> Self {
        let sphere = IcoSphere::new(20, |_| ());
        let vertices = sphere
            .raw_points()
            .iter()
            .map(|point| Vertex {
                position: [point[0], point[1], point[2]],
            })
            .collect::<Vec<Vertex>>();

        let vertex_buffer = backend.create_buffer(&BufferInitDescriptor {
            label: Some("Planet Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        });

        let index_buffer = backend.create_buffer(&BufferInitDescriptor {
            label: Some("Planet Index Buffer"),
            contents: bytemuck::cast_slice(&sphere.get_all_indices()),
            usage: BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: sphere.get_all_indices().len() as u32,
            rotation: 0,
            noise_offset: [fastrand::f32(), fastrand::f32(), fastrand::f32()],
        }
    }

    pub fn increment_rotation(&mut self) {
        self.rotation += 1;
    }

    pub fn reroll(&mut self) {
        self.noise_offset = [fastrand::f32(), fastrand::f32(), fastrand::f32()]
    }

    // Accessors

    pub fn rotation(&self) -> [[f32; 4]; 4] {
        Matrix4::from_angle_y(Deg(self.rotation as f32 / 4.)).into()
    }

    pub fn vertex_buffer(&self) -> &Buffer {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &Buffer {
        &self.index_buffer
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    pub fn noise_offset(&self) -> [f32; 3] {
        self.noise_offset
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
}

impl Vertex {
    pub const ATTRIBUTES: [VertexAttribute; 1] = vertex_attr_array![0 => Float32x3];

    pub fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}
