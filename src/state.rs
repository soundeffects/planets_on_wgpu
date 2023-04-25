use crate::{backend::Backend, camera::CameraController, pipeline::Pipeline};
use winit::{dpi::PhysicalSize, event::WindowEvent, window::Window};

pub struct State {
    pipeline: Pipeline,
    controller: CameraController,
}

impl State {
    pub async fn new(window: &Window) -> Self {
        // Our backend class handles interfacing with the GPU
        // Our pipeline class handles rendering models and camera transforms
        let pipeline = Pipeline::new(Backend::new(window).await);

        let controller = CameraController::new(3.5, 1.5, 0.05, 0.0005, 0.15, 0.001, 0.2);

        Self {
            pipeline,
            controller,
        }
    }

    // Accessors

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.pipeline.resize(new_size.width, new_size.height);
    }

    pub fn reconfigure_surface(&mut self) {
        self.pipeline.reconfigure();
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        self.controller.process_events(event)
    }

    pub fn update(&mut self) {
        self.pipeline.update(&mut self.controller);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.pipeline.render()
    }
}
