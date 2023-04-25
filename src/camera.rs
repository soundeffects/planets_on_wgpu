use cgmath::*;
use winit::{
    dpi::PhysicalPosition,
    event::{
        ElementState, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent,
    },
};

pub struct Camera {
    position: Point3<f32>,
    target: Point3<f32>,
    up: Vector3<f32>,
    aspect: f32,
    fov: f32,
    near: f32,
    far: f32,
}

impl Camera {
    pub fn new(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        aspect: f32,
        fov: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            position,
            target,
            up,
            aspect,
            fov,
            near,
            far,
        }
    }

    pub fn position(&self) -> [f32; 4] {
        self.position.to_homogeneous().into()
    }

    pub fn projection(&self) -> [[f32; 4]; 4] {
        let view = Matrix4::look_at_rh(self.position, self.target, self.up);
        let proj = perspective(Deg(self.fov), self.aspect, self.near, self.far);

        return (OPENGL_TO_WGPU_MATRIX * proj * view).into();
    }

    pub fn update_position(&mut self, position: Point3<f32>) {
        self.position = position;
    }

    pub fn update_target(&mut self, target: Point3<f32>) {
        self.target = target;
    }

    pub fn update_up_vector(&mut self, up: Vector3<f32>) {
        self.up = up;
    }

    pub fn update_aspect_ratio(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }
}

pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
);

pub struct CameraController {
    surface_key_sensitivity: f64,
    surface_mouse_sensitivity: f64,
    orbit_key_sensitivity: f64,
    orbit_mouse_sensitivity: f64,
    max_speed: f64,
    mouse_track: bool,
    motion: Vector2<f64>,
    previous_position: Option<PhysicalPosition<f64>>,
    orbit_distance: f64,
    surface_distance: f64,
    surface_mode: bool,
    surface_interpolation: f64,
}

impl CameraController {
    pub fn new(
        orbit_distance: f64,
        surface_distance: f64,
        surface_key_sensitivity: f64,
        surface_mouse_sensitivity: f64,
        orbit_key_sensitivity: f64,
        orbit_mouse_sensitivity: f64,
        max_speed: f64,
    ) -> Self {
        Self {
            surface_key_sensitivity,
            surface_mouse_sensitivity,
            orbit_key_sensitivity,
            orbit_mouse_sensitivity,
            max_speed,
            mouse_track: false,
            motion: Vector2::new(0., 0.),
            previous_position: None,
            orbit_distance,
            surface_distance,
            surface_mode: false,
            surface_interpolation: 0.,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        let mut acceleration = Vector2::<f64>::zero();

        match event {
            WindowEvent::MouseInput { state, .. } => {
                if state == &ElementState::Pressed {
                    self.mouse_track = true;
                } else {
                    self.previous_position = None;
                    self.mouse_track = false;
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.mouse_track = false;
                self.previous_position = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if !self.mouse_track {
                    return false;
                }
                if let Some(previous_position) = self.previous_position {
                    acceleration = Vector2::new(
                        position.x - previous_position.x,
                        previous_position.y - position.y,
                    );

                    if self.surface_mode {
                        acceleration *= self.surface_mouse_sensitivity;
                    } else {
                        acceleration *= self.orbit_mouse_sensitivity;
                    }
                }
                self.previous_position = Some(position.clone());
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, y),
                ..
            } => {
                if y > &0. {
                    self.surface_mode = false;
                } else if y < &0. {
                    self.surface_mode = true;
                } else {
                    return false;
                }
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                if state != &ElementState::Pressed {
                    return false;
                }
                match keycode {
                    VirtualKeyCode::Up => {
                        acceleration -= Vector2::unit_y();
                    }
                    VirtualKeyCode::Left => {
                        acceleration += Vector2::unit_x();
                    }
                    VirtualKeyCode::Down => {
                        acceleration += Vector2::unit_y();
                    }
                    VirtualKeyCode::Right => {
                        acceleration -= Vector2::unit_x();
                    }
                    VirtualKeyCode::Space => {
                        self.surface_mode = !self.surface_mode;
                    }
                    _ => return false,
                }
                if self.surface_mode {
                    acceleration *= self.surface_key_sensitivity;
                } else {
                    acceleration *= self.orbit_key_sensitivity;
                }
            }
            _ => return false,
        }

        if acceleration.magnitude() >= self.max_speed {
            self.motion += acceleration.normalize_to(self.max_speed);
        } else {
            self.motion += acceleration;
        }
        true
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        // Get context variables
        let origin = Self::point_from_vec(Vector3::zero());
        let planet_normal = (origin - camera.position).normalize();
        let tangent = planet_normal.cross(Vector3::unit_y());
        let bitangent = tangent.cross(planet_normal);

        // Find orbit position, target, and up
        let orbit_position = origin.to_vec()
            - (planet_normal + tangent * self.motion.x as f32 + bitangent * self.motion.y as f32)
                .normalize_to(self.orbit_distance as f32);
        let orbit_target = origin.to_vec();
        let orbit_up = Vector3::unit_y();

        // Find surface position, target, and up
        let horizontal = Vector3::new(planet_normal.x, 0., planet_normal.z).normalize();
        let horizontal_tangent = horizontal.cross(Vector3::unit_y());
        let surface_position = origin.to_vec()
            - (horizontal + horizontal_tangent * self.motion.y as f32)
                .normalize_to(self.surface_distance as f32);
        let planet_radius = 1.0;
        let surface_target = (horizontal + horizontal_tangent).normalize_to(planet_radius);
        let surface_up = -horizontal;

        // Interpolate between orbit and surface
        let final_position =
            Self::lerp(orbit_position, surface_position, self.surface_interpolation);
        let final_target = Self::lerp(orbit_target, surface_target, self.surface_interpolation);
        let final_up = Self::lerp(orbit_up, surface_up, self.surface_interpolation);

        // Update camera
        camera.update_position(Self::point_from_vec(final_position));
        camera.update_target(Self::point_from_vec(final_target));
        camera.update_up_vector(final_up);

        // Ease the transition between orbit and surface
        if self.surface_mode {
            if self.surface_interpolation >= 0.99 {
                self.surface_interpolation = 1.;
            } else {
                let difference = 1. - self.surface_interpolation;
                self.surface_interpolation = 1. - difference * 0.9;
            }
        } else {
            if self.surface_interpolation <= 0.01 {
                self.surface_interpolation = 0.;
            } else {
                self.surface_interpolation *= 0.9;
            }
        }

        // Motion inertia; slowly trends towards zero
        if self.motion.magnitude() <= 0.01 {
            self.motion = Vector2::zero();
        } else {
            self.motion *= 0.9;
        }
    }

    // Helpers

    fn lerp(a: Vector3<f32>, b: Vector3<f32>, transition: f64) -> Vector3<f32> {
        a * (1.0 - transition) as f32 + b * transition as f32
    }

    fn point_from_vec(a: Vector3<f32>) -> Point3<f32> {
        Point3 {
            x: a.x,
            y: a.y,
            z: a.z,
        }
    }
}
