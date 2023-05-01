#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod backend;
mod camera;
mod pipeline;
mod planet;
mod state;

use state::State;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    // Configure logger for web/native architectures
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger for web.");
        } else {
            env_logger::init();
        }
    }

    // Create winit window, with an event loop
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    // On web, get a canvas element from winit to insert into the document
    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set the size manually
        // when on the web
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(1280, 720));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.body();
                let canvas = web_sys::Element::from(window.canvas());
                dst?.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to the document body.");
    }

    // Create our render state
    let mut state = State::new(&window).await;

    // Event loop prompts the render state based on inputs or render completions
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            window_id,
            ref event,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => (),
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.reconfigure_surface(),
                // System is out of memory, so we quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (outdated or timeout) should be resolved on
                // the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // As soon as the previous frame (redraw) has finished, request the
            // next.
            window.request_redraw();
        }
        _ => (),
    });
}
