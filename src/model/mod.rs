use burn::backend::{Autodiff, Wgpu};

// Define backend type using WGPU and Autodiff
// Autodiff enables automatic gradient calculation (backpropagation)
pub type CustomBackend = Wgpu<f32, i32>; // WGPU backend with f32 floats, i32 ints
pub type CustomAutodiffBackend = Autodiff<CustomBackend>;
// Directory to save model artifacts (checkpoints, final model)
pub const MODEL_ARTIFACT_DIR: &str = "./model";

pub mod config;
pub mod model;
pub mod step;
pub mod test;
pub mod train;
