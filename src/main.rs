use burn::{
    backend::wgpu::WgpuDevice, // Import Wgpu backend and Autodiff
    optim::AdamConfig,
    prelude::Config, // Optimizer
};
use model::config::{ModelConfig, TrainingConfig};

mod data;
mod model;
mod utils;

fn main() -> anyhow::Result<()> {
    utils::Logger::init(None);

    let device = WgpuDevice::DefaultDevice;

    // Create a default config file if it doesn't exist
    if std::fs::metadata(format!("{}/config.json", model::MODEL_ARTIFACT_DIR)).is_err() {
        log::info!("config.json not found, creating default.");
        let default_config = TrainingConfig::new(ModelConfig::default(), AdamConfig::new());
        default_config
            .save(format!("{}/config.json", model::MODEL_ARTIFACT_DIR))
            .expect("Failed to save default config.json");
    }

    // Run the training process
    model::train::run_training(device.clone())?;

    // --- Inference Example (Optional) ---
    // You could add code here to load the trained model and make predictions
    // on new data points.
    // 1. Load the model state using a Recorder and `load_file`.
    // 2. Create a new data point (Tensor).
    // 3. Pass it through `model.forward(input)`.
    // 4. Interpret the output (e.g., find the index with the highest value using `argmax`).
    log::info!(
        "To run inference, you would load the model from {}/model.mpk.gz and use its forward method.",
        model::MODEL_ARTIFACT_DIR
    );

    let full_testing_results = model::test::run_testing(device, None)?;

    log::info!(
        "Final accuracy: {:.2}% (correct: {}, incorrect: {})",
        full_testing_results.correct_percentage * 100.0,
        full_testing_results.correct_count,
        full_testing_results.incorrect_count
    );

    Ok(())
}
