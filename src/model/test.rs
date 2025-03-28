use burn::{
    backend::wgpu::WgpuDevice,
    prelude::{Config, Module},
    record::{CompactRecorder, Recorder},
};

use crate::{
    data::{DATASET_PATH, batcher::IrisBatcher, dataset::IrisDataset},
    model::{CustomBackend, MODEL_ARTIFACT_DIR, config::TrainingConfig},
};

pub struct TestingResults {
    pub correct_count: usize,
    pub incorrect_count: usize,
    pub total_count: usize,
    pub correct_percentage: f32,
}

pub fn run_testing(
    device: WgpuDevice,
    dataset: Option<IrisDataset>,
) -> anyhow::Result<TestingResults> {
    // Load the configuration
    let config = TrainingConfig::load(format!("{}/config.json", MODEL_ARTIFACT_DIR))
        .unwrap_or_else(|_| panic!("Failed to load config.json. Create one or use defaults."));
    let record = CompactRecorder::new()
        .load(format!("{MODEL_ARTIFACT_DIR}/model").into(), &device)
        .expect("Trained model should exist");
    let normalizer = config.normalizer.as_ref().unwrap(); // Unwrapping is safe because we're sure it's not None (model has been trained, so it exists)

    // Load dataset
    // Note: Burn doesn't have a built-in remote Iris loader like MNIST yet.
    // You need to download iris.csv manually (e.g., from Kaggle or UCI)
    // and place it in the project root or specify the correct path.
    let dataset = dataset.unwrap_or(IrisDataset::new(DATASET_PATH)?);
    let batcher = IrisBatcher::<CustomBackend>::new(device.clone(), normalizer.clone());

    // --- Neural Network Concept: Dataloaders ---
    // Efficiently load and batch data, often in parallel using multiple workers.
    let dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .build(dataset);

    let model = config
        .model
        .init::<CustomBackend>(&device)
        .load_record(record);

    // test accuracy on training set
    let mut results = TestingResults {
        correct_count: 0,
        incorrect_count: 0,
        total_count: 0,
        correct_percentage: 0.0,
    };
    for batch in dataloader.iter() {
        let output = model.forward(batch.features).argmax(1).to_data();
        let output = output
            .as_slice::<i32>()
            .map_err(|_| anyhow::anyhow!("Unable to convert output to i32 slice."))?;
        let targets = batch.targets.to_data();
        let targets = targets
            .as_slice::<i32>()
            .map_err(|_| anyhow::anyhow!("Unable to convert output to i32 slice."))?;

        for (predicted, correct) in output.into_iter().zip(targets.into_iter()) {
            if predicted == correct {
                results.correct_count += 1;
            } else {
                results.incorrect_count += 1;
            }
            results.total_count += 1;
        }
    }
    results.correct_percentage = results.correct_count as f32 / results.total_count as f32;

    Ok(results)
}
