use burn::{
    backend::wgpu::WgpuDevice,
    prelude::{Backend, Config, Module},
    record::CompactRecorder,
    train::{
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, CpuMemory, CpuUse, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};

use crate::{
    data::{
        DATASET_PATH, batcher::IrisBatcher, dataset::IrisDataset, normalizer::NormalizerConfig,
    },
    model::{
        CustomAutodiffBackend, CustomBackend, MODEL_ARTIFACT_DIR, config::TrainingConfig,
        test::run_testing,
    },
    utils::misc::create_artifact_dir,
};

// --- Neural Network Concept: Training Loop ---
// The core process of teaching the network:
// 1.  **Forward Pass:** Feed a batch of data through the network to get predictions.
// 2.  **Calculate Loss:** Compare predictions with actual labels using the loss function.
// 3.  **Backward Pass (Backpropagation):** Calculate gradients (derivatives of the loss with respect to each network parameter/weight). This tells us how much each weight contributed to the error. Burn's `Autodiff` backend handles this automatically.
// 4.  **Optimizer Step:** Adjust the network's weights based on the gradients and the learning rate. The goal is to slightly change weights to reduce the loss on the next iteration. Common optimizers like `Adam` use sophisticated methods to adapt the learning rate.
// 5.  **Repeat:** Iterate through the dataset multiple times (epochs).
pub fn run_training(device: WgpuDevice) -> anyhow::Result<()> {
    log::info!("Training on device {:?}", device);

    create_artifact_dir(format!("{}/training", MODEL_ARTIFACT_DIR).as_str());

    // Load the configuration
    let mut config = TrainingConfig::load(format!("{}/config.json", MODEL_ARTIFACT_DIR))
        .unwrap_or_else(|_| panic!("Failed to load config.json. Create one or use defaults."));
    log::info!("Config:\n\n{}\n", config);
    let optim_config = config.optimizer.init(); // Initialize optimizer config

    // Read stdin for a newline ("Press enter to continue...")
    log::info!("Press enter to continue...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    // Set random seed for reproducibility
    CustomAutodiffBackend::seed(config.seed);

    // Load dataset
    // Note: Burn doesn't have a built-in remote Iris loader like MNIST yet.
    // You need to download iris.csv manually (e.g., from Kaggle or UCI)
    // and place it in the project root or specify the correct path.
    let dataset_full = IrisDataset::new(DATASET_PATH)?;

    // --- Data Splitting ---
    // We split the data into training and testing sets.
    // Training set: Used to train the model.
    // Testing set: Used to evaluate the model's performance on unseen data.
    let ratio = 0.8; // 80% for training, 20% for testing
    let (dataset_train, dataset_test) = dataset_full.random_split(ratio, config.seed);

    // --- Neural Network Concept: Normalizer ---
    // We normalize the data to have zero mean and unit variance.
    // This is important for the training process.
    // Create normalizer configuration, updating the config if it was None
    if config.normalizer.is_none() {
        config.normalizer = Some(NormalizerConfig::from_dataset(&dataset_train));
        config.save(format!("{}/config.json", MODEL_ARTIFACT_DIR))?;
    }
    // Unwrapping is safe because we're sure it's not None
    let normalizer = config.normalizer.as_ref().unwrap();

    // Create batchers
    let batcher_train =
        IrisBatcher::<CustomAutodiffBackend>::new(device.clone(), normalizer.clone());
    let batcher_test = IrisBatcher::<CustomBackend>::new(device.clone(), normalizer.clone()); // No Autodiff needed for testing

    // --- Neural Network Concept: Dataloaders ---
    // Efficiently load and batch data, often in parallel using multiple workers.
    let dataloader_train = burn::data::dataloader::DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train.clone());

    let dataloader_test = burn::data::dataloader::DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed) // Shuffling test set usually not necessary, but doesn't hurt
        .num_workers(config.num_workers)
        .build(dataset_test.clone());

    // --- Model Initialization ---
    let model = config.model.init::<CustomAutodiffBackend>(&device);

    // --- Learner Setup ---
    // The `Learner` orchestrates the training process using the components we defined.
    let learner = LearnerBuilder::new(format!("{}/training", MODEL_ARTIFACT_DIR).as_str())
        // Use Checkpointer for saving model state periodically
        .summary()
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .with_file_checkpointer(CompactRecorder::new()) // Save model state compactly
        .early_stopping(MetricEarlyStoppingStrategy::new::<
            LossMetric<CustomAutodiffBackend>,
        >(
            // Stop if validation loss doesn't improve
            Aggregate::Mean,   // Aggregate metric over batches
            Direction::Lowest, // We want the highest accuracy
            Split::Valid,      // Monitor the validation split
            StoppingCondition::NoImprovementSince { n_epochs: 50 }, // Stop if no improvement for 50 epochs
        ))
        .devices(vec![device.clone()]) // Specify the device(s) to train on
        .num_epochs(config.num_epochs)
        .build(model, optim_config, config.learning_rate);

    // --- Start Training ---
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // --- Saving the Trained Model ---
    // The learner automatically saves checkpoints, but we can explicitly save the final model.
    log::info!("Training complete. Saving final model...");
    model_trained
        .save_file(
            format!("{}/model", MODEL_ARTIFACT_DIR),
            &CompactRecorder::new(),
        )
        .expect("Failed to save trained model");

    log::info!("Model saved to {}/model.mpk.gz", MODEL_ARTIFACT_DIR);

    // train on top of the training and test datasets
    let train_dataset_results = run_testing(device.clone(), Some(dataset_train))?;
    let test_dataset_results = run_testing(device.clone(), Some(dataset_test))?;

    log::info!(
        "Train dataset accuracy: {:.2}% (correct: {}, incorrect: {})",
        train_dataset_results.correct_percentage * 100.0,
        train_dataset_results.correct_count,
        train_dataset_results.incorrect_count
    );

    log::info!(
        "Test dataset accuracy: {:.2}% (correct: {}, incorrect: {})",
        test_dataset_results.correct_percentage * 100.0,
        test_dataset_results.correct_count,
        test_dataset_results.incorrect_count
    );

    Ok(())
}
