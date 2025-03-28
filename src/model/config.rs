use burn::{
    nn::{LinearConfig, Relu},
    optim::AdamConfig,
    prelude::{Backend, Config},
};

use crate::data::normalizer::NormalizerConfig;

use super::model::Model;

#[derive(Config, Debug)]
pub struct ModelConfig {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_size: 4,
            hidden_size: 16,
            output_size: 3,
        }
    }
}

impl ModelConfig {
    /// Creates a new model.
    /// Args:
    ///     input_size: Number of input features (4 for Iris).
    ///     hidden_size: Number of neurons in the hidden layer.
    ///     output_size: Number of output classes (3 for Iris).
    ///     device: The device to create the model on.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let linear1_config = LinearConfig::new(self.input_size, self.hidden_size);
        let output_config = LinearConfig::new(self.hidden_size, self.output_size);

        Model {
            linear1: linear1_config.init(device),
            activation1: Relu::new(),
            output: output_config.init(device),
        }
    }
}

// --- Neural Network Concept: Training Configuration ---
// We define parameters that control the training process.
#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub normalizer: Option<NormalizerConfig>,
    pub optimizer: AdamConfig,
    #[config(default = 2000)] // Number of training epochs
    pub num_epochs: usize,
    #[config(default = 3)] // Batch size
    pub batch_size: usize,
    #[config(default = 10)] // Number of parallel workers for data loading
    pub num_workers: usize,
    #[config(default = 42)] // Seed for reproducibility
    pub seed: u64,
    #[config(default = 1e-3)] // Learning rate for the optimizer
    pub learning_rate: f64,
}
