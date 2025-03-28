use burn::{
    data::dataset::Dataset,
    prelude::{Backend, Tensor},
};
use serde::{Deserialize, Serialize};

use super::{NUM_FEATURES, dataset::IrisDataset};

/// Z-Score Normalization
#[derive(Clone, Debug)]
pub struct Normalizer<B: Backend> {
    pub means: Tensor<B, 2>,
    pub std_devs: Tensor<B, 2>,
}

impl<B: Backend> Normalizer<B> {
    const EPSILON: f32 = 1e-8; // Prevent division by zero
    // Creates a new normalizer.
    // pub fn new(device: &B::Device, min: &[f32], max: &[f32]) -> Self {
    //     let min = Tensor::<B, 1>::from_floats(min, device).unsqueeze();
    //     let max = Tensor::<B, 1>::from_floats(max, device).unsqueeze();
    //     Self { min, max }
    // }

    // Normalizes the input image according to the housing dataset min/max.
    // pub fn normalize(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
    //     (input - self.min.clone()) / (self.max.clone() - self.min.clone())
    // }

    pub fn normalize(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        (input - self.means.clone()) / (self.std_devs.clone() + Self::EPSILON)
    }
}

#[derive(Deserialize, Clone, Serialize)]
pub struct NormalizerConfig {
    pub means: [f32; 4],
    pub std_devs: [f32; 4],
}

impl NormalizerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Normalizer<B> {
        let means = Tensor::<B, 1>::from_floats(self.means, device).unsqueeze();
        let std_devs = Tensor::<B, 1>::from_floats(self.std_devs, device).unsqueeze();

        Normalizer { means, std_devs }
    }

    pub fn from_dataset(dataset: &IrisDataset) -> Self {
        // --- Calculate Mean and Std Dev from TRAINING data ONLY ---
        let mut means = [0.0f32; NUM_FEATURES];
        let mut std_devs = [0.0f32; NUM_FEATURES];
        let n_train = dataset.len() as f32;

        // Calculate means
        for i in 0..dataset.len() {
            if let Some(item) = dataset.get(i) {
                let features = item.features_as_array();
                for j in 0..NUM_FEATURES {
                    means[j] += features[j];
                }
            }
        }
        for j in 0..NUM_FEATURES {
            means[j] /= n_train;
        }

        // Calculate standard deviations
        for i in 0..dataset.len() {
            if let Some(item) = dataset.get(i) {
                let features = item.features_as_array();
                for j in 0..NUM_FEATURES {
                    std_devs[j] += (features[j] - means[j]).powi(2);
                }
            }
        }
        for j in 0..NUM_FEATURES {
            // Use n_train (population std dev) or n_train - 1 (sample std dev)
            // For feature scaling, population std dev is common.
            std_devs[j] = (std_devs[j] / n_train).sqrt();
        }

        Self { means, std_devs }
    }
}
