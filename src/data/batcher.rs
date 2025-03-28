// --- Neural Network Concept: Batching ---
// Processing data one sample at a time can be inefficient.
// Batching groups multiple samples together into a 'batch'.
// This allows for parallel processing (especially on GPUs) and more stable gradient estimates during training.

use burn::{
    data::dataloader::batcher::Batcher,
    prelude::{Backend, Int, Tensor, TensorData},
};

use super::{
    dataset::IrisItem,
    normalizer::{Normalizer, NormalizerConfig},
};

// Define a custom batcher for the Iris dataset
#[derive(Clone)]
pub struct IrisBatcher<B: Backend> {
    device: B::Device,
    normalizer: Normalizer<B>,
}

impl<B: Backend> IrisBatcher<B> {
    pub fn new(device: B::Device, normalizer: NormalizerConfig) -> Self {
        Self {
            device: device.clone(),
            normalizer: normalizer.init(&device),
        }
    }
}

// Implement the Batcher trait
impl<B: Backend> Batcher<IrisItem, IrisBatch<B>> for IrisBatcher<B> {
    fn batch(&self, items: Vec<IrisItem>) -> IrisBatch<B> {
        let features: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|item| TensorData::from(item.features_as_array()))
            .map(|data| Tensor::<B, 1>::from_data(data.convert::<f32>(), &self.device)) // Create 1D Tensor
            .map(|tensor| tensor.unsqueeze()) // Reshape 1D [4] to 2D [1, 4]
            .collect();

        // Normalize features
        let features = self
            .normalizer
            .normalize(Tensor::cat(features, 0))
            .to_device(&self.device);

        let targets: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| {
                Tensor::from_data(
                    TensorData::from([item.target_to_int()]).convert::<i32>(),
                    &self.device,
                )
            })
            .collect();

        let targets = Tensor::cat(targets, 0).to_device(&self.device); // Shape: [batch_size]

        IrisBatch { features, targets }
    }
}

// Define the structure for a batch of Iris data
#[derive(Clone, Debug)]
pub struct IrisBatch<B: Backend> {
    pub features: Tensor<B, 2>,     // Shape: [batch_size, 4]
    pub targets: Tensor<B, 1, Int>, // Shape: [batch_size]
}
