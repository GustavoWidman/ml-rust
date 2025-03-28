pub mod batcher;
pub mod dataset;
pub mod normalizer;

pub const DATASET_PATH: &str = "./data/iris.csv";

// Pre-computed statistics for the iris dataset features
pub const NUM_FEATURES: usize = 4;
// pub const FEATURES_MIN: [f32; NUM_FEATURES] = [4.3, 2.0, 1.0, 0.1];
// pub const FEATURES_MAX: [f32; NUM_FEATURES] = [7.9, 4.4, 6.9, 2.5];
