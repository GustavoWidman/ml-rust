use burn::{
    prelude::Backend,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::batcher::IrisBatch;

use super::model::Model;

impl<B: AutodiffBackend> TrainStep<IrisBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: IrisBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.features, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<IrisBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: IrisBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.features, batch.targets)
    }
}
