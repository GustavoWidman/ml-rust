use burn::{
    nn::{Linear, Relu, loss::CrossEntropyLossConfig},
    prelude::{Backend, Int, Module, Tensor},
    train::ClassificationOutput,
};

// --- Neural Network Concept: Model Architecture ---
// We define the structure of our network: layers and how they connect.
// We'll use a simple Multi-Layer Perceptron (MLP).
// Input (4 features) -> Linear Layer 1 -> ReLU Activation -> Linear Layer 2 -> ReLU Activation -> Output (3 classes)
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub linear1: Linear<B>,
    pub activation1: Relu,
    pub output: Linear<B>,
}

// --- Neural Network Concept: Layers ---
// *   `Linear` (or Dense): Performs a linear transformation (matrix multiplication + bias addition). Learns linear relationships.
//     `y = Wx + b`, where W is weights, b is bias, x is input, y is output.
// *   `ReLU` (Rectified Linear Unit): An activation function. It introduces non-linearity, allowing the network to learn complex patterns.
//     `f(x) = max(0, x)`. It simply outputs the input if positive, and zero otherwise.

impl<B: Backend> Model<B> {
    // --- Neural Network Concept: Forward Pass ---
    // This defines how data flows through the network from input to output.
    /// Performs the forward pass.
    /// Args:
    ///     input: The input tensor (batch of features). Shape: [batch_size, 4].
    /// Returns:
    ///     Output tensor (logits). Shape: [batch_size, 3].
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input); // Apply first linear layer
        let x = self.activation1.forward(x); // Apply ReLU activation
        self.output.forward(x) // Apply output layer (output logits)
    }

    // Forward pass for classification tasks (used by the trainer)
    pub fn forward_classification(
        &self,
        features: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(features);
        // --- Neural Network Concept: Loss Function ---
        // Measures how wrong the model's predictions are compared to the actual labels.
        // `CrossEntropyLoss` is common for multi-class classification. It compares the predicted probabilities (derived from logits) with the true class label.
        // The goal of training is to minimize this loss.
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}
