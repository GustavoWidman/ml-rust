<div align="center">

# 🦀 ML-Rust: Simple Neural Network in Rust

[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![Burn](https://img.shields.io/badge/burn-0.16.0-red.svg)](https://github.com/tracel-ai/burn)
[![wgpu](https://img.shields.io/badge/backend-wgpu-blue)](https://wgpu.rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A machine learning application demonstrating neural networks implementation in Rust using the Burn framework. This project showcases an Iris flower classifier with a simple multilayer perceptron.*

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Neural Network Concepts](#-neural-network-concepts)
- [Model Architecture](#model-architecture)
- [Dataset](#-dataset)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Pure Rust Implementation** - Neural networks built entirely in Rust
- **GPU Acceleration** - Uses WGPU backend for hardware acceleration
- **Burn Framework** - Leverages Burn's elegant ML framework
- **Well-documented Code** - Code explains neural network concepts throughout
- **Easy to Understand** - Perfect for ML beginners
- **Modular Design** - Separation of model, data, and training logic
- **Iris Classifier** - Classic ML problem for demonstration

---

## 📋 Requirements

- Rust 1.85 or higher
- Cargo (Rust's package manager)
- [Optional] GPU with Vulkan, Metal, or DirectX support for acceleration

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/GustavoWidman/ml-rust.git
cd ml-rust

# Download Iris dataset from https://archive.ics.uci.edu/dataset/53/iris
curl -o iris.zip https://archive.ics.uci.edu/static/public/53/iris.zip
# Extract just the data
unzip iris.zip iris.data
mkdir -p data
# Add CSV headers
echo "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),variety" > data/iris.csv
# Add CSV data
cat iris.data >> data/iris.csv
# Get rid of temporary files
rm iris.data iris.zip

# Build the project
cargo build --release
```

---

## 🚀 Usage

Run the neural network training and evaluation:

```bash
cargo run --release
```

The model will be trained on the Iris dataset and results will be displayed. The trained model (as well as logs and checkpoints of the training and validation epochs) will be saved to the `./model` directory.

### Configuration

The model can be configured through the `config.json` file:

```json
{
  "model": {
    "input_size": 4,
    "hidden_size": 16,
    "output_size": 3
  },
  "optimizer": {
    "weight_decay": null,
    "grad_clipping": null,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 0.00001
  },
  "normalizer": {
    "means": [
      5.872501,
      3.0466664,
      3.8725002,
      1.2575002
    ],
    "std_devs": [
      0.77952564,
      0.43778494,
      1.7053328,
      0.7548358
    ]
  },
  "num_epochs": 2000,
  "batch_size": 3,
  "num_workers": 10,
  "seed": 42,
  "learning_rate": 0.001
}
```

---

## 📁 Project Structure

```bash
ml-rust/
├── src/
│   ├── data/             # Data handling modules
│   │   ├── batcher.rs    # Batching for efficient processing
│   │   ├── dataset.rs    # Iris dataset implementation
│   │   ├── mod.rs        # Module exports
│   │   └── normalizer.rs # Feature normalization
│   ├── model/            # Neural network model
│   │   ├── config.rs     # Model configuration
│   │   ├── model.rs      # Model architecture
│   │   ├── mod.rs        # Module exports
│   │   ├── step.rs       # Training and validation steps
│   │   ├── test.rs       # Testing utilities
│   │   └── train.rs      # Training loop
│   ├── utils/            # Utility functions
│   │   ├── log.rs        # Logging utilities
│   │   ├── misc.rs       # Miscellaneous helpers
│   │   └── mod.rs        # Module exports
│   └── main.rs           # Application entry point
├── data/                 # Dataset directory (created at runtime)
├── model/                # Saved model directory (created at runtime)
├── Cargo.toml            # Dependencies and project metadata
├── LICENSE.txt           # Project license (MIT)
└── README.md             # This file
```

---

## 🧠 Neural Network Concepts

This project demonstrates several key neural network concepts:

### Model Architecture

The model is a simple Multi-Layer Perceptron (MLP) with:

- Input layer (4 features from Iris dataset)
- Hidden layer with ReLU activation
- Output layer (3 classes)

```rust
pub struct Model<B: Backend> {
    pub linear1: Linear<B>,
    pub activation1: Relu,
    pub output: Linear<B>,
}
```

### Forward Pass

Data flows through the network from input to output:

```rust
pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
    let x = self.linear1.forward(input); // Apply first linear layer
    let x = self.activation1.forward(x); // Apply ReLU activation
    self.output.forward(x) // Apply output layer (output logits)
}
```

### Training Loop

1. **Forward Pass**: Feed a batch of data through the network
2. **Calculate Loss**: Compare predictions with actual labels
3. **Backward Pass**: Calculate gradients with automatic differentiation
4. **Optimizer Step**: Adjust weights to reduce loss
5. **Repeat**: Run through multiple epochs

### Normalization

Features are standardized using Z-score normalization:

```rust
normalized_value = (value - mean) / standard_deviation
```

---

## 📊 Dataset

The project uses the famous Iris flower dataset introduced by Ronald Fisher in 1936. It contains measurements for 150 iris flowers from three different species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Each sample has four features:

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

The dataset is split 80/20 for training and testing.

---

## 👥 Contributing

Even though this is a personal learning project, contributions are welcome! I'm always open to suggestions, improvement, and new things to learn! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <sub>Built with ❤️ using Rust 🦀</sub>
</div>
