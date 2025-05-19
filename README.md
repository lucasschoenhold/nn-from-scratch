# Neural Network from Scratch

A Python implementation of a neural network (Multi-Layer Perceptron) built entirely from scratch for learning and understanding the fundamentals of deep learning.

## Overview

This project provides a minimalist implementation of neural networks with automatic differentiation, focusing on clarity and understanding rather than performance. The code recreates basic functionality similar to PyTorch's autograd engine to demonstrate how neural networks and backpropagation work at a fundamental level.

Key features:
- Custom autograd engine (`Engine.py`)
- Neural network components (`nn.py`)
- Optimization algorithms (`optimizer.py`)
- Visualization of computational graphs
- Classification example

## Project Structure

```
nn-from-scratch/
├── src/
│   ├── Engine.py     # Custom autograd engine with Value class
│   ├── nn.py         # Neural network implementation (Neuron, Layer, MLP)
│   └── optimizer.py  # Optimization algorithms (SGD)
├── notebooks/        # Jupyter notebooks explaining concepts
│   ├── 01-01-derivatives.ipynb
│   ├── 01-02-backprob-manually.ipynb
│   ├── 01-03-micrograd.ipynb
│   ├── 01-04-micrograd-more-functions.ipynb
│   └── 01-05-micrograd-mlp.ipynb
├── img/              # Images used in notebooks
├── classification.py # Example of binary classification
├── pyproject.toml    # Project dependencies
└── README.md         # This file
```

## Installation

This project requires Python 3.12 or higher. To set up the environment:

```bash
# Clone the repository
git clone https://github.com/lucasschoenhold/nn-from-scratch.git
cd nn-from-scratch
```
### Create a virtual environment (optional but recommended)
We're using the `uv` package-manager by Astral to manage the virtual environment. If you don't have it installed, you can download it from [here](https://docs.astral.sh/uv/#installation).
To set up the virtual environment, with the dependencies specified in `pyproject.toml`, run:

```bash
# Install dependencies using uv
uv sync
```

## Usage

### Basic Example

```python
from src.nn import MLP
from src.Engine import Value

# Create a neural network with 3 inputs, two hidden layers (4 neurons each), and 1 output
model = MLP(3, [4, 4, 1])

# Forward pass
x = [2.0, 3.0, -1.0]
output = model(x)
print(output)

# Backward pass
output.backward()
```

### Binary Classification Example

You can run the included classification example:

```bash
python classification.py
```

This will train a model on a synthetic moon-shaped dataset and visualize the decision boundary.

## Key Concepts

### Automatic Differentiation

The `Value` class in `Engine.py` implements a computation graph that tracks operations and enables automatic computation of gradients through backpropagation. Each operation (`+`, `*`, etc.) creates a new node in the graph with a defined backward pass.

### Neural Network Architecture

- **Neuron**: Basic computational unit that performs a weighted sum of inputs followed by a non-linear activation function (tanh)
- **Layer**: Collection of neurons that process the same input
- **MLP (Multi-Layer Perceptron)**: Series of layers that transform input data into outputs

### Optimization

The `optimizer.py` module implements Stochastic Gradient Descent (SGD) for updating model parameters to minimize the loss function.

## Educational Resources

The notebooks folder contains step-by-step explanations of:
- Derivatives and gradients
- Manual backpropagation
- Autograd implementation
- Non-linear functions
- Building a Multi-Layer Perceptron
- Optimization with gradient descent

## Visualization

This project includes tools for visualizing the computational graph using Graphviz, making it easier to understand the flow of operations and gradients through the network.

```python
from src.Engine import Value, draw_graph

# Create some values and perform operations
x = Value(2.0, label="x")
y = Value(3.0, label="y")
z = x * y + x
z.label = "z"

# Compute gradients
z.backward()

# Visualize the computational graph
draw_graph(z)
```

## Dependencies

The project dependencies are specified in `pyproject.toml` and include:
- NumPy
- Matplotlib
- Graphviz (for visualization)
- Scikit-learn (for datasets)
- Jupyter (for notebooks)
- PyTorch (for comparison in some notebooks)

## Acknowledgments

This project is inspired by:
- Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)
- Stanford's CS231n course on Convolutional Neural Networks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

&copy; 2025 Lucas Schönhold