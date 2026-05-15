"""Small ANN skeleton implemented with NumPy.

We will complete this step by step while learning:
1. forward pass
2. loss
3. gradients
4. parameter updates
"""

import numpy as np


class TinyANN:
    """One-hidden-layer neural network for binary classification."""

    def __init__(self, input_size: int, hidden_size: int = 8, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)

        self.w1 = rng.normal(0.0, 0.1, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.normal(0.0, 0.1, size=(hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run a forward pass and return probabilities."""
        hidden_linear = x @ self.w1 + self.b1
        hidden = relu(hidden_linear)
        output_linear = hidden @ self.w2 + self.b2
        return sigmoid(output_linear)

    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """Run one full-batch backpropagation and gradient descent step."""
        sample_count = x.shape[0]

        hidden_linear = x @ self.w1 + self.b1
        hidden = relu(hidden_linear)
        output_linear = hidden @ self.w2 + self.b2
        prediction = sigmoid(output_linear)
        loss = binary_cross_entropy(y, prediction)

        # Backpropagation starts here.
        d_output_linear = (prediction - y) / sample_count
        d_w2 = hidden.T @ d_output_linear
        d_b2 = d_output_linear.sum(axis=0, keepdims=True)

        d_hidden = d_output_linear @ self.w2.T
        d_hidden_linear = d_hidden * relu_derivative(hidden_linear)
        d_w1 = x.T @ d_hidden_linear
        d_b1 = d_hidden_linear.sum(axis=0, keepdims=True)

        # Gradient descent update happens here.
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2
        self.w1 -= learning_rate * d_w1
        self.b1 -= learning_rate * d_b1

        return loss


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary classification loss for labels 0/1 and sigmoid predictions."""
    epsilon = 1e-8
    clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
    return float(loss.mean())
