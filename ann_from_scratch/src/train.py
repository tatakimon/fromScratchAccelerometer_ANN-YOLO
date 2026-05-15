"""Prepare window features for ANN training.

At this step we only load X/y arrays and normalize the input features.
Training/backpropagation will be added after we inspect these arrays.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from model_numpy import TinyANN, binary_cross_entropy, relu, sigmoid


PROCESSED_DIR = Path("data/accelerometer/processed")
TRAIN_CSV = PROCESSED_DIR / "train_window_features.csv"
TEST_CSV = PROCESSED_DIR / "test_window_features.csv"
EPOCHS = 500
LEARNING_RATE = 0.1

METADATA_COLUMNS = {
    "window_index",
    "start_sample",
    "end_sample",
    "window_size",
    "step_size",
    "label",
    "label_name",
}


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return the columns used as ANN inputs."""
    return [column for column in data.columns if column not in METADATA_COLUMNS]


def load_xy(path: Path, feature_columns: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one split file and return X features, y labels, and feature names."""
    data = pd.read_csv(path)

    if feature_columns is None:
        feature_columns = get_feature_columns(data)

    x = data[feature_columns].to_numpy(dtype=np.float32)
    y = data["label"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return x, y, feature_columns


def normalize_from_train(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize train/test using only train-set mean and std."""
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)

    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std
    return x_train_norm, x_test_norm, mean, std


def print_first_prediction_calculation(model: TinyANN, x_train_norm: np.ndarray) -> None:
    """Show one manual forward-pass calculation for the first training row."""
    x0 = x_train_norm[0:1]

    hidden_linear = x0 @ model.w1 + model.b1
    hidden = relu(hidden_linear)
    output_linear = hidden @ model.w2 + model.b2
    prediction = sigmoid(output_linear)

    print("manual calculation for first training row:")
    print(f"x0 shape:            {x0.shape}")
    print(f"x0 first 5 values:   {x0[0, :5]}")
    print(f"w1 first neuron dot: {(x0 @ model.w1[:, 0:1]).item():.6f}")
    print(f"b1 first value:      {model.b1[0, 0]:.6f}")
    print(f"hidden_linear first: {hidden_linear[0, 0]:.6f}")
    print(f"hidden first:        {hidden[0, 0]:.6f}")
    print(f"output_linear:       {output_linear[0, 0]:.6f}")
    print(f"sigmoid output:      {prediction[0, 0]:.6f}")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    predicted_labels = (y_pred >= 0.5).astype(np.float32)
    return float((predicted_labels == y_true).mean())


def train_model(
    model: TinyANN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Train with full-batch gradient descent and print progress."""
    print("training:")
    for epoch in range(EPOCHS + 1):
        if epoch > 0:
            model.train_step(x_train, y_train, learning_rate=LEARNING_RATE)

        if epoch % 50 == 0:
            train_pred = model.forward(x_train)
            test_pred = model.forward(x_test)
            train_loss = binary_cross_entropy(y_train, train_pred)
            test_loss = binary_cross_entropy(y_test, test_pred)
            train_acc = accuracy(y_train, train_pred)
            test_acc = accuracy(y_test, test_pred)
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"test_loss={test_loss:.6f} "
                f"train_acc={train_acc:.3f} "
                f"test_acc={test_acc:.3f}"
            )


def print_prediction_examples(model: TinyANN, x: np.ndarray, y: np.ndarray, title: str) -> None:
    predictions = model.forward(x)
    print(title)
    for prediction, target in zip(predictions[:10], y[:10]):
        predicted_label = int(prediction[0] >= 0.5)
        print(f"  pred={prediction[0]:.4f} label={predicted_label} target={int(target[0])}")


def main() -> None:
    x_train, y_train, feature_columns = load_xy(TRAIN_CSV)
    x_test, y_test, _ = load_xy(TEST_CSV, feature_columns)
    x_train_norm, x_test_norm, mean, std = normalize_from_train(x_train, x_test)
    model = TinyANN(input_size=x_train_norm.shape[1], hidden_size=8, seed=42)
    train_predictions = model.forward(x_train_norm)
    test_predictions = model.forward(x_test_norm)
    train_loss = binary_cross_entropy(y_train, train_predictions)
    test_loss = binary_cross_entropy(y_test, test_predictions)

    print(f"feature count: {len(feature_columns)}")
    print(f"X_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {x_test.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print()
    print("first 5 feature names:")
    for feature in feature_columns[:5]:
        print(f"  {feature}")
    print()
    print("first normalized training row:")
    print(x_train_norm[0])
    print()
    print("normalization arrays:")
    print(f"mean shape: {mean.shape}")
    print(f"std shape:  {std.shape}")
    print()
    print("model parameter shapes:")
    print(f"w1: {model.w1.shape}")
    print(f"b1: {model.b1.shape}")
    print(f"w2: {model.w2.shape}")
    print(f"b2: {model.b2.shape}")
    print()
    print("initial prediction shapes:")
    print(f"train predictions: {train_predictions.shape}")
    print(f"test predictions:  {test_predictions.shape}")
    print()
    print("first 10 untrained train predictions:")
    for prediction, target in zip(train_predictions[:10], y_train[:10]):
        print(f"  pred={prediction[0]:.4f} target={int(target[0])}")
    print()
    print_first_prediction_calculation(model, x_train_norm)
    print()
    print("untrained loss:")
    print(f"train loss: {train_loss:.6f}")
    print(f"test loss:  {test_loss:.6f}")
    print()
    train_model(model, x_train_norm, y_train, x_test_norm, y_test)
    print()
    print_prediction_examples(model, x_test_norm, y_test, "first 10 trained test predictions:")


if __name__ == "__main__":
    main()
