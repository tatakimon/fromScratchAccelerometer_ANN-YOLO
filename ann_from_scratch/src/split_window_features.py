"""Split window features into train/test sets without overlap leakage.

Important idea:
Do not randomly split overlapping windows. Neighboring windows share samples,
so train/test must be separated by time order first.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/accelerometer/processed")
INPUT_CSV = PROCESSED_DIR / "window_features.csv"
TRAIN_CSV = PROCESSED_DIR / "train_window_features.csv"
TEST_CSV = PROCESSED_DIR / "test_window_features.csv"
TRAIN_NPY = PROCESSED_DIR / "train_window_features.npy"
TEST_NPY = PROCESSED_DIR / "test_window_features.npy"

TEST_FRACTION = 0.2
RANDOM_SEED = 42
METADATA_COLUMNS = {
    "window_index",
    "start_sample",
    "end_sample",
    "window_size",
    "step_size",
    "label_name",
}


def split_one_class(data: pd.DataFrame, label_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one class by time order, keeping the newest block for test.

    Because windows overlap, the last training window before the split can
    share raw samples with the first test window. We remove that boundary
    window from training.
    """
    class_rows = data[data["label_name"] == label_name].sort_values("start_sample")
    test_count = max(1, int(round(len(class_rows) * TEST_FRACTION)))
    split_index = len(class_rows) - test_count

    test = class_rows.iloc[split_index:].copy()
    first_test_sample = int(test["start_sample"].min())
    train = class_rows.iloc[:split_index].copy()
    train = train[train["end_sample"] < first_test_sample].copy()
    return train, test


def feature_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric model columns: all features plus label as the last column."""
    columns = [column for column in data.columns if column not in METADATA_COLUMNS]
    columns.remove("label")
    return columns + ["label"]


def main() -> None:
    data = pd.read_csv(INPUT_CSV)

    train_parts = []
    test_parts = []
    for label_name in ["normal", "anomaly"]:
        train, test = split_one_class(data, label_name)
        train_parts.append(train)
        test_parts.append(test)

    train_data = pd.concat(train_parts, ignore_index=True)
    test_data = pd.concat(test_parts, ignore_index=True)

    rng = np.random.default_rng(RANDOM_SEED)
    train_data = train_data.iloc[rng.permutation(len(train_data))].reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    train_data.to_csv(TRAIN_CSV, index=False)
    test_data.to_csv(TEST_CSV, index=False)

    model_columns = feature_columns(data)
    np.save(TRAIN_NPY, train_data[model_columns].to_numpy(dtype=np.float32))
    np.save(TEST_NPY, test_data[model_columns].to_numpy(dtype=np.float32))

    print(f"train rows: {len(train_data)}")
    print(train_data["label_name"].value_counts().to_string())
    print(f"test rows:  {len(test_data)}")
    print(test_data["label_name"].value_counts().to_string())
    print(f"model input features: {len(model_columns) - 1}")
    print(f"train csv: {TRAIN_CSV}")
    print(f"test csv:  {TEST_CSV}")


if __name__ == "__main__":
    main()
