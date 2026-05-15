"""Data loading helpers for accelerometer CSV files."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "x", "y", "z"]
PROCESSED_COLUMNS = ["timestamp", "x_g", "y_g", "z_g", "label"]


def load_accelerometer_csv(path: str | Path) -> pd.DataFrame:
    """Load one accelerometer CSV and validate the expected columns."""
    csv_path = Path(path)
    data = pd.read_csv(csv_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    return data[REQUIRED_COLUMNS].copy()


def load_labeled_folder(folder: str | Path, label: int) -> list[tuple[pd.DataFrame, int]]:
    """Load all CSV files from a folder and attach the same label to each file."""
    folder_path = Path(folder)
    rows = []

    for csv_path in sorted(folder_path.glob("*.csv")):
        rows.append((load_accelerometer_csv(csv_path), label))

    return rows


def load_processed_samples(path: str | Path) -> pd.DataFrame:
    """Load processed sample-level accelerometer data."""
    csv_path = Path(path)
    data = pd.read_csv(csv_path)

    missing = [column for column in PROCESSED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    return data.copy()


def load_processed_dataset(folder: str | Path) -> pd.DataFrame:
    """Load and combine the processed normal/anomaly sample CSV files."""
    folder_path = Path(folder)
    parts = []

    for name in ["normal_samples.csv", "anomaly_samples.csv"]:
        path = folder_path / name
        if path.exists():
            parts.append(load_processed_samples(path))

    if not parts:
        raise FileNotFoundError(f"No processed sample CSV files found in {folder_path}")

    return pd.concat(parts, ignore_index=True)
