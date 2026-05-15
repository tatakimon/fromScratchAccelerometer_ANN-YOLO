"""Feature extraction for accelerometer windows."""

import numpy as np
import pandas as pd


AXES = ["x_g", "y_g", "z_g"]


def window_signal(data: pd.DataFrame, window_size: int, step_size: int) -> list[pd.DataFrame]:
    """Split a dataframe into fixed-size overlapping windows."""
    windows = []

    for start in range(0, len(data) - window_size + 1, step_size):
        windows.append(data.iloc[start : start + window_size])

    return windows


def simple_window_features(window: pd.DataFrame, axes: list[str] | None = None) -> np.ndarray:
    """Return basic statistics for one accelerometer window."""
    selected_axes = axes or AXES
    values = window[selected_axes]
    features = []

    for axis in selected_axes:
        series = values[axis].to_numpy(dtype=float)
        features.extend(
            [
                series.mean(),
                series.std(),
                series.min(),
                series.max(),
            ]
        )

    return np.array(features, dtype=float)


def window_feature_dict(window: pd.DataFrame) -> dict[str, float]:
    """Return named features for one accelerometer window."""
    features = {}

    for axis in AXES:
        series = window[axis].to_numpy(dtype=float)
        features[f"{axis}_mean"] = float(series.mean())
        features[f"{axis}_std"] = float(series.std())
        features[f"{axis}_min"] = float(series.min())
        features[f"{axis}_max"] = float(series.max())
        features[f"{axis}_rms"] = float(np.sqrt(np.mean(series**2)))

    x = window["x_g"].to_numpy(dtype=float)
    y = window["y_g"].to_numpy(dtype=float)
    z = window["z_g"].to_numpy(dtype=float)
    magnitude = np.sqrt(x**2 + y**2 + z**2)

    features["magnitude_mean"] = float(magnitude.mean())
    features["magnitude_std"] = float(magnitude.std())
    features["magnitude_min"] = float(magnitude.min())
    features["magnitude_max"] = float(magnitude.max())
    features["magnitude_rms"] = float(np.sqrt(np.mean(magnitude**2)))

    return features
