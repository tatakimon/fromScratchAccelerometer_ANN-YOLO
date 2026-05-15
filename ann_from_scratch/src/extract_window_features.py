"""Create window-level features from processed accelerometer samples."""

from pathlib import Path

import numpy as np
import pandas as pd

from features import window_feature_dict


WINDOW_SIZE = 512
STEP_SIZE = 256
PROCESSED_DIR = Path("data/accelerometer/processed")


def load_processed_samples(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}_samples.csv")


def extract_features_for_class(name: str, label: int) -> pd.DataFrame:
    samples = load_processed_samples(name)
    rows = []

    for window_index, start in enumerate(range(0, len(samples) - WINDOW_SIZE + 1, STEP_SIZE)):
        end = start + WINDOW_SIZE
        window = samples.iloc[start:end]
        feature_row = {
            "window_index": window_index,
            "start_sample": start,
            "end_sample": end - 1,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "label": label,
            "label_name": name,
        }
        feature_row.update(window_feature_dict(window))
        rows.append(feature_row)

    return pd.DataFrame(rows)


def main() -> None:
    normal = extract_features_for_class("normal", label=0)
    anomaly = extract_features_for_class("anomaly", label=1)
    features = pd.concat([normal, anomaly], ignore_index=True)

    output_csv = PROCESSED_DIR / "window_features.csv"
    output_npy = PROCESSED_DIR / "window_features.npy"

    features.to_csv(output_csv, index=False)

    feature_columns = [
        column
        for column in features.columns
        if column not in {"window_index", "start_sample", "end_sample", "window_size", "step_size", "label_name"}
    ]
    np.save(output_npy, features[feature_columns].to_numpy(dtype=np.float32))

    print(f"normal windows:  {len(normal)}")
    print(f"anomaly windows: {len(anomaly)}")
    print(f"feature columns: {len(feature_columns) - 1}")
    print(f"csv:             {output_csv}")
    print(f"npy:             {output_npy}")


if __name__ == "__main__":
    main()
