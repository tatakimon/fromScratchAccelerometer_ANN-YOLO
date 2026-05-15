"""Plot selected window-level features for normal vs anomaly samples."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROCESSED_DIR = Path("data/accelerometer/processed")
PLOT_DIR = PROCESSED_DIR / "plots"
FEATURES_TO_PLOT = [
    "magnitude_std",
    "magnitude_max",
    "x_g_std",
    "y_g_std",
    "z_g_std",
    "z_g_min",
]


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(PROCESSED_DIR / "window_features.csv")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Window feature comparison")

    for axis, feature in zip(axes.ravel(), FEATURES_TO_PLOT):
        normal = data.loc[data["label_name"] == "normal", feature]
        anomaly = data.loc[data["label_name"] == "anomaly", feature]
        axis.boxplot([normal, anomaly], tick_labels=["normal", "anomaly"])
        axis.set_title(feature)
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = PLOT_DIR / "window_feature_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
