"""Plot processed normal/anomaly accelerometer samples."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_RATE_HZ = 416.0
PROCESSED_DIR = Path("data/accelerometer/processed")
PLOT_DIR = PROCESSED_DIR / "plots"


def load_samples(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}_samples.csv"
    data = pd.read_csv(path)
    data["time_seconds"] = data["sample_index"] / SAMPLE_RATE_HZ
    data["magnitude_g"] = np.sqrt(data["x_g"] ** 2 + data["y_g"] ** 2 + data["z_g"] ** 2)
    return data


def plot_axes(data: pd.DataFrame, name: str) -> Path:
    output_path = PLOT_DIR / f"{name}_axes.png"

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{name.capitalize()} accelerometer axes")

    for axis_plot, column, color in zip(axes, ["x_g", "y_g", "z_g"], ["tab:red", "tab:green", "tab:blue"]):
        axis_plot.plot(data["time_seconds"], data[column], color=color, linewidth=0.8)
        axis_plot.set_ylabel(column)
        axis_plot.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_magnitude(normal: pd.DataFrame, anomaly: pd.DataFrame) -> Path:
    output_path = PLOT_DIR / "normal_vs_anomaly_magnitude.png"

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(normal["time_seconds"], normal["magnitude_g"], label="normal", linewidth=0.8)
    axis.plot(anomaly["time_seconds"], anomaly["magnitude_g"], label="anomaly", linewidth=0.8)
    axis.set_title("Acceleration magnitude comparison")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Magnitude (g)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    normal = load_samples("normal")
    anomaly = load_samples("anomaly")

    outputs = [
        plot_axes(normal, "normal"),
        plot_axes(anomaly, "anomaly"),
        plot_magnitude(normal, anomaly),
    ]

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
