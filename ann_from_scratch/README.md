# ANN From Scratch: Accelerometer Anomaly Detection

Goal: understand a small artificial neural network by building it with NumPy instead of PyTorch.

The current version is a complete learning pipeline:

```text
raw Tera Term logs
-> sample-level processed data
-> 512-sample windows
-> window features
-> no-leak train/test split
-> NumPy ANN training
```

## Sensor Setup

The raw logs were captured with:

```text
Number of axes: 3
Data rate: 416 Hz
Range: 2 g
Samples per axis: 512
```

One 512-sample window is:

```text
512 / 416 = 1.23 seconds
```

## Raw Data Format

Expected files:

```text
data/accelerometer/raw/normal/teraterm.log
data/accelerometer/raw/anomaly/teraterm.log
```

Each raw line has one timestamp followed by many `x y z` triples:

```text
[2026-05-14 20:06:52.009] x1 y1 z1 x2 y2 z2 ...
```

The values are treated as milligravity (`mg`). The processor creates `g` values by dividing by 1000.

## Important Columns

Processed sample CSV files contain:

```text
sample_index
timestamp
line_timestamp
line_number
sample_in_line
x_mg, y_mg, z_mg
x_g, y_g, z_g
label
label_name
```

Column meanings:

- `sample_index`: global sample number after expanding the log.
- `timestamp`: estimated per-sample timestamp using `sample_index / 416`.
- `line_timestamp`: original timestamp printed once at the start of a raw log line.
- `line_number`: raw log line number.
- `sample_in_line`: which `x y z` triple inside that raw line.
- `label`: `0` for normal, `1` for anomaly.

For plotting and ML, `sample_index / 416` is more reliable than the Tera Term line timestamps.

## Pipeline Commands

Run from the repository root.

### 1. Process Raw Logs

```powershell
python ann_from_scratch\src\data_processor.py
```

Outputs:

```text
data/accelerometer/processed/normal_samples.csv
data/accelerometer/processed/anomaly_samples.csv
data/accelerometer/processed/normal_samples.npy
data/accelerometer/processed/anomaly_samples.npy
```

### 2. Plot Sample Data

```powershell
python ann_from_scratch\src\plot_samples.py
```

Outputs:

```text
data/accelerometer/processed/plots/normal_axes.png
data/accelerometer/processed/plots/anomaly_axes.png
data/accelerometer/processed/plots/normal_vs_anomaly_magnitude.png
```

### 3. Extract Window Features

```powershell
python ann_from_scratch\src\extract_window_features.py
```

Current settings:

```text
window_size = 512
step_size = 256
```

This creates 50% overlapping windows. Each window becomes 20 input features.

Outputs:

```text
data/accelerometer/processed/window_features.csv
data/accelerometer/processed/window_features.npy
```

### 4. Plot Window Features

```powershell
python ann_from_scratch\src\plot_window_features.py
```

Output:

```text
data/accelerometer/processed/plots/window_feature_comparison.png
```

### 5. Split Train/Test Without Leakage

```powershell
python ann_from_scratch\src\split_window_features.py
```

Why this matters:

```text
window 0: samples 0-511
window 1: samples 256-767
```

These windows overlap. Randomly splitting overlapping windows can put shared physical samples into both train and test. The split script avoids that by splitting by time order per class and removing boundary overlap.

The exact rule is:

```text
1. Keep each class in time order.
2. Put the earlier windows into train.
3. Put the later windows into test.
4. Remove the boundary training window if it overlaps the first test window.
5. Shuffle only the training rows.
6. Keep the test rows ordered for easier inspection.
```

Why shuffle only training rows:

```text
Training should see a mixed order of normal/anomaly examples.
Testing does not learn, so test order does not affect accuracy or loss.
```

Outputs:

```text
data/accelerometer/processed/train_window_features.csv
data/accelerometer/processed/test_window_features.csv
data/accelerometer/processed/train_window_features.npy
data/accelerometer/processed/test_window_features.npy
```

### 6. Train the NumPy ANN

```powershell
python ann_from_scratch\src\train.py
```

The ANN uses:

```text
input layer: 20 features
hidden layer: 8 neurons
output layer: 1 neuron
```

The output is a probability:

```text
0 -> normal
1 -> anomaly
```

## Files

- `src/data_processor.py` - parse raw Tera Term logs.
- `src/plot_samples.py` - plot sample-level axes and magnitude.
- `src/features.py` - create window-level feature values.
- `src/extract_window_features.py` - build `window_features.csv`.
- `src/plot_window_features.py` - compare normal/anomaly feature distributions.
- `src/split_window_features.py` - create no-leak train/test splits.
- `src/model_numpy.py` - NumPy ANN, loss, backpropagation, gradient descent.
- `src/train.py` - load data, normalize features, train, and print predictions.

## Current Limitation

The current data is short:

```text
normal:  about 22 seconds
anomaly: about 25 seconds
```

This is enough for learning. For a more trustworthy detector, collect longer and separate recordings.
