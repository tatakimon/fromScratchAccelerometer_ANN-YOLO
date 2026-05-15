# PyTorch Learning Workspace

This workspace has two learning projects:

1. `ann_from_scratch` - accelerometer anomaly detection with a NumPy neural network.
2. `yolo_mug_pen` - YOLO detection for hands holding a coffee mug or a pen.

Project 1 now has a complete learning pipeline from raw accelerometer logs to a trained ANN written from scratch. It is still a learning/MVP version, not a production anomaly detector.

## Project Layout

```text
.
|-- ann_from_scratch/
|   |-- README.md
|   `-- src/
|-- yolo_mug_pen/
|-- data/
|   |-- accelerometer/
|   `-- yolo_mug_pen/
|-- docs/
|-- scripts/
|-- requirements.txt
`-- README.md
```

## Environment

Create a virtual environment before installing packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the environment:

```powershell
python scripts\check_env.py
```

## Project 1 Status

Completed steps:

1. Parse Tera Term accelerometer logs.
2. Save clean sample-level CSV and NumPy files.
3. Plot normal/anomaly accelerometer signals.
4. Create 512-sample overlapping windows.
5. Extract 20 statistical features per window.
6. Split train/test data without overlap leakage.
7. Build a one-hidden-layer ANN with NumPy.
8. Add forward pass, binary cross entropy, backpropagation, and gradient descent.

Current dataset is short, around 22-25 seconds per class. It is enough to learn the pipeline, but not enough to claim a reliable anomaly detector.

## Project 1 Command Order

```powershell
python ann_from_scratch\src\data_processor.py
python ann_from_scratch\src\plot_samples.py
python ann_from_scratch\src\extract_window_features.py
python ann_from_scratch\src\plot_window_features.py
python ann_from_scratch\src\split_window_features.py
python ann_from_scratch\src\train.py
```

## Next Work

For Project 1, the best next improvement is collecting longer and more varied recordings:

- 3 minutes normal
- 3 minutes anomaly
- ideally multiple separate recordings per class

For Project 2, the next step is collecting and labeling mug/pen images.
