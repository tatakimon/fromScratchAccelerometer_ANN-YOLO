# Learning Plan

## Project 1: ANN From Accelerometer Data

Goal: learn the full neural-network pipeline by building the model with NumPy.

Current status: first complete learning version is done.

Completed concepts:

1. Raw data parsing
2. Signal plotting
3. Windowing time-series data
4. Feature extraction
5. Train/test splitting without overlap leakage
6. Input normalization
7. ANN forward pass
8. Sigmoid output for binary classification
9. Binary cross entropy loss
10. Backpropagation
11. Gradient descent
12. Basic evaluation with accuracy and predictions

Important design choices:

- Use 512-sample windows because the sensor logger is configured for 512 samples per axis.
- Use `step_size = 256` to create 50% overlap.
- Split train/test by time order before shuffling training rows.
- Remove the boundary train window if it overlaps the first test window.
- Shuffle only training rows, not raw samples and not the test set.
- Normalize using training mean/std only.
- Use 20 window-level features as ANN inputs.

Current ANN shape:

```text
20 inputs -> 8 hidden neurons -> 1 output
```

Current limitation:

The current dataset is short. It is good for learning, but not enough to prove generalization.

Recommended next improvements:

1. Collect 3 minutes normal and 3 minutes anomaly.
2. Collect multiple separate recordings per class.
3. Keep one full recording as a final test set.
4. Save trained model weights and normalization values.
5. Compare the NumPy model with a PyTorch implementation.

## Project 2: YOLO Pen/Sunglasses Detection

Goal: train a YOLO model for two classes:

```text
0 PEN
1 SUNGLASSES
```

Dataset idea:

- Around 69 images are available now.
- Use varied backgrounds, lighting, camera angles, and hand positions.
- Keep train and validation images separate.

Learning steps:

1. Collect images.
2. Label bounding boxes.
3. Check image/label pairs.
4. Train a small YOLO model.
5. Test on new images.
6. Improve the dataset based on mistakes.

## Rule

Move one concept at a time. For every model result, ask whether the data split and evaluation are fair before trusting the number.
