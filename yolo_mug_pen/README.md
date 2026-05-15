# YOLO Mug/Pen Hand Detection

Goal: train a YOLO model to detect two classes:

- `holding_mug`
- `holding_pen`

Start with around 100 images. More important than count: variety and correct labels.

## Dataset Layout

YOLO expects matching image and label files:

```text
data/yolo_mug_pen/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

For example:

```text
images/train/img_001.jpg
labels/train/img_001.txt
```

Each label file contains bounding boxes in YOLO format:

```text
class_id center_x center_y width height
```

All coordinates are normalized from 0 to 1.

## Label IDs

```text
0 holding_mug
1 holding_pen
```

## First Milestone

1. Collect 100 images.
2. Split roughly 80 train and 20 validation.
3. Label with a tool such as Label Studio, CVAT, Roboflow, or makesense.ai.
4. Run `python yolo_mug_pen/src/check_dataset.py`.
5. Only then run training.

