# YOLO Pen/Sunglasses Detection

Goal: train a YOLOv8 model to detect two classes:

```text
0 PEN
1 SUNGLASSES
```

The folder name is still `yolo_mug_pen` because this project started as a mug/pen idea. The active dataset is now pen/sunglasses.

## Dataset Source

The current images and labels come from a Roboflow YOLOv8 export:

```text
data/yolo_mug_pen/roboflow/
```

Roboflow exported all files into `roboflow/train`, so we prepare our own local train/validation split.

Run:

```powershell
python yolo_mug_pen\src\prepare_roboflow_split.py
```

This copies paired files into:

```text
data/yolo_mug_pen/images/train
data/yolo_mug_pen/images/val
data/yolo_mug_pen/labels/train
data/yolo_mug_pen/labels/val
```

## Dataset Layout

YOLO expects matching image and label files:

```text
data/yolo_mug_pen/
|-- images/
|   |-- train/
|   `-- val/
|-- labels/
|   |-- train/
|   `-- val/
`-- dataset.yaml
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

## Check Dataset

After preparing the split:

```powershell
python yolo_mug_pen\src\check_dataset.py
```

Only train after image/label pairs are valid.

## Train

```powershell
python yolo_mug_pen\src\train.py
```

The first run uses `yolov8n.pt`, a small YOLOv8 model suitable for learning.
