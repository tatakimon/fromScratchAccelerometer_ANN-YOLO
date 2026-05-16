# YOLO Pen/Sunglasses Dataset

This folder stores the YOLO dataset used by `yolo_mug_pen`.

Active classes:

```text
0 PEN
1 SUNGLASSES
```

## Source Export

Roboflow export:

```text
roboflow/train/images
roboflow/train/labels
```

The Roboflow source folder is kept unchanged.

## Prepared Dataset

The local training layout is:

```text
images/
|-- train/
`-- val/
labels/
|-- train/
`-- val/
dataset.yaml
```

Create or refresh the prepared split with:

```powershell
python yolo_mug_pen\src\prepare_roboflow_split.py
```

Then verify matching image/label pairs with:

```powershell
python yolo_mug_pen\src\check_dataset.py
```
