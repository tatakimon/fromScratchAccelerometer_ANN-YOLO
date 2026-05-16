"""Run the trained YOLO model on validation images and save predictions."""

import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("runs/ultralytics_config").resolve()))

from ultralytics import YOLO


WEIGHTS = Path("runs/detect/runs/yolo_mug_pen/first_run/weights/best.pt")
VAL_IMAGES = Path("data/yolo_mug_pen/images/val")
OUTPUT_PROJECT = Path("runs/yolo_mug_pen_predictions").resolve()


def main() -> None:
    model = YOLO(str(WEIGHTS))
    results = model.predict(
        source=str(VAL_IMAGES),
        conf=0.25,
        imgsz=640,
        save=True,
        project=str(OUTPUT_PROJECT),
        name="val",
        exist_ok=True,
    )

    print(f"predicted images: {len(results)}")
    print(f"output folder: {OUTPUT_PROJECT / 'val'}")


if __name__ == "__main__":
    main()
