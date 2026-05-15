"""Train a YOLO model on the mug/pen hand dataset."""

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.pt")
    model.train(
        data="data/yolo_mug_pen/dataset.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        project="runs/yolo_mug_pen",
        name="first_run",
    )


if __name__ == "__main__":
    main()

