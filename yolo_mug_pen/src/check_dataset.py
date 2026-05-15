"""Check whether the YOLO dataset has matching image and label files."""

from pathlib import Path


DATASET_ROOT = Path("data/yolo_mug_pen")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_stems(folder: Path, extensions: set[str]) -> set[str]:
    return {
        path.stem
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }


def check_split(split: str) -> bool:
    image_folder = DATASET_ROOT / "images" / split
    label_folder = DATASET_ROOT / "labels" / split

    image_stems = collect_stems(image_folder, IMAGE_EXTENSIONS)
    label_stems = collect_stems(label_folder, {".txt"})

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    print(f"{split}: {len(image_stems)} images, {len(label_stems)} labels")

    if missing_labels:
        print(f"  labels missing for: {missing_labels[:10]}")
    if missing_images:
        print(f"  images missing for: {missing_images[:10]}")

    return not missing_labels and not missing_images


def main() -> None:
    train_ok = check_split("train")
    val_ok = check_split("val")

    if train_ok and val_ok:
        print("Dataset file matching looks ok.")
    else:
        print("Fix missing image/label pairs before training.")


if __name__ == "__main__":
    main()

