"""Prepare a local train/val split from the Roboflow YOLOv8 export.

The current Roboflow export contains all images under `roboflow/train`.
This script copies paired image/label files into the project layout:

    data/yolo_mug_pen/images/train
    data/yolo_mug_pen/images/val
    data/yolo_mug_pen/labels/train
    data/yolo_mug_pen/labels/val

Validation files are selected by original filename group, not by Roboflow's
hashed filename, so duplicate exports of the same source frame stay together.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil


DATASET_ROOT = Path("data/yolo_mug_pen")
ROBOFLOW_ROOT = DATASET_ROOT / "roboflow" / "train"
SOURCE_IMAGES = ROBOFLOW_ROOT / "images"
SOURCE_LABELS = ROBOFLOW_ROOT / "labels"

TARGET_IMAGES = DATASET_ROOT / "images"
TARGET_LABELS = DATASET_ROOT / "labels"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASS_NAMES = {
    "0": "PEN",
    "1": "SUNGLASSES",
}
VAL_FILE_COUNTS = {
    "0": 8,
    "1": 9,
}


def original_group_name(path: Path) -> str:
    """Return the source image id before Roboflow's `.rf.<hash>` suffix."""
    return path.stem.split(".rf.")[0]


def class_id_for_label(label_path: Path) -> str:
    """Return the single class id present in a label file."""
    classes = {
        line.split()[0]
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(classes) != 1:
        raise ValueError(f"{label_path} should contain exactly one class, found {sorted(classes)}")
    return next(iter(classes))


def collect_pairs() -> list[tuple[Path, Path, str, str]]:
    """Collect image/label pairs as `(image, label, class_id, group_name)`."""
    pairs = []

    for label_path in sorted(SOURCE_LABELS.glob("*.txt")):
        image_path = None
        for extension in IMAGE_EXTENSIONS:
            candidate = SOURCE_IMAGES / f"{label_path.stem}{extension}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            raise FileNotFoundError(f"No image found for {label_path}")

        class_id = class_id_for_label(label_path)
        pairs.append((image_path, label_path, class_id, original_group_name(label_path)))

    return pairs


def choose_validation_groups(pairs: list[tuple[Path, Path, str, str]]) -> set[str]:
    """Choose validation groups from the end of each class in sorted order."""
    groups_by_class: dict[str, dict[str, list[tuple[Path, Path, str, str]]]] = defaultdict(lambda: defaultdict(list))

    for pair in pairs:
        _, _, class_id, group_name = pair
        groups_by_class[class_id][group_name].append(pair)

    validation_groups = set()
    for class_id, target_count in VAL_FILE_COUNTS.items():
        selected_count = 0
        for group_name in sorted(groups_by_class[class_id], reverse=True):
            group_size = len(groups_by_class[class_id][group_name])
            if selected_count + group_size > target_count:
                continue
            validation_groups.add(group_name)
            selected_count += group_size
            if selected_count == target_count:
                break

        if selected_count != target_count:
            raise ValueError(
                f"Could not select exactly {target_count} validation files for "
                f"{CLASS_NAMES[class_id]}; selected {selected_count}"
            )

    return validation_groups


def clear_split_outputs() -> None:
    """Remove old copied images/labels while keeping `.gitkeep` files."""
    for folder in [
        TARGET_IMAGES / "train",
        TARGET_IMAGES / "val",
        TARGET_LABELS / "train",
        TARGET_LABELS / "val",
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_file() and (path.suffix.lower() in IMAGE_EXTENSIONS or path.suffix.lower() == ".txt"):
                path.unlink()


def copy_pair(image_path: Path, label_path: Path, split: str) -> None:
    shutil.copy2(image_path, TARGET_IMAGES / split / image_path.name)
    shutil.copy2(label_path, TARGET_LABELS / split / label_path.name)


def main() -> None:
    pairs = collect_pairs()
    validation_groups = choose_validation_groups(pairs)
    clear_split_outputs()

    counts = defaultdict(int)
    class_counts = defaultdict(lambda: defaultdict(int))

    for image_path, label_path, class_id, group_name in pairs:
        split = "val" if group_name in validation_groups else "train"
        copy_pair(image_path, label_path, split)
        counts[split] += 1
        class_counts[split][CLASS_NAMES[class_id]] += 1

    print(f"total pairs: {len(pairs)}")
    for split in ["train", "val"]:
        print(f"{split}: {counts[split]} images")
        for class_name in CLASS_NAMES.values():
            print(f"  {class_name}: {class_counts[split][class_name]}")


if __name__ == "__main__":
    main()
