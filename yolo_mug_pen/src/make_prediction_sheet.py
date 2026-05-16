"""Create one contact-sheet image from YOLO prediction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def resize_to_tile(image: Image.Image, tile_width: int, tile_height: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (tile_width, tile_height), "white")
    x = (tile_width - image.width) // 2
    y = (tile_height - image.height) // 2
    tile.paste(image, (x, y))
    return tile


def make_sheet(
    image_paths: list[Path],
    output_path: Path,
    columns: int,
    tile_width: int,
    tile_height: int,
) -> None:
    rows = (len(image_paths) + columns - 1) // columns
    label_height = 28
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(image_paths):
        row = index // columns
        column = index % columns
        x = column * tile_width
        y = row * (tile_height + label_height)

        with Image.open(image_path) as image:
            tile = resize_to_tile(image, tile_width, tile_height)

        sheet.paste(tile, (x, y))
        draw.text((x + 8, y + tile_height + 6), image_path.name[:42], fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a grid image from prediction outputs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("runs/yolo_mug_pen_predictions/val"),
        help="Folder containing predicted images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/yolo_mug_pen_predictions/prediction_sheet.png"),
        help="Output contact-sheet image.",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--tile-height", type=int, default=520)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    image_paths = find_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No prediction images found in {args.input}")

    make_sheet(
        image_paths=image_paths,
        output_path=args.output,
        columns=args.columns,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
    )
    print(f"images: {len(image_paths)}")
    print(f"sheet:  {args.output}")


if __name__ == "__main__":
    main()
