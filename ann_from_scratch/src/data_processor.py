"""Convert Tera Term accelerometer logs into clean training files.

Input line format:

    [2026-05-14 20:06:52.009] x1 y1 z1 x2 y2 z2 ...

The logger stores many accelerometer samples on one timestamped line. This
script expands those triples into one row per sample.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_RATE_HZ = 416.0
DEFAULT_RAW_DIR = Path("data/accelerometer/raw")
DEFAULT_PROCESSED_DIR = Path("data/accelerometer/processed")
TIMESTAMP_RE = re.compile(r"^\[(?P<timestamp>[^\]]+)\]\s*(?P<values>.*)$")
FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class ParseStats:
    source: Path
    label_name: str
    label: int
    lines_seen: int
    lines_parsed: int
    lines_skipped: int
    samples: int
    duration_seconds: float
    truncated_values: int
    output_csv: Path
    output_npy: Path


def parse_log_line(line: str) -> tuple[datetime, np.ndarray] | None:
    """Parse one timestamped log line into a timestamp and ``N x 3`` array."""
    match = TIMESTAMP_RE.match(line.strip())
    if match is None:
        return None

    timestamp = datetime.fromisoformat(match.group("timestamp"))
    values = [float(value) for value in FLOAT_RE.findall(match.group("values"))]

    usable_count = (len(values) // 3) * 3
    if usable_count == 0:
        return None

    samples = np.asarray(values[:usable_count], dtype=np.float32).reshape(-1, 3)
    return timestamp, samples


def expand_log_file(path: str | Path, label_name: str, label: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read one Tera Term log and return one dataframe row per accelerometer sample."""
    log_path = Path(path)
    rows = []
    sequence_index = 0
    lines_seen = 0
    lines_parsed = 0
    truncated_values = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            lines_seen += 1
            parsed = parse_log_line(line)
            if parsed is None:
                continue

            line_timestamp, samples = parsed
            lines_parsed += 1

            value_count = len(FLOAT_RE.findall(TIMESTAMP_RE.match(line.strip()).group("values")))
            truncated_values += value_count % 3

            for sample_offset, (x_mg, y_mg, z_mg) in enumerate(samples):
                # Keep both the logger timestamp and a reconstructed sample time.
                # The reconstructed time uses the known sensor rate: 416 samples/sec.
                sample_time = line_timestamp + timedelta(seconds=sample_offset / SAMPLE_RATE_HZ)
                rows.append(
                    {
                        "sample_index": sequence_index,
                        "timestamp": sample_time.isoformat(timespec="milliseconds"),
                        "line_timestamp": line_timestamp.isoformat(timespec="milliseconds"),
                        "line_number": line_number,
                        "sample_in_line": sample_offset,
                        "x_mg": float(x_mg),
                        "y_mg": float(y_mg),
                        "z_mg": float(z_mg),
                        "x_g": float(x_mg / 1000.0),
                        "y_g": float(y_mg / 1000.0),
                        "z_g": float(z_mg / 1000.0),
                        "label": label,
                        "label_name": label_name,
                    }
                )
                sequence_index += 1

    data = pd.DataFrame(rows)
    stats = {
        "lines_seen": lines_seen,
        "lines_parsed": lines_parsed,
        "lines_skipped": lines_seen - lines_parsed,
        "samples": len(data),
        "truncated_values": truncated_values,
    }
    return data, stats


def process_log_file(
    source: str | Path,
    output_dir: str | Path,
    label_name: str,
    label: int,
) -> ParseStats:
    """Parse one log file and save CSV plus NumPy array outputs."""
    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data, stats = expand_log_file(source_path, label_name, label)
    output_csv = output_path / f"{label_name}_samples.csv"
    output_npy = output_path / f"{label_name}_samples.npy"

    data.to_csv(output_csv, index=False)
    np.save(output_npy, data[["x_g", "y_g", "z_g", "label"]].to_numpy(dtype=np.float32))

    return ParseStats(
        source=source_path,
        label_name=label_name,
        label=label,
        lines_seen=stats["lines_seen"],
        lines_parsed=stats["lines_parsed"],
        lines_skipped=stats["lines_skipped"],
        samples=stats["samples"],
        duration_seconds=stats["samples"] / SAMPLE_RATE_HZ,
        truncated_values=stats["truncated_values"],
        output_csv=output_csv,
        output_npy=output_npy,
    )


def process_default_dataset(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> list[ParseStats]:
    """Process the expected normal/anomaly Tera Term logs."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    jobs = [
        (raw_path / "normal" / "teraterm.log", "normal", 0),
        (raw_path / "anomaly" / "teraterm.log", "anomaly", 1),
    ]

    results = []
    for source, label_name, label in jobs:
        if not source.exists():
            print(f"missing: {source}")
            continue
        results.append(process_log_file(source, processed_path, label_name, label))

    return results


def print_stats(results: list[ParseStats]) -> None:
    for result in results:
        print(f"{result.label_name}:")
        print(f"  source:           {result.source}")
        print(f"  lines seen:       {result.lines_seen}")
        print(f"  lines parsed:     {result.lines_parsed}")
        print(f"  lines skipped:    {result.lines_skipped}")
        print(f"  samples:          {result.samples}")
        print(f"  duration seconds: {result.duration_seconds:.2f}")
        print(f"  truncated values: {result.truncated_values}")
        print(f"  csv:              {result.output_csv}")
        print(f"  npy:              {result.output_npy}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process Tera Term accelerometer logs.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, type=Path)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR, type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results = process_default_dataset(args.raw_dir, args.processed_dir)
    print_stats(results)


if __name__ == "__main__":
    main()
