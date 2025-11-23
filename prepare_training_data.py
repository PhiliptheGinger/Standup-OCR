"""Prepare training data for handwriting OCR fine-tuning.

This script reads a structured JSON dataset describing scanned notebook pages
and produces paired image and ground truth text files suitable for training
models such as Tesseract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Sequence


ARROW_TABS = {
    "main": 0,
    "sub": 1,
    "subsub": 2,
    "sub3": 3,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare OCR training data from JSON annotations.")
    parser.add_argument(
        "--json",
        default="standup_option_c_indexed.json",
        type=Path,
        help="Path to the JSON file containing annotated entries.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Directory containing the original JPEG images.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where processed images and .gt.txt files will be stored.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "Integer offset to add to file_number to reconstruct the actual image number. "
            "For example, if images start at 120.jpg, use --offset 119."
        ),
    )
    return parser.parse_args()


def load_dataset(json_path: Path) -> Dict:
    """Load the dataset from a JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(base_out_dir: Path) -> Dict[str, Path]:
    """Create output subdirectories for images and ground truth files."""
    images_dir = base_out_dir / "images"
    gt_dir = base_out_dir / "gt"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    return {"images": images_dir, "gt": gt_dir}


def indentation_for_arrow(arrow: str) -> str:
    """Return the tab prefix for a given arrow type."""
    level = ARROW_TABS.get(arrow, ARROW_TABS["sub"])
    return "\t" * level


def build_transcription(title: str, lines: Sequence[Dict[str, str]]) -> str:
    """Construct the transcription string for a page."""
    parts: List[str] = []
    if title:
        parts.append(title)

    for line in lines:
        prefix = indentation_for_arrow(line.get("arrow", "sub"))
        parts.append(f"{prefix}{line.get('text', '')}")

    return "\n".join(parts)


def copy_image(src: Path, dest: Path) -> None:
    """Copy an image from source to destination."""
    shutil.copy2(src, dest)


def write_ground_truth(path: Path, text: str) -> None:
    """Write the ground truth transcription to a file."""
    path.write_text(text, encoding="utf-8")


def process_entries(
    entries: Iterable[Dict],
    image_dir: Path,
    out_images_dir: Path,
    out_gt_dir: Path,
    offset: int,
) -> Dict[str, int]:
    """Process entries to generate training data.

    Returns a summary containing the count of processed entries and min/max actual numbers.
    """
    count = 0
    min_number = None
    max_number = None

    for entry in entries:
        actual_number = entry["file_number"] + offset
        src_image = image_dir / f"{actual_number}.jpg"
        dest_image = out_images_dir / f"{actual_number}.jpg"
        dest_gt = out_gt_dir / f"{actual_number}.gt.txt"

        transcription = build_transcription(entry.get("title", ""), entry.get("lines", []))

        copy_image(src_image, dest_image)
        write_ground_truth(dest_gt, transcription)

        count += 1
        if min_number is None or actual_number < min_number:
            min_number = actual_number
        if max_number is None or actual_number > max_number:
            max_number = actual_number

    return {
        "count": count,
        "min_number": min_number if min_number is not None else 0,
        "max_number": max_number if max_number is not None else 0,
    }


def print_summary(summary: Dict[str, int], notebooks: Sequence[Dict]) -> None:
    """Print a summary of processing results."""
    print("Processing complete.")
    print(f"Entries processed: {summary['count']}")
    print(f"Actual number range: {summary['min_number']} - {summary['max_number']}")
    print("Notebook ranges:")
    for notebook in notebooks:
        name = notebook.get("name", "unknown")
        page_min = notebook.get("page_min", "?")
        page_max = notebook.get("page_max", "?")
        print(f"  {name}: pages {page_min} to {page_max}")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.json)

    out_dirs = ensure_dirs(args.out_dir)

    summary = process_entries(
        entries=dataset.get("entries", []),
        image_dir=args.image_dir,
        out_images_dir=out_dirs["images"],
        out_gt_dir=out_dirs["gt"],
        offset=args.offset,
    )

    print_summary(summary, dataset.get("notebooks", []))


if __name__ == "__main__":
    main()
