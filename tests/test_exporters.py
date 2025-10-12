from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from exporters import save_line_crops  # type: ignore
from line_store import Line  # type: ignore


def test_save_line_crops_writes_metadata(tmp_path):
    image_path = tmp_path / "page.png"
    Image.new("L", (40, 20), color=255).save(image_path)

    out_dir = tmp_path / "train"
    lines = [
        Line(
            id=1,
            baseline=[(5, 10), (25, 10)],
            bbox=(4, 6, 26, 12),
            text="Sample text",
            order_key=(0, 0, 0, 1, 1),
            selected=False,
            is_manual=True,
        ),
        Line(
            id=2,
            baseline=[(5, 15), (30, 15)],
            bbox=(4, 13, 30, 18),
            text="Second line",
            order_key=(0, 0, 0, 2, 1),
            selected=False,
            is_manual=False,
        ),
    ]

    save_line_crops(image_path, lines, out_dir)

    metadata_path = out_dir / f"{image_path.stem}.boxes.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    assert len(metadata) == 2

    first = metadata[0]
    assert first["image"].endswith("_line01.png")
    assert first["text_file"].endswith("_line01.gt.txt")
    assert first["text"] == "Sample text"
    assert first["bbox"] == {"left": 4, "top": 6, "right": 26, "bottom": 12}
    assert first["is_manual"] is True

    second = metadata[1]
    assert second["is_manual"] is False
    assert second["text"] == "Second line"
