"""Export helpers for Kraken-compatible datasets."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List
from xml.etree import ElementTree as ET

from PIL import Image

try:  # pragma: no cover - allow importing as a script
    from .line_store import Line
except ImportError:  # pragma: no cover - fallback for flat imports
    from line_store import Line


def _sorted_lines(lines: Iterable[Line]) -> List[Line]:
    return sorted(lines, key=lambda line: line.order_key)


def save_line_crops(image_path: Path, lines: Iterable[Line], out_dir: Path) -> None:
    """Save cropped line images and ``.gt.txt`` files for Kraken training."""

    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        base = image.convert("L")
        width, height = base.size
        padding = 4
        for index, line in enumerate(_sorted_lines(lines), start=1):
            left, top, right, bottom = line.bbox
            crop_box = (
                max(0, int(left) - padding),
                max(0, int(top) - padding),
                min(width, int(right) + padding),
                min(height, int(bottom) + padding),
            )
            crop = base.crop(crop_box)
            base_name = f"{image_path.stem}_line{index:02d}"
            image_out = out_dir / f"{base_name}.png"
            crop.save(image_out)
            text_out = out_dir / f"{base_name}.gt.txt"
            text_out.write_text(line.text or "", encoding="utf8")


def _format_points(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def save_pagexml(image_path: Path, lines: Iterable[Line], out_path: Path) -> None:
    """Write PAGE-XML annotations for the provided ``lines``."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        width, height = image.size

    namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    ET.register_namespace("", namespace)
    root = ET.Element("{namespace}PcGts".format(namespace=namespace))

    metadata = ET.SubElement(root, "{namespace}Metadata".format(namespace=namespace))
    ET.SubElement(metadata, "{namespace}Creator".format(namespace=namespace)).text = "Standup-OCR"
    ET.SubElement(metadata, "{namespace}Created".format(namespace=namespace)).text = datetime.utcnow().isoformat()

    page = ET.SubElement(
        root,
        "{namespace}Page".format(namespace=namespace),
        {
            "imageFilename": image_path.name,
            "imageWidth": str(width),
            "imageHeight": str(height),
        },
    )
    region = ET.SubElement(page, "{namespace}TextRegion".format(namespace=namespace), {"id": "r1"})

    for index, line in enumerate(_sorted_lines(lines), start=1):
        line_id = f"l{line.id or index}"
        text_line = ET.SubElement(region, "{namespace}TextLine".format(namespace=namespace), {"id": line_id})
        left, top, right, bottom = line.bbox
        coords = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        ]
        ET.SubElement(
            text_line,
            "{namespace}Coords".format(namespace=namespace),
            {"points": _format_points(coords)},
        )
        ET.SubElement(
            text_line,
            "{namespace}Baseline".format(namespace=namespace),
            {"points": _format_points(line.baseline)},
        )
        text_equiv = ET.SubElement(text_line, "{namespace}TextEquiv".format(namespace=namespace))
        unicode_el = ET.SubElement(text_equiv, "{namespace}Unicode".format(namespace=namespace))
        unicode_el.text = line.text or ""

    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
