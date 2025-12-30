"""Export helpers for Kraken-compatible datasets."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import json
import logging
import math
import re
from xml.etree import ElementTree as ET

from PIL import ExifTags, Image

try:  # pragma: no cover - allow importing as a script
    from .line_store import Line
    from .exif_metadata import encode_metadata
except ImportError:  # pragma: no cover - fallback for flat imports
    from line_store import Line
    from exif_metadata import encode_metadata


def _sorted_lines(lines: Iterable[Line]) -> List[Line]:
    return sorted(lines, key=lambda line: line.order_key)


def _normalize_old_pagexml(raw: str) -> str:
    """Normalize legacy PAGE-XML that stored full URIs as tag names.

    Some previous exports wrote tags like ``<http://schema...PcGts>`` instead of
    declaring a namespace. This helper rewrites those into proper PAGE-XML with a
    default namespace so ``xml.etree`` can parse them.
    """

    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

    # Accept tags that inline the full URI with or without a trailing slash, e.g.
    # <http://schema...2019-07-15PcGts> or <http://schema.../PcGts>
    pattern = re.compile(
        r"<(/?)(?:https?://[^>\s]+)?/?"
        r"(PcGts|Metadata|Creator|Created|Page|TextRegion|TextLine|Coords|Baseline|TextEquiv|Unicode)"
        r"([^>]*)>"
    )
    def _repl(m: re.Match[str]) -> str:
        closing, tag, attrs = m.group(1), m.group(2), m.group(3) or ""
        attrs = attrs.strip()
        if attrs:
            attrs = " " + attrs
        return f"<{('/' if closing else '')}{tag}{attrs}>"

    fixed = pattern.sub(_repl, raw)
    # Ensure the root PcGts has the PAGE namespace after normalization.
    fixed = re.sub(r"<PcGts\b", f"<PcGts xmlns=\"{ns}\"", fixed, count=1)
    return fixed


try:
    _ORIENTATION_TAG = next(tag for tag, name in ExifTags.TAGS.items() if name == "Orientation")
except Exception:  # pragma: no cover - Pillow should always expose this tag
    _ORIENTATION_TAG = 274


def load_pagexml(
    pagexml_path: Path,
    image_path: Optional[Path] = None,
    prepared_size: Optional[Tuple[int, int]] = None,
) -> List[Line]:
    """Load annotations from PAGE-XML and normalize them for the GUI.

    ``load_pagexml`` extracts ``TextLine`` entries (Coords + Baseline) and returns
    ``Line`` instances sorted in a top-to-bottom reading order. When ``image_path``
    (and optionally ``prepared_size``) are provided, the coordinates are
    reprojected so they align with the EXIF-corrected image shown in the
    annotation tool.

    Args:
        pagexml_path: Source PAGE-XML path.
        image_path: Optional page image used to infer EXIF orientation.
        prepared_size: Expected ``(width, height)`` of the display image. If
            omitted, it is derived from ``image_path`` when available.

    Returns:
        List of ``Line`` objects suitable for the annotation GUI.

    Raises:
        ValueError: If the XML is malformed or missing required elements.
    """
    try:
        tree = ET.parse(pagexml_path)
        root = tree.getroot()
    except ET.ParseError:
        raw = pagexml_path.read_text(encoding="utf8", errors="ignore")
        stripped = raw.lstrip()
        try:
            root = ET.fromstring(stripped)
        except ET.ParseError:
            # Attempt to fix legacy namespace-less exports that used full URIs as tag names
            repaired = _normalize_old_pagexml(raw)
            try:
                root = ET.fromstring(repaired)
            except ET.ParseError as exc:  # pragma: no cover - defensive fallback
                raise ValueError(f"Invalid PAGE-XML file {pagexml_path}: {exc}")
    
    namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    ns = {"page": namespace}

    page_elem = root.find(".//page:Page", ns)
    page_dims: Optional[Tuple[int, int]] = None
    if page_elem is not None:
        try:
            width = int(page_elem.get("imageWidth", "0"))
            height = int(page_elem.get("imageHeight", "0"))
            if width > 0 and height > 0:
                page_dims = (width, height)
        except (TypeError, ValueError):
            page_dims = None

    lines: List[Line] = []
    line_id_counter = 1
    
    # Find all TextLine elements in the document
    for text_line_elem in root.findall(".//page:TextLine", ns):
        line_id = text_line_elem.get("id", f"l{line_id_counter}")
        
        # Extract bounding box from Coords
        coords_elem = text_line_elem.find("page:Coords", ns)
        if coords_elem is None:
            logging.warning(f"TextLine {line_id} missing Coords element, skipping")
            continue
        
        points_str = coords_elem.get("points", "")
        if not points_str:
            logging.warning(f"TextLine {line_id} has empty Coords, skipping")
            continue
        
        try:
            coord_points = _parse_points(points_str)
            if len(coord_points) < 2:
                logging.warning(f"TextLine {line_id} has insufficient points, skipping")
                continue
            bbox = _coords_to_bbox(coord_points)
        except ValueError as exc:
            logging.warning(f"TextLine {line_id} has invalid Coords: {exc}, skipping")
            continue
        
        # Extract baseline
        baseline_elem = text_line_elem.find("page:Baseline", ns)
        baseline: List[Tuple[float, float]] = []
        if baseline_elem is not None:
            baseline_points_str = baseline_elem.get("points", "")
            if baseline_points_str:
                try:
                    baseline = _parse_points(baseline_points_str)
                except ValueError:
                    logging.warning(f"TextLine {line_id} has invalid Baseline, using empty baseline")
        
        # Extract text content
        text_equiv_elem = text_line_elem.find("page:TextEquiv", ns)
        unicode_elem = None
        if text_equiv_elem is not None:
            unicode_elem = text_equiv_elem.find("page:Unicode", ns)
        
        text = unicode_elem.text if unicode_elem is not None and unicode_elem.text else ""
        
        # Create order key: (line, column, word, char, seq)
        # Use line_id_counter as the line ordinal
        order_key = (line_id_counter, 1, 1, 1, 1)
        
        line = Line(
            id=line_id_counter,
            baseline=baseline,
            bbox=bbox,
            text=text,
            order_key=order_key,
            selected=False,
            is_manual=False,
        )
        lines.append(line)
        line_id_counter += 1
    
    if not lines:
        logging.warning(f"No valid TextLine elements found in {pagexml_path}")
        return lines

    _normalize_pagexml_lines(lines, page_dims, image_path, prepared_size)
    return _sort_lines_by_geometry(lines)


def _parse_points(points_str: str) -> List[Tuple[float, float]]:
    """Parse PAGE-XML points string (comma-separated x,y pairs) into list of tuples.
    
    Args:
        points_str: String like "10,20 30,40 50,60"
        
    Returns:
        List of (x, y) tuples.
        
    Raises:
        ValueError: If points string is malformed.
    """
    points: List[Tuple[float, float]] = []
    for point_str in points_str.strip().split():
        parts = point_str.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid point format: {point_str}")
        try:
            x = float(parts[0])
            y = float(parts[1])
            points.append((x, y))
        except ValueError as exc:
            raise ValueError(f"Could not parse coordinates from '{point_str}': {exc}")
    return points


def _coords_to_bbox(points: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    """Convert polygon points to axis-aligned bounding box.
    
    Args:
        points: List of (x, y) coordinate tuples.
        
    Returns:
        Tuple of (left, top, right, bottom).
    """
    xs = [x for x, y in points]
    ys = [y for x, y in points]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def save_line_crops(image_path: Path, lines: Iterable[Line], out_dir: Path) -> None:
    """Save cropped line images and ``.gt.txt`` files for Kraken training."""

    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        base = image.convert("L")
        width, height = base.size
        padding = 4
        metadata: List[dict] = []
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
            local_bbox = {
                "left": int(left - crop_box[0]),
                "top": int(top - crop_box[1]),
                "right": int(right - crop_box[0]),
                "bottom": int(bottom - crop_box[1]),
            }
            payload = {
                "source_image": image_path.name,
                "line_id": line.id,
                "order_key": list(line.order_key),
                "is_manual": bool(line.is_manual),
                "bbox_page": {
                    "left": int(left),
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                },
                "bbox_local": local_bbox,
                "baseline_page": [[float(x), float(y)] for x, y in line.baseline],
                "baseline_local": [
                    [float(x - crop_box[0]), float(y - crop_box[1])] for x, y in line.baseline
                ],
                "crop_box": {
                    "left": crop_box[0],
                    "top": crop_box[1],
                    "right": crop_box[2],
                    "bottom": crop_box[3],
                },
                "padding": padding,
            }
            exif_bytes = encode_metadata(crop, payload)
            crop.save(image_out, exif=exif_bytes)
            text_out = out_dir / f"{base_name}.gt.txt"
            text_out.write_text(line.text or "", encoding="utf8")
            metadata.append(
                {
                    "line": index,
                    "image": image_out.name,
                    "text_file": text_out.name,
                    "text": line.text or "",
                    "bbox": {
                        "left": int(left),
                        "top": int(top),
                        "right": int(right),
                        "bottom": int(bottom),
                    },
                    "is_manual": bool(line.is_manual),
                }
            )

    if metadata:
        metadata_path = out_dir / f"{image_path.stem}.boxes.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")


def _format_points(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def save_pagexml(image_path: Path, lines: Iterable[Line], out_path: Path) -> None:
    """Write PAGE-XML annotations for the provided ``lines``."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        width, height = image.size
        exif = image.getexif()
        orientation = int(exif.get(_ORIENTATION_TAG, 1))
        if orientation in (5, 6, 7, 8):
            width, height = height, width

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


def _sort_lines_by_geometry(lines: List[Line]) -> List[Line]:
    if not lines:
        return []

    def _key(line: Line) -> Tuple[float, float]:
        left, top, right, bottom = line.bbox
        mid_y = (top + bottom) / 2.0
        return (round(mid_y, 3), float(left))

    ordered = sorted(lines, key=_key)
    for index, line in enumerate(ordered, start=1):
        line.id = index
        line.order_key = (index, 1, 1, 1, 1)
    return ordered


def _normalize_pagexml_lines(
    lines: List[Line],
    page_dims: Optional[Tuple[int, int]],
    image_path: Optional[Path],
    prepared_size: Optional[Tuple[int, int]],
) -> None:
    if not lines:
        return

    target_size = prepared_size
    raw_size: Optional[Tuple[int, int]] = None
    orientation = 1
    if image_path is not None:
        try:
            with Image.open(image_path) as image:
                raw_size = image.size
                exif = image.getexif()
                orientation = int(exif.get(_ORIENTATION_TAG, 1))
                if target_size is None and raw_size:
                    target_size = _orientation_dimensions(raw_size, orientation)
        except OSError:
            raw_size = None

    if target_size is None:
        target_size = page_dims or raw_size
    if target_size is None:
        return

    source_size = page_dims or raw_size or target_size
    current_size = source_size
    transform: Optional[Callable[[float, float], Tuple[float, float]]] = None

    # PAGE-XML produced by external tools (e.g. Kraken) is typically in the raw
    # (non-EXIF-transposed) coordinate system. The annotation UI displays the
    # EXIF-corrected image, so we must reproject coordinates. Previously we only
    # did this when boxes went out-of-bounds; that misses "wrong but in-bounds"
    # cases. For the common rotated orientations (5-8), always reproject when
    # the PAGE-XML coordinate space matches the raw image size.
    needs_orientation = (
        raw_size is not None
        and source_size == raw_size
        and orientation in _ORIENTATION_TRANSFORMS
        and orientation in (5, 6, 7, 8)
    )

    if needs_orientation:
        orient_fn, current_size = _ORIENTATION_TRANSFORMS[orientation](current_size)
        transform = orient_fn

    if current_size != target_size and current_size[0] > 0 and current_size[1] > 0:
        sx = target_size[0] / current_size[0]
        sy = target_size[1] / current_size[1]

        if transform is None:
            transform = lambda x, y, sx=sx, sy=sy: (x * sx, y * sy)
        else:
            previous = transform
            transform = lambda x, y, prev=previous, sx=sx, sy=sy: (
                prev(x, y)[0] * sx,
                prev(x, y)[1] * sy,
            )

    if transform is None:
        return

    _apply_transform_to_lines(lines, transform)


def _orientation_dimensions(size: Tuple[int, int], orientation: int) -> Tuple[int, int]:
    width, height = size
    if orientation in (5, 6, 7, 8):
        return (height, width)
    return size


def _make_orientation_transform(orientation: int, size: Tuple[int, int]):
    width, height = size
    max_x = max(0.0, float(width - 1))
    max_y = max(0.0, float(height - 1))

    if orientation == 1:
        return lambda x, y: (x, y), size
    if orientation == 2:
        return lambda x, y: (max_x - x, y), size
    if orientation == 3:
        return lambda x, y: (max_x - x, max_y - y), size
    if orientation == 4:
        return lambda x, y: (x, max_y - y), size
    if orientation == 5:
        return lambda x, y: (y, x), (height, width)
    if orientation == 6:
        return lambda x, y: (max_y - y, x), (height, width)
    if orientation == 7:
        return lambda x, y: (max_y - y, max_x - x), (height, width)
    if orientation == 8:
        return lambda x, y: (y, max_x - x), (height, width)
    return lambda x, y: (x, y), size


_ORIENTATION_TRANSFORMS: Dict[int, Callable[[Tuple[int, int]], Tuple[Callable[[float, float], Tuple[float, float]], Tuple[int, int]]]] = {
    idx: (lambda size, idx=idx: _make_orientation_transform(idx, size)) for idx in range(1, 9)
}


def _lines_extend_beyond_canvas(lines: List[Line], canvas: Tuple[int, int]) -> bool:
    width, height = canvas
    if width <= 0 or height <= 0:
        return False
    margin_x = max(8.0, width * 0.05)
    margin_y = max(8.0, height * 0.05)
    out_of_bounds = 0
    for line in lines:
        left, top, right, bottom = line.bbox
        if (
            right < -margin_x
            or bottom < -margin_y
            or left > width + margin_x
            or top > height + margin_y
            or right > width + margin_x
            or bottom > height + margin_y
            or left < -margin_x
            or top < -margin_y
        ):
            out_of_bounds += 1
    threshold = max(1, len(lines) // 3)
    return out_of_bounds >= threshold


def _apply_transform_to_lines(lines: List[Line], transform: Callable[[float, float], Tuple[float, float]]) -> None:
    for line in lines:
        left, top, right, bottom = line.bbox
        corners = [
            transform(float(left), float(top)),
            transform(float(right), float(top)),
            transform(float(right), float(bottom)),
            transform(float(left), float(bottom)),
        ]
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        line.bbox = (
            int(math.floor(min(xs))),
            int(math.floor(min(ys))),
            int(math.ceil(max(xs))),
            int(math.ceil(max(ys))),
        )
        if line.baseline:
            line.baseline = [tuple(transform(float(x), float(y))) for x, y in line.baseline]
