from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Tuple

from PIL import Image

try:  # pragma: no cover
    from .exporters import load_pagexml
    from .kraken_adapter import is_available as kraken_available, segment_lines
except ImportError:  # pragma: no cover
    from exporters import load_pagexml  # type: ignore
    from kraken_adapter import is_available as kraken_available, segment_lines  # type: ignore


def load_tokens(
    item: Any,
    options: Any,
    baseline_to_bbox: Callable[[List[Tuple[int, int]]], Tuple[int, int, int, int]],
    extract_tokens: Callable[[Image.Image], List[Any]],
    base_image: Image.Image,
    token_factory: Callable[[str, Tuple[int, int, int, int], Tuple[int, int, int, int, int], Tuple[int, int, int], Tuple[int, int, int]], Any],
) -> Tuple[List[Any], bool, str]:
    """Load line tokens for an item based on the configured segmentation mode."""

    tokens: List[OcrToken]
    pagexml_used = False

    if options.segmentation == "load" and options.pagexml_dir:
        pagexml_path = options.pagexml_dir / f"{item.path.stem}.xml"
        if not pagexml_path.exists():
            return [], False, "PAGE-XML missing; draw boxes manually."
        try:
            prepared_size = getattr(base_image, "size", None)
            lines = load_pagexml(pagexml_path, image_path=item.path, prepared_size=prepared_size)
            tokens = [
                token_factory(
                    line.text,
                    line.bbox,
                    line.order_key,
                    getattr(line, "baseline", (0, 0, 0)),
                    getattr(line, "origin", (0, 0, 0)),
                )
                for line in lines
            ]
            pagexml_used = True
            if not tokens:
                return [], True, "Loaded PAGE-XML but found no TextLine boxes."
            return tokens, True, f"Loaded {len(tokens)} boxes from PAGE-XML."
        except Exception as exc:
            return [], False, f"Invalid PAGE-XML, draw boxes manually: {exc}"

    if options.segmentation == "auto" and options.engine == "kraken" and kraken_available():
        try:
            baselines = segment_lines(item.path)
        except RuntimeError as exc:
            return [], False, str(exc)
        tokens = [
            token_factory(
                "",
                baseline_to_bbox(baseline),
                (1, 1, 1, index + 1, 1),
                (0, 0, 0),
                (0, 0, 0),
            )
            for index, baseline in enumerate(baselines)
        ]
        return tokens, False, "Kraken returned baseline segmentation." if tokens else ""

    tokens = extract_tokens(base_image)
    return tokens, False, "" if tokens else "No boxes found. Switch to Draw mode and drag to add line boxes."
