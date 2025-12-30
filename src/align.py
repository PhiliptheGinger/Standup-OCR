from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Any

from dataclasses import dataclass

if TYPE_CHECKING:  # pragma: no cover
    from .annotation import OverlayItem  # type: ignore


@dataclass
class AlignmentResult:
    assignments: List[Tuple[Any, str]]
    extra_segments: List[str]
    overlays_to_remove: List[Any]


def _line_similarity(text: str, hint: str, scorer: Callable[[str, str], float]) -> float:
    try:
        return float(scorer(text, hint))
    except Exception:
        return 0.0


def align_segments_to_overlays(
    segments: Sequence[str],
    overlays: Sequence[Any],
    *,
    use_text_scoring: bool,
    text_similarity: Callable[[str, str], float],
) -> AlignmentResult:
    """Greedy alignment of transcript segments to overlays with optional text scoring."""

    if not overlays:
        return AlignmentResult([], list(segments), [])

    centers: List[Tuple[float, Any, str]] = []
    for overlay in overlays:
        y_center = (overlay.bbox[1] + overlay.bbox[3]) / 2.0
        text_hint = overlay.token.text if overlay.token and overlay.token.text else overlay.entry.get()
        centers.append((y_center, overlay, text_hint))

    min_center = min((c[0] for c in centers), default=0.0)
    max_center = max((c[0] for c in centers), default=0.0)
    span = max(1.0, max_center - min_center)
    available: List[Tuple[float, Any, str]] = [
        ((center - min_center) / span if span else 0.0, overlay, text_hint)
        for center, overlay, text_hint in centers
    ]

    assignments: List[Tuple[Any, str]] = []
    extra_segments: List[str] = []

    for idx, text in enumerate(segments):
        if not available:
            extra_segments.append(text)
            continue
        target = 0.0 if len(segments) == 1 else idx / (len(segments) - 1)
        best_index = 0
        best_score: Optional[float] = None
        for i, (norm_center, overlay, text_hint) in enumerate(available):
            if use_text_scoring:
                geom_score = 1.0 - min(abs(norm_center - target), 1.0)
                if geom_score < 0.0:
                    geom_score = 0.0
                text_score = _line_similarity(text, text_hint, text_similarity)
                score = 0.6 * geom_score + 0.4 * text_score
                key = score
            else:
                key = -abs(norm_center - target)
            if best_score is None or key > best_score:
                best_score = key
                best_index = i
        _norm_center, best_overlay, _hint = available.pop(best_index)
        assignments.append((best_overlay, text))

    assigned_ids = {id(overlay) for overlay, _ in assignments}
    overlays_to_remove = [overlay for overlay in overlays if id(overlay) not in assigned_ids]
    return AlignmentResult(assignments, extra_segments, overlays_to_remove)
