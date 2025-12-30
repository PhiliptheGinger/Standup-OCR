from __future__ import annotations

"""X-schema alignment models, planners, and appliers."""

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]
TokenOrder = Tuple[int, int, int, int, int]
SpanId = str
PageId = str

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TranscriptSpan:
    id: SpanId
    page_id: PageId
    text: str
    line_index: int
    char_start: int
    char_end: int
    kind: Literal["line", "word"] = "line"
    source_file: Optional[Path] = None
    source_line_no: Optional[int] = None
    pagexml_id: Optional[str] = None
    words: Optional[List[str]] = None


@dataclass
class LayoutSpan:
    id: SpanId
    page_id: PageId
    bbox: BBox
    baseline: List[Point]
    order_key: TokenOrder
    text_hint: str
    is_manual: bool
    paragraph_key: Optional[Tuple[int, int, int]] = None
    line_key: Optional[Tuple[int, int, int]] = None
    overlay_item: Any | None = None
    overlay_store_id: Optional[int] = None
    line: Any | None = None
    pagexml_id: Optional[str] = None
    is_virtual: bool = False
    column_index: Optional[int] = None


@dataclass
class Block:
    id: str
    page_id: PageId
    transcript_spans: List[TranscriptSpan]
    layout_spans: List[LayoutSpan]
    source: Literal["pagexml", "kraken", "manual", "mixed", "legacy"]
    region_id: Optional[str] = None


@dataclass
class AlignmentLink:
    transcripts: List[TranscriptSpan]
    layouts: List[LayoutSpan]
    kind: Literal["one_to_one", "many_to_one", "one_to_many", "many_to_many"] = "one_to_one"
    score: float = 0.0


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@dataclass
class AssignText:
    layout_id: SpanId
    transcript_id: SpanId
    text: str


@dataclass
class CreateBox:
    new_layout_id: SpanId
    transcript_id: SpanId
    text: str
    after_layout_id: Optional[SpanId] = None
    before_layout_id: Optional[SpanId] = None
    bbox: Optional[BBox] = None


@dataclass
class HideBox:
    layout_id: SpanId


@dataclass
class MergeBoxes:
    target_layout_id: SpanId
    source_layout_ids: List[SpanId]
    transcript_id: SpanId
    text: str
    new_bbox: Optional[BBox] = None


@dataclass
class SplitBox:
    layout_id: SpanId
    transcript_ids: List[SpanId]
    result_layout_ids: List[SpanId]
    new_bboxes: Optional[List[BBox]] = None


@dataclass
class MoveGroup:
    layout_ids: List[SpanId]
    dx: int
    dy: int


@dataclass
class ResizeBox:
    layout_id: SpanId
    new_bbox: BBox


XOperation = AssignText | CreateBox | HideBox | MergeBoxes | SplitBox | MoveGroup | ResizeBox


@dataclass
class AlignmentPlan:
    page_id: PageId
    blocks: List[Block]
    links: List[AlignmentLink]
    operations: List[XOperation] = field(default_factory=list)
    telemetry: Optional["PlannerTelemetry"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "links": [_serialize_link(link) for link in self.links],
            "operations": [_serialize_operation(op) for op in self.operations],
            "telemetry": self.telemetry.summary() if self.telemetry else None,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class PlannerTelemetry:
    page_id: Optional[PageId] = None
    blocks: int = 0
    links: int = 0
    merges: int = 0
    splits: int = 0
    assignments: int = 0
    created: int = 0
    hidden: int = 0
    move_groups: int = 0
    resize_boxes: int = 0
    extra_transcripts: int = 0
    residual_layouts: int = 0
    notes: List[str] = field(default_factory=list)

    def record_block(self) -> None:
        self.blocks += 1

    def record_links(self, count: int) -> None:
        self.links += count

    def record_extras(self, count: int) -> None:
        self.extra_transcripts += count

    def record_residuals(self, count: int) -> None:
        self.residual_layouts += count

    def add_note(self, message: str) -> None:
        self.notes.append(message)

    def track_operations(self, operations: Sequence[XOperation]) -> None:
        for op in operations:
            if isinstance(op, AssignText):
                self.assignments += 1
            elif isinstance(op, MergeBoxes):
                self.merges += 1
            elif isinstance(op, SplitBox):
                self.splits += 1
            elif isinstance(op, CreateBox):
                self.created += 1
            elif isinstance(op, HideBox):
                self.hidden += 1
            elif isinstance(op, MoveGroup):
                self.move_groups += 1
            elif isinstance(op, ResizeBox):
                self.resize_boxes += 1

    @property
    def total_operations(self) -> int:
        return (
            self.assignments
            + self.merges
            + self.splits
            + self.created
            + self.hidden
            + self.move_groups
            + self.resize_boxes
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "blocks": self.blocks,
            "links": self.links,
            "operations": self.total_operations,
            "merges": self.merges,
            "splits": self.splits,
            "creates": self.created,
            "hides": self.hidden,
            "move_groups": self.move_groups,
            "resize_boxes": self.resize_boxes,
            "extra_transcripts": self.extra_transcripts,
            "residual_layouts": self.residual_layouts,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _normalize_transcript(text: str) -> List[Tuple[str, int, int]]:
    normalized = text.replace("\r\n", "\n").rstrip("\n")
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        start = cursor
        end = cursor + len(raw_line)
        if line:
            spans.append((line, start, end))
        cursor = end + 1
    return spans


def extract_transcript_spans(page_id: PageId, transcript_text: str) -> List[TranscriptSpan]:
    spans = []
    for idx, (line, start, end) in enumerate(_normalize_transcript(transcript_text)):
        words = [token for token in line.split() if token]
        spans.append(
            TranscriptSpan(
                id=f"ts:{page_id}:{idx}",
                page_id=page_id,
                text=line,
                line_index=idx,
                char_start=start,
                char_end=end,
                words=words or None,
            )
        )
    return spans


def _overlay_text(overlay: Any) -> str:
    entry = getattr(overlay, "entry", None)
    if entry is not None:
        getter = getattr(entry, "get", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                pass
        value = getattr(entry, "value", None)
        if isinstance(value, str):
            return value
    token = getattr(overlay, "token", None)
    if token is not None and getattr(token, "text", None):
        return token.text
    return ""


def extract_layout_spans_from_overlays(page_id: PageId, overlays: Sequence[Any]) -> List[LayoutSpan]:
    spans: List[LayoutSpan] = []
    for overlay in overlays:
        bbox = getattr(overlay, "bbox", getattr(overlay, "bbox_base", (0, 0, 0, 0)))
        baseline = list(getattr(getattr(overlay, "token", None), "baseline", []))
        order_key = getattr(overlay, "order_key", (0, 0, 0, 0, 0))
        span = LayoutSpan(
            id=f"ls:{page_id}:{getattr(overlay, 'rect_id', id(overlay))}",
            page_id=page_id,
            bbox=bbox,
            baseline=baseline,
            order_key=order_key,
            text_hint=_overlay_text(overlay),
            is_manual=bool(getattr(overlay, "is_manual", False)),
            paragraph_key=getattr(overlay, "token", None) and getattr(overlay.token, "paragraph_key", None),
            line_key=getattr(overlay, "token", None) and getattr(overlay.token, "line_key", None),
            overlay_item=overlay,
        )
        spans.append(span)
    spans.sort(key=lambda span: (span.order_key, span.bbox[1], span.bbox[0]))
    return spans


def extract_layout_spans_from_lines(page_id: PageId, lines: Sequence[Any]) -> List[LayoutSpan]:
    spans: List[LayoutSpan] = []
    for line in lines:
        span = LayoutSpan(
            id=f"line:{page_id}:{getattr(line, 'id', id(line))}",
            page_id=page_id,
            bbox=getattr(line, "bbox", (0, 0, 0, 0)),
            baseline=list(getattr(line, "baseline", [])),
            order_key=getattr(line, "order_key", (0, 0, 0, 0, 0)),
            text_hint=getattr(line, "text", ""),
            is_manual=bool(getattr(line, "is_manual", False)),
            line=line,
            pagexml_id=getattr(line, "pagexml_id", None),
        )
        spans.append(span)
    spans.sort(key=lambda span: (span.order_key, span.bbox[1], span.bbox[0]))
    return spans


def build_blocks_for_page(
    page_id: PageId,
    transcript_spans: Sequence[TranscriptSpan],
    layout_spans: Sequence[LayoutSpan],
    *,
    source: Literal["pagexml", "kraken", "manual", "mixed", "legacy"] = "manual",
) -> List[Block]:
    return [
        Block(
            id=f"block:{page_id}:0",
            page_id=page_id,
            transcript_spans=list(transcript_spans),
            layout_spans=list(layout_spans),
            source=source,
        )
    ]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@dataclass
class PlannerConfig:
    use_text_scoring: bool = False
    geom_weight: float = 0.6
    text_weight: float = 0.4
    segmentation_mode: Literal["pagexml_trusted", "auto", "manual"] = "auto"
    enable_merges: bool = True
    enable_splits: bool = True
    enable_block_ops: bool = True
    merge_gap_ratio: float = 0.15
    merge_gain_threshold: float = 0.15
    split_height_threshold: int = 40
    block_move_threshold: int = 8
    block_resize_slack: int = 4
    word_score_weight: float = 0.35
    column_merge_slack: int = 45
    enable_reflow: bool = True
    reflow_min_lines: int = 6
    reflow_span_ratio: float = 0.65
    reflow_line_spacing_multiplier: float = 1.25


class AlignmentPlanner:
    def __init__(self, config: PlannerConfig, text_similarity: Callable[[str, str], float]) -> None:
        self.config = config
        self._text_similarity = text_similarity

    def plan(
        self,
        page_id: PageId,
        blocks: Sequence[Block],
        telemetry: Optional[PlannerTelemetry] = None,
    ) -> AlignmentPlan:
        active_telemetry = telemetry or PlannerTelemetry(page_id=page_id)
        if active_telemetry.page_id is None:
            active_telemetry.page_id = page_id
        links: List[AlignmentLink] = []
        operations: List[XOperation] = []
        for block in blocks:
            active_telemetry.record_block()
            block_links, block_ops = self._plan_block(block, active_telemetry)
            links.extend(block_links)
            operations.extend(block_ops)
        plan = AlignmentPlan(page_id=page_id, blocks=list(blocks), links=links, operations=operations, telemetry=active_telemetry)
        self._log_plan_summary(plan.telemetry)
        return plan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _plan_block(
        self,
        block: Block,
        telemetry: Optional[PlannerTelemetry],
    ) -> Tuple[List[AlignmentLink], List[XOperation]]:
        if not block.transcript_spans and not block.layout_spans:
            return [], []

        sorted_layouts = self._sorted_layouts(block.layout_spans)
        columns, layout_columns = self._derive_columns(sorted_layouts)
        for layout in sorted_layouts:
            if layout.id in layout_columns:
                layout.column_index = layout_columns[layout.id]
        min_center, span = self._center_stats(sorted_layouts, len(block.transcript_spans))
        median_height = self._median_line_height(sorted_layouts)
        line_gap = self._median_line_gap(sorted_layouts)
        if line_gap <= 0.0:
            line_gap = median_height + max(2, self.config.block_resize_slack)
        fallback_left, fallback_right = self._default_horizontal_bounds(sorted_layouts)
        layout_span_raw = self._layout_vertical_span(sorted_layouts)
        needs_reflow = self._needs_reflow(block, sorted_layouts, layout_span_raw, median_height)
        available = self._prepare_layout_queue(sorted_layouts, min_center, span)
        links: List[AlignmentLink] = []
        extra_transcripts: List[TranscriptSpan] = []
        layout_by_id = {layout.id: layout for layout in sorted_layouts}
        last_assigned_center: Optional[float] = None

        for idx, transcript in enumerate(block.transcript_spans):
            if not available:
                extra_transcripts.append(transcript)
                continue
            target_pos = self._target_position(idx, len(block.transcript_spans))
            best_index: Optional[int] = None
            best_score: Optional[float] = None
            for i, (norm_center, _center_value, layout) in enumerate(available):
                score = self._score_pair(transcript, layout.text_hint, target_pos, norm_center)
                score += self._monotonic_bias(last_assigned_center, self._center_y(layout.bbox))
                if best_score is None or score > best_score:
                    best_score = score
                    best_index = i
            if best_index is None:
                extra_transcripts.append(transcript)
                continue
            norm_center, center_value, layout = available.pop(best_index)
            last_assigned_center = center_value
            link = AlignmentLink(
                transcripts=[transcript],
                layouts=[layout],
                kind="one_to_one",
                score=best_score or 0.0,
            )
            links.append(link)

        unused_layouts = [layout for _norm, _center, layout in available]

        if self.config.enable_merges and unused_layouts and block.layout_spans:
            self._promote_merges(links, unused_layouts)
            if telemetry and unused_layouts:
                telemetry.add_note(f"Merged {len(unused_layouts)} overlays in {block.id}")

        linked_layout_ids = {layout.id for link in links for layout in link.layouts}
        residual_layout_ids = {
            layout.id
            for layout in block.layout_spans
            if layout.id not in linked_layout_ids
        }

        if self.config.enable_splits and extra_transcripts and links:
            self._schedule_splits(links, extra_transcripts)
            if telemetry and extra_transcripts:
                telemetry.add_note(f"Split {len(extra_transcripts)} transcripts in {block.id}")
            extra_transcripts = []

        operations: List[XOperation] = []

        assigned_layout_ids = set()
        for link in links:
            if link.kind == "many_to_one":
                merge_op = MergeBoxes(
                    target_layout_id=link.layouts[0].id,
                    source_layout_ids=[layout.id for layout in link.layouts],
                    transcript_id=link.transcripts[0].id,
                    text=link.transcripts[0].text,
                    new_bbox=_union_bbox([layout.bbox for layout in link.layouts]),
                )
                operations.append(merge_op)
                assigned_layout_ids.update(layout.id for layout in link.layouts)
                operations.append(
                    AssignText(
                        layout_id=link.layouts[0].id,
                        transcript_id=link.transcripts[0].id,
                        text=link.transcripts[0].text,
                    )
                )
            elif link.kind == "one_to_many":
                result_layout_ids = [layout.id for layout in link.layouts]
                new_bboxes = [layout.bbox for layout in link.layouts]
                transcript_ids = [span.id for span in link.transcripts]
                operations.append(
                    SplitBox(
                        layout_id=link.layouts[0].id,
                        transcript_ids=transcript_ids,
                        result_layout_ids=result_layout_ids,
                        new_bboxes=new_bboxes,
                    )
                )
                for transcript, layout in zip(link.transcripts, link.layouts):
                    operations.append(
                        AssignText(
                            layout_id=layout.id,
                            transcript_id=transcript.id,
                            text=transcript.text,
                        )
                    )
                    assigned_layout_ids.add(layout.id)
            else:
                operations.append(
                    AssignText(
                        layout_id=link.layouts[0].id,
                        transcript_id=link.transcripts[0].id,
                        text=link.transcripts[0].text,
                    )
                )
                assigned_layout_ids.add(link.layouts[0].id)

        for layout_id in residual_layout_ids:
            if layout_id in assigned_layout_ids:
                continue
            layout = layout_by_id.get(layout_id)
            if layout is None:
                continue
            if layout.is_manual:
                continue
            if layout.text_hint and layout.text_hint.strip():
                continue
            operations.append(HideBox(layout_id=layout_id))

        if extra_transcripts:
            anchor_layout_id: Optional[SpanId] = links[-1].layouts[0].id if links else None
            reference_bbox: Optional[BBox] = links[-1].layouts[0].bbox if links else None
            anchor_column_bounds = self._column_bounds_for_layout(anchor_layout_id, layout_columns, columns)
            virtual_center = last_assigned_center
            for transcript in extra_transcripts:
                virtual_center = self._next_virtual_center(virtual_center, line_gap, min_center)
                bbox = self._virtual_bbox(
                    reference_bbox,
                    fallback_left,
                    fallback_right,
                    virtual_center,
                    median_height,
                    anchor_column_bounds,
                )
                new_layout_id = f"create:{transcript.id}"
                operations.append(
                    CreateBox(
                        new_layout_id=new_layout_id,
                        transcript_id=transcript.id,
                        text=transcript.text,
                        after_layout_id=anchor_layout_id,
                        bbox=bbox,
                    )
                )
                anchor_layout_id = new_layout_id
                reference_bbox = bbox
                anchor_column_bounds = (bbox[0], bbox[2])

        if needs_reflow and links:
            reflow_ops = self._plan_reflow_resizes(
                links,
                columns,
                fallback_left,
                fallback_right,
                median_height,
                line_gap,
            )
            if reflow_ops:
                operations.extend(reflow_ops)
                if telemetry:
                    telemetry.add_note(f"Reflowed {len(reflow_ops)} layouts in {block.id}")

        if telemetry:
            telemetry.record_links(len(links))
            telemetry.record_extras(len(extra_transcripts))
            telemetry.record_residuals(len(residual_layout_ids))
            telemetry.track_operations(operations)

        if self.config.enable_block_ops:
            extra_ops = self._block_level_adjustments(block, links, min_center, span, telemetry)
            operations.extend(extra_ops)
            if telemetry:
                telemetry.track_operations(extra_ops)

        return links, operations

    def _log_plan_summary(self, telemetry: Optional[PlannerTelemetry]) -> None:
        if telemetry is None:
            return
        summary = telemetry.summary()
        logger.debug(
            "Planner summary page=%s blocks=%d links=%d ops=%d merges=%d splits=%d creates=%d hides=%d move_groups=%d resize=%d",
            summary["page_id"],
            summary["blocks"],
            summary["links"],
            summary["operations"],
            summary["merges"],
            summary["splits"],
            summary["creates"],
            summary["hides"],
            summary["move_groups"],
            summary["resize_boxes"],
        )
        for note in summary["notes"]:
            logger.debug("Planner note: %s", note)

    def _prepare_layout_queue(
        self,
        layouts: Sequence[LayoutSpan],
        min_center: float,
        span: float,
    ) -> List[Tuple[float, float, LayoutSpan]]:
        if not layouts:
            return []
        queue: List[Tuple[float, float, LayoutSpan]] = []
        for layout in layouts:
            center = self._center_y(layout.bbox)
            norm = (center - min_center) / span if span else 0.0
            queue.append((norm, center, layout))
        return queue

    def _center_stats(self, layouts: Sequence[LayoutSpan], transcript_count: int) -> Tuple[float, float]:
        if not layouts:
            return 0.0, 1.0
        centers = [self._center_y(layout.bbox) for layout in layouts]
        min_center = min(centers)
        max_center = max(centers)
        span = max(1.0, max_center - min_center)
        layout_count = len(layouts)
        if transcript_count > 1 and layout_count > 1 and layout_count > transcript_count:
            layout_span = max(1.0, layout_count - 1)
            transcript_span = max(1.0, transcript_count - 1)
            ratio = min(1.0, transcript_span / layout_span)
            span = max(1.0, (max_center - min_center) * ratio)
        return min_center, span

    def _target_position(self, index: int, count: int) -> float:
        if count <= 1:
            return 0.0
        return index / (count - 1)

    def _score_pair(self, transcript: TranscriptSpan, hint: str, target: float, norm_center: float) -> float:
        geom_score = 1.0 - min(abs(norm_center - target), 1.0)
        if geom_score < 0.0:
            geom_score = 0.0
        if not self.config.use_text_scoring:
            return geom_score
        text_score = self._text_similarity(transcript.text, hint)
        if transcript.words:
            word_score = max((self._text_similarity(word, hint) for word in transcript.words), default=0.0)
            blended = max(text_score, word_score * (1.0 + self.config.word_score_weight))
            text_score = min(1.0, blended)
        return self.config.geom_weight * geom_score + self.config.text_weight * text_score

    def _monotonic_bias(self, last_center: Optional[float], candidate_center: float) -> float:
        if last_center is None:
            return 0.0
        delta = candidate_center - last_center
        if delta < -1.0:
            backtrack = abs(delta)
            return -min(0.3, 0.02 + (backtrack / 80.0))
        forward_slack = max(1.0, self.config.block_move_threshold)
        if delta <= forward_slack:
            return 0.02
        excess = delta - forward_slack
        penalty = min(0.6, 0.02 + (excess / forward_slack) * 0.05)
        return -penalty

    def _sorted_layouts(self, layouts: Sequence[LayoutSpan]) -> List[LayoutSpan]:
        if not layouts:
            return []
        return sorted(
            layouts,
            key=lambda span: (span.order_key, span.bbox[1], span.bbox[0]),
        )

    def _derive_columns(self, layouts: Sequence[LayoutSpan]) -> Tuple[List[Tuple[int, int]], Dict[SpanId, int]]:
        if not layouts:
            return [], {}
        slack = max(5, self.config.column_merge_slack)
        columns: List[Dict[str, float]] = []
        mapping: Dict[SpanId, int] = {}
        for span in sorted(layouts, key=lambda item: item.bbox[0]):
            left, right = span.bbox[0], span.bbox[2]
            assigned = False
            for idx, column in enumerate(columns):
                if self._within_column(left, right, column, slack):
                    count = column["count"] + 1
                    column["left"] = ((column["left"] * column["count"]) + left) / count
                    column["right"] = ((column["right"] * column["count"]) + right) / count
                    column["count"] = count
                    mapping[span.id] = idx
                    assigned = True
                    break
            if not assigned:
                columns.append({"left": float(left), "right": float(right), "count": 1.0})
                mapping[span.id] = len(columns) - 1
        normalized: List[Tuple[int, int]] = []
        for column in columns:
            left = int(round(column["left"]))
            right = int(round(column["right"]))
            if right <= left:
                right = left + max(10, self.config.split_height_threshold)
            normalized.append((left, right))
        return normalized, mapping

    def _within_column(self, left: int, right: int, column: Dict[str, float], slack: int) -> bool:
        return (
            abs(left - column["left"]) <= slack
            and abs(right - column["right"]) <= slack
        )

    def _column_bounds_for_layout(
        self,
        layout_id: Optional[SpanId],
        layout_columns: Dict[SpanId, int],
        columns: Sequence[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        if layout_id is None:
            return None
        column_index = layout_columns.get(layout_id)
        if column_index is None:
            return None
        if 0 <= column_index < len(columns):
            return columns[column_index]
        return None

    def _median_line_height(self, layouts: Sequence[LayoutSpan]) -> int:
        if not layouts:
            return max(12, self.config.split_height_threshold)
        heights = [self._bbox_height(span.bbox) for span in layouts]
        try:
            return max(1, int(round(statistics.median(heights))))
        except statistics.StatisticsError:
            return max(1, heights[0])

    def _median_line_gap(self, layouts: Sequence[LayoutSpan]) -> float:
        centers = sorted(self._center_y(span.bbox) for span in layouts)
        if len(centers) < 2:
            return 0.0
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1) if centers[i + 1] > centers[i]]
        if not gaps:
            return 0.0
        try:
            return float(statistics.median(gaps))
        except statistics.StatisticsError:
            return gaps[0]

    def _layout_vertical_span(self, layouts: Sequence[LayoutSpan]) -> float:
        if not layouts:
            return 0.0
        centers = [self._center_y(span.bbox) for span in layouts]
        return max(centers) - min(centers)

    def _default_horizontal_bounds(self, layouts: Sequence[LayoutSpan]) -> Tuple[int, int]:
        if not layouts:
            return (0, self.config.split_height_threshold * 3)
        lefts = [span.bbox[0] for span in layouts]
        rights = [span.bbox[2] for span in layouts]
        try:
            median_left = int(round(statistics.median(lefts)))
            median_right = int(round(statistics.median(rights)))
        except statistics.StatisticsError:
            median_left, median_right = lefts[0], rights[0]
        if median_right <= median_left:
            median_right = median_left + max(10, self.config.split_height_threshold)
        return median_left, median_right

    def _next_virtual_center(self, current: Optional[float], line_gap: float, min_center: float) -> float:
        gap = line_gap if line_gap > 0 else max(12.0, float(self.config.split_height_threshold))
        if current is None:
            base = min_center if min_center else 0.0
        else:
            base = current
        return base + gap

    def _virtual_bbox(
        self,
        reference_bbox: Optional[BBox],
        fallback_left: int,
        fallback_right: int,
        center_y: float,
        line_height: int,
        column_bounds: Optional[Tuple[int, int]] = None,
    ) -> BBox:
        if column_bounds is not None:
            left, right = column_bounds
        elif reference_bbox is not None:
            left, _, right, _ = reference_bbox
        else:
            left, right = fallback_left, fallback_right
        if right <= left:
            right = left + max(10, line_height)
        half_height = max(1, line_height // 2)
        top = int(round(center_y - half_height))
        bottom = top + max(1, line_height)
        return (int(left), top, int(right), bottom)

    def _promote_merges(self, links: List[AlignmentLink], unused_layouts: List[LayoutSpan]) -> None:
        if not links:
            return
        sorted_unused = sorted(unused_layouts, key=lambda span: self._center_y(span.bbox))
        for layout in sorted_unused:
            target_link = self._best_merge_target(layout, links)
            if target_link is None:
                continue
            target_link.layouts.append(layout)
            target_link.kind = "many_to_one"

    def _best_merge_target(self, layout: LayoutSpan, links: List[AlignmentLink]) -> Optional[AlignmentLink]:
        best: Optional[AlignmentLink] = None
        best_gain: Optional[float] = None
        for link in links:
            gain = self._merge_gain(layout, link)
            if gain is None:
                continue
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best = link
        if best_gain is None or best_gain < self.config.merge_gain_threshold:
            return None
        return best

    def _merge_gain(self, layout: LayoutSpan, link: AlignmentLink) -> Optional[float]:
        # Be very conservative when segmentation is based on trusted PAGE-XML.
        gap_limit = self.config.merge_gap_ratio
        gain_threshold = self.config.merge_gain_threshold
        if self.config.segmentation_mode == "pagexml_trusted":
            gap_limit = min(gap_limit, 0.05)
            gain_threshold = max(gain_threshold, 0.5)
        target_layout = link.layouts[0]
        gap_ratio = self._vertical_gap_ratio(layout.bbox, target_layout.bbox)
        if gap_ratio > gap_limit:
            return None
        text_hint = layout.text_hint.strip()
        transcript_text = link.transcripts[0].text if link.transcripts else ""
        text_score = 1.0 if not text_hint else self._text_similarity(text_hint, transcript_text)
        gap_score = 1.0 - gap_ratio
        gain = (gap_score * 0.7) + (text_score * 0.3)
        if gain < gain_threshold:
            return None
        return gain

    def _vertical_gap_ratio(self, bbox_a: BBox, bbox_b: BBox) -> float:
        center_a = self._center_y(bbox_a)
        center_b = self._center_y(bbox_b)
        height = max(1.0, min(bbox_a[3] - bbox_a[1], bbox_b[3] - bbox_b[1]))
        return abs(center_a - center_b) / height

    def _bbox_height(self, bbox: BBox) -> int:
        return max(1, bbox[3] - bbox[1])

    def _block_level_adjustments(
        self,
        block: Block,
        links: Sequence[AlignmentLink],
        min_center: float,
        span: float,
        telemetry: Optional[PlannerTelemetry],
    ) -> List[XOperation]:
        ops: List[XOperation] = []
        move_op = self._plan_move_group(block, links, min_center, span)
        if move_op is not None:
            ops.append(move_op)
        resize_ops = self._plan_resize_operations(block)
        ops.extend(resize_ops)
        if telemetry and ops:
            telemetry.add_note(f"Block ops in {block.id}: {len(ops)} change(s)")
        return ops

    def _plan_move_group(
        self,
        block: Block,
        links: Sequence[AlignmentLink],
        min_center: float,
        span: float,
    ) -> Optional[MoveGroup]:
        if not links or span <= 1.0 or len(block.transcript_spans) <= 1:
            return None
        line_min, line_span = self._line_index_stats(block.transcript_spans)
        if line_span == 0:
            return None
        layout_ids: List[SpanId] = []
        dy_samples: List[float] = []
        for link in links:
            base_layout = link.layouts[0]
            layout_ids.append(base_layout.id)
            transcript_index = min((span.line_index for span in link.transcripts), default=line_min)
            target_norm = (transcript_index - line_min) / line_span if line_span else 0.0
            target_center = min_center + (target_norm * span)
            dy_samples.append(target_center - self._center_y(base_layout.bbox))
        if not dy_samples:
            return None
        median_shift = statistics.median(dy_samples)
        if abs(median_shift) < self.config.block_move_threshold:
            return None
        max_deviation = max(abs(value - median_shift) for value in dy_samples)
        if max_deviation > max(3.0, abs(median_shift) * 0.5):
            return None
        dy = int(round(median_shift))
        if dy == 0:
            return None
        return MoveGroup(layout_ids=list(dict.fromkeys(layout_ids)), dx=0, dy=dy)

    def _plan_resize_operations(self, block: Block) -> List[XOperation]:
        # Never resize when segmentation is manual or comes from PAGE-XML –
        # treat those boxes as ground truth containers.
        if self.config.segmentation_mode in ("manual", "pagexml_trusted"):
            return []
        ops: List[XOperation] = []
        slack = max(1, self.config.block_resize_slack)
        for span in block.layout_spans:
            if span.is_manual or span.text_hint.strip():
                continue
            height = self._bbox_height(span.bbox)
            if height <= self.config.split_height_threshold:
                continue
            new_top = span.bbox[1] + slack
            new_bottom = span.bbox[3] - slack
            if new_bottom - new_top <= self.config.split_height_threshold // 2:
                continue
            ops.append(
                ResizeBox(
                    layout_id=span.id,
                    new_bbox=(span.bbox[0], new_top, span.bbox[2], new_bottom),
                )
            )
        return ops

    def _needs_reflow(
        self,
        block: Block,
        layouts: Sequence[LayoutSpan],
        layout_span: float,
        median_height: int,
    ) -> bool:
        if not self.config.enable_reflow:
            return False
        if block.source == "manual":
            return False
        transcript_count = len(block.transcript_spans)
        if transcript_count < self.config.reflow_min_lines:
            return False
        auto_layouts = [span for span in layouts if not span.is_manual]
        if len(auto_layouts) < max(2, transcript_count // 2):
            return False
        expected_span = float(median_height * max(1, transcript_count - 1))
        expected_span *= max(1.0, self.config.reflow_line_spacing_multiplier)
        if expected_span <= 0.0:
            return False
        ratio = layout_span / expected_span if expected_span else 0.0
        return ratio < self.config.reflow_span_ratio

    def _plan_reflow_resizes(
        self,
        links: Sequence[AlignmentLink],
        columns: Sequence[Tuple[int, int]],
        fallback_left: int,
        fallback_right: int,
        median_height: int,
        line_gap: float,
    ) -> List[ResizeBox]:
        if not links:
            return []
        candidates: List[AlignmentLink] = []
        for link in links:
            if not link.layouts:
                continue
            layout = link.layouts[0]
            if not self._is_reflow_candidate(layout):
                continue
            candidates.append(link)
        if len(candidates) < self.config.reflow_min_lines:
            return []
        min_gap = median_height + max(2, self.config.block_resize_slack)
        gap = max(line_gap, float(min_gap))
        gap *= max(1.0, self.config.reflow_line_spacing_multiplier)
        return self._redistribute_links(
            candidates,
            columns,
            fallback_left,
            fallback_right,
            median_height,
            gap,
        )

    def _is_reflow_candidate(self, layout: LayoutSpan) -> bool:
        if layout.is_manual or layout.is_virtual:
            return False
        return True

    def _redistribute_links(
        self,
        links: Sequence[AlignmentLink],
        columns: Sequence[Tuple[int, int]],
        fallback_left: int,
        fallback_right: int,
        median_height: int,
        line_gap: float,
    ) -> List[ResizeBox]:
        if not links:
            return []
        buckets: Dict[int, List[AlignmentLink]] = {}
        fallback_bucket: List[AlignmentLink] = []
        for link in links:
            layout = link.layouts[0]
            column_idx = layout.column_index
            if column_idx is None or column_idx >= len(columns):
                fallback_bucket.append(link)
            else:
                buckets.setdefault(column_idx, []).append(link)
        ops: List[ResizeBox] = []
        for column_index in sorted(buckets):
            bounds = columns[column_index]
            ops.extend(
                self._reflow_column_links(
                    buckets[column_index],
                    bounds,
                    fallback_left,
                    fallback_right,
                    line_gap,
                    median_height,
                )
            )
        if fallback_bucket:
            ops.extend(
                self._reflow_column_links(
                    fallback_bucket,
                    None,
                    fallback_left,
                    fallback_right,
                    line_gap,
                    median_height,
                )
            )
        return ops

    def _reflow_column_links(
        self,
        links: Sequence[AlignmentLink],
        column_bounds: Optional[Tuple[int, int]],
        fallback_left: int,
        fallback_right: int,
        line_gap: float,
        line_height: int,
    ) -> List[ResizeBox]:
        if not links:
            return []
        centers = [self._center_y(link.layouts[0].bbox) for link in links]
        start_center = centers[0] if centers else 0.0
        if centers:
            start_center = min(centers)
        gap = max(1.0, line_gap)
        operations: List[ResizeBox] = []
        for idx, link in enumerate(links):
            layout = link.layouts[0]
            target_center = start_center + (idx * gap)
            bbox = self._virtual_bbox(
                layout.bbox,
                fallback_left,
                fallback_right,
                target_center,
                max(1, line_height),
                column_bounds,
            )
            operations.append(ResizeBox(layout_id=layout.id, new_bbox=bbox))
        return operations

    def _line_index_stats(self, transcripts: Sequence[TranscriptSpan]) -> Tuple[int, int]:
        if not transcripts:
            return 0, 0
        indices = [span.line_index for span in transcripts]
        line_min = min(indices)
        line_span = max(1, max(indices) - line_min)
        return line_min, line_span

    def _schedule_splits(self, links: List[AlignmentLink], extras: List[TranscriptSpan]) -> None:
        if not links:
            return
        height_threshold = self.config.split_height_threshold
        if self.config.segmentation_mode == "pagexml_trusted":
            height_threshold = max(height_threshold * 2, height_threshold + 40)
        tall_links = [link for link in links if self._bbox_height(link.layouts[0].bbox) >= height_threshold]
        candidates = tall_links or links
        if not candidates:
            return
        candidate_index = 0
        for transcript in extras:
            target_link = candidates[candidate_index % len(candidates)]
            target_link.transcripts.append(transcript)
            target_link.kind = "one_to_many"
            candidate_index += 1
        for link in links:
            if link.kind != "one_to_many":
                continue
            target_layout = link.layouts[0]
            total = len(link.transcripts)
            weights = [max(1, len(span.text)) for span in link.transcripts]
            bboxes = _split_bbox(target_layout.bbox, total, weights)
            new_layouts = [target_layout]
            for idx in range(1, total):
                new_layouts.append(
                    LayoutSpan(
                        id=f"{target_layout.id}#split{idx}",
                        page_id=target_layout.page_id,
                        bbox=bboxes[idx],
                        baseline=list(target_layout.baseline),
                        order_key=target_layout.order_key,
                        text_hint="",
                        is_manual=True,
                        is_virtual=True,
                        column_index=target_layout.column_index,
                    )
                )
            target_layout.bbox = bboxes[0]
            link.layouts = new_layouts

    def _center_y(self, bbox: BBox) -> float:
        return (bbox[1] + bbox[3]) / 2.0


def _union_bbox(bboxes: Iterable[BBox]) -> Optional[BBox]:
    bboxes = list(bboxes)
    if not bboxes:
        return None
    left = min(bbox[0] for bbox in bboxes)
    top = min(bbox[1] for bbox in bboxes)
    right = max(bbox[2] for bbox in bboxes)
    bottom = max(bbox[3] for bbox in bboxes)
    return left, top, right, bottom


def _split_bbox(bbox: BBox, parts: int, weights: Optional[Sequence[int]] = None) -> List[BBox]:
    if parts <= 1:
        return [bbox]
    left, top, right, bottom = bbox
    height = max(1, bottom - top)
    if weights is None or len(weights) != parts:
        weights = [1] * parts
    normalized = [max(1, value) for value in weights]
    total = sum(normalized) or parts
    bboxes: List[BBox] = []
    current_top = top
    for idx, weight in enumerate(normalized):
        if idx == parts - 1:
            current_bottom = bottom
        else:
            portion = max(1, int(round(height * (weight / total))))
            current_bottom = min(bottom, current_top + portion)
        bboxes.append((left, current_top, right, current_bottom))
        current_top = current_bottom
    if len(bboxes) > parts:
        bboxes = bboxes[:parts]
    if bboxes:
        last = bboxes[-1]
        bboxes[-1] = (last[0], last[1], last[2], bottom)
    return bboxes


def _serialize_link(link: AlignmentLink) -> Dict[str, Any]:
    return {
        "transcripts": [span.id for span in link.transcripts],
        "layouts": [layout.id for layout in link.layouts],
        "kind": link.kind,
        "score": link.score,
    }


def _serialize_operation(op: XOperation) -> Dict[str, Any]:
    if isinstance(op, AssignText):
        return {
            "op": "AssignText",
            "layout_id": op.layout_id,
            "transcript_id": op.transcript_id,
            "text": op.text,
        }
    if isinstance(op, CreateBox):
        return {
            "op": "CreateBox",
            "new_layout_id": op.new_layout_id,
            "transcript_id": op.transcript_id,
            "text": op.text,
            "after_layout_id": op.after_layout_id,
            "before_layout_id": op.before_layout_id,
            "bbox": op.bbox,
        }
    if isinstance(op, HideBox):
        return {
            "op": "HideBox",
            "layout_id": op.layout_id,
        }
    if isinstance(op, MergeBoxes):
        return {
            "op": "MergeBoxes",
            "target_layout_id": op.target_layout_id,
            "source_layout_ids": list(op.source_layout_ids),
            "transcript_id": op.transcript_id,
            "text": op.text,
            "new_bbox": op.new_bbox,
        }
    if isinstance(op, SplitBox):
        return {
            "op": "SplitBox",
            "layout_id": op.layout_id,
            "transcript_ids": list(op.transcript_ids),
            "result_layout_ids": list(op.result_layout_ids),
            "new_bboxes": list(op.new_bboxes) if op.new_bboxes else None,
        }
    if isinstance(op, MoveGroup):
        return {
            "op": "MoveGroup",
            "layout_ids": list(op.layout_ids),
            "dx": op.dx,
            "dy": op.dy,
        }
    if isinstance(op, ResizeBox):
        return {
            "op": "ResizeBox",
            "layout_id": op.layout_id,
            "new_bbox": op.new_bbox,
        }
    raise TypeError(f"Unsupported operation for serialization: {type(op)!r}")


def plan_to_json(plan: AlignmentPlan, *, indent: Optional[int] = None) -> str:
    return plan.to_json(indent=indent)


# ---------------------------------------------------------------------------
# Applier
# ---------------------------------------------------------------------------


class XContext(Protocol):
    def ensure_snapshot(self) -> None:
        ...

    def has_layout(self, layout_id: SpanId) -> bool:
        ...

    def create_layout(
        self,
        layout_id: SpanId,
        *,
        text: str,
        bbox: Optional[BBox],
        after_layout_id: Optional[SpanId],
        before_layout_id: Optional[SpanId],
    ) -> None:
        ...

    def set_layout_text(self, layout_id: SpanId, text: str) -> None:
        ...

    def set_layout_bbox(self, layout_id: SpanId, bbox: BBox) -> None:
        ...

    def get_layout_bbox(self, layout_id: SpanId) -> BBox:
        ...

    def remove_layout(self, layout_id: SpanId) -> None:
        ...


class XApplier:
    def __init__(self, context: XContext) -> None:
        self.context = context
        self._snapshot_taken = False

    def apply(self, operations: Sequence[XOperation]) -> None:
        if not operations:
            return
        self._ensure_snapshot()
        for op in operations:
            if isinstance(op, AssignText):
                self.context.set_layout_text(op.layout_id, op.text)
            elif isinstance(op, CreateBox):
                self.context.create_layout(
                    op.new_layout_id,
                    text=op.text,
                    bbox=op.bbox,
                    after_layout_id=op.after_layout_id,
                    before_layout_id=op.before_layout_id,
                )
            elif isinstance(op, HideBox):
                if self.context.has_layout(op.layout_id):
                    self.context.remove_layout(op.layout_id)
            elif isinstance(op, MergeBoxes):
                if op.new_bbox is not None:
                    self.context.set_layout_bbox(op.target_layout_id, op.new_bbox)
                for extra_id in op.source_layout_ids:
                    if extra_id == op.target_layout_id:
                        continue
                    if self.context.has_layout(extra_id):
                        self.context.remove_layout(extra_id)
            elif isinstance(op, SplitBox):
                self._apply_split(op)
            elif isinstance(op, MoveGroup):
                self._apply_move(op)
            elif isinstance(op, ResizeBox):
                if self.context.has_layout(op.layout_id):
                    self.context.set_layout_bbox(op.layout_id, op.new_bbox)

    def _apply_split(self, op: SplitBox) -> None:
        if not op.result_layout_ids:
            return
        bboxes = op.new_bboxes or []
        for idx, layout_id in enumerate(op.result_layout_ids):
            bbox = bboxes[idx] if idx < len(bboxes) else None
            if idx == 0:
                if bbox is not None:
                    self.context.set_layout_bbox(layout_id, bbox)
            else:
                self.context.create_layout(
                    layout_id,
                    text="",
                    bbox=bbox,
                    after_layout_id=op.result_layout_ids[idx - 1],
                    before_layout_id=None,
                )

    def _apply_move(self, op: MoveGroup) -> None:
        for layout_id in op.layout_ids:
            if not self.context.has_layout(layout_id):
                continue
            bbox = self.context.get_layout_bbox(layout_id)
            left, top, right, bottom = bbox
            new_bbox = (left + op.dx, top + op.dy, right + op.dx, bottom + op.dy)
            self.context.set_layout_bbox(layout_id, new_bbox)

    def _ensure_snapshot(self) -> None:
        if not self._snapshot_taken:
            self.context.ensure_snapshot()
            self._snapshot_taken = True


__all__ = [
    "AlignmentLink",
    "AlignmentPlan",
    "AlignmentPlanner",
    "AssignText",
    "Block",
    "CreateBox",
    "HideBox",
    "LayoutSpan",
    "MergeBoxes",
    "MoveGroup",
    "PlannerConfig",
    "PlannerTelemetry",
    "ResizeBox",
    "SplitBox",
    "TranscriptSpan",
    "XApplier",
    "XContext",
    "build_blocks_for_page",
    "extract_layout_spans_from_lines",
    "extract_layout_spans_from_overlays",
    "extract_transcript_spans",
    "plan_to_json",
]
