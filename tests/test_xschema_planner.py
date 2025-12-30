import json
import importlib.util
import sys
from pathlib import Path

_XSHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "xschema.py"
_XSHEMA_SPEC = importlib.util.spec_from_file_location("xschema_test_module", _XSHEMA_PATH)
_XSHEMA = importlib.util.module_from_spec(_XSHEMA_SPEC)
assert _XSHEMA_SPEC is not None and _XSHEMA_SPEC.loader is not None
sys.modules[_XSHEMA_SPEC.name] = _XSHEMA
_XSHEMA_SPEC.loader.exec_module(_XSHEMA)

AlignmentPlanner = _XSHEMA.AlignmentPlanner
PlannerConfig = _XSHEMA.PlannerConfig
Block = _XSHEMA.Block
HideBox = _XSHEMA.HideBox
ResizeBox = _XSHEMA.ResizeBox
LayoutSpan = _XSHEMA.LayoutSpan
TranscriptSpan = _XSHEMA.TranscriptSpan


def _make_transcript(line_index: int, text: str, page_id: str = "page-1") -> TranscriptSpan:
    return TranscriptSpan(
        id=f"ts:{page_id}:{line_index}",
        page_id=page_id,
        text=text,
        line_index=line_index,
        char_start=line_index * 10,
        char_end=(line_index * 10) + len(text),
    )


def _make_layout(order: int, bbox, layout_id: str, text_hint: str = "", *, manual: bool = False, page_id: str = "page-1") -> LayoutSpan:
    return LayoutSpan(
        id=layout_id,
        page_id=page_id,
        bbox=bbox,
        baseline=[],
        order_key=(order, 0, 0, 0, 0),
        text_hint=text_hint,
        is_manual=manual,
    )


def test_alignment_plan_serializes_to_dict_and_json():
    planner = AlignmentPlanner(PlannerConfig(), lambda _a, _b: 0.0)
    transcripts = [_make_transcript(0, "hello")]
    layouts = [_make_layout(0, (0, 0, 100, 20), "ls:page-1:0", text_hint="hello")]
    block = Block(
        id="block:page-1:0",
        page_id="page-1",
        transcript_spans=transcripts,
        layout_spans=layouts,
        source="manual",
    )
    plan = planner.plan("page-1", [block])

    payload = plan.to_dict()
    assert payload["page_id"] == "page-1"
    assert payload["links"]
    assert payload["operations"]
    expected_ops = plan.telemetry.total_operations if plan.telemetry else 0
    assert payload["telemetry"]["operations"] == expected_ops

    json_payload = plan.to_json()
    parsed = json.loads(json_payload)
    assert parsed["page_id"] == "page-1"
    assert parsed["links"] == payload["links"]


def test_planner_only_hides_auto_blank_residuals():
    planner = AlignmentPlanner(PlannerConfig(), lambda _a, _b: 0.0)
    transcripts = [_make_transcript(0, "aligned"), _make_transcript(1, "second")]
    layouts = [
        _make_layout(0, (0, 0, 100, 20), "ls:page-1:aligned"),
        _make_layout(1, (0, 40, 100, 60), "ls:page-1:manual", manual=True),
        _make_layout(2, (0, 80, 100, 100), "ls:page-1:text", text_hint="UI"),
        _make_layout(3, (0, 120, 100, 160), "ls:page-1:auto_blank"),
    ]
    block = Block(
        id="block:page-1:0",
        page_id="page-1",
        transcript_spans=transcripts,
        layout_spans=layouts,
        source="manual",
    )
    plan = planner.plan("page-1", [block])

    hide_ops = [op for op in plan.operations if isinstance(op, HideBox)]
    assert [op.layout_id for op in hide_ops] == ["ls:page-1:auto_blank"]


def test_create_box_uses_reference_bbox_when_no_splits_possible():
    # Force a scenario with more transcript lines than layouts where splits are
    # not used, so the planner must emit a CreateBox for the extra transcript.
    cfg = PlannerConfig(enable_splits=False)
    planner = AlignmentPlanner(cfg, lambda _a, _b: 0.0)
    transcripts = [_make_transcript(0, "first"), _make_transcript(1, "second")]
    base_bbox = (10, 20, 110, 40)
    layouts = [_make_layout(0, base_bbox, "ls:page-1:base")]
    block = Block(
        id="block:page-1:0",
        page_id="page-1",
        transcript_spans=transcripts,
        layout_spans=layouts,
        source="manual",
    )
    plan = planner.plan("page-1", [block])

    create_ops = [op for op in getattr(plan, "operations", []) if type(op).__name__ == "CreateBox"]
    assert len(create_ops) == 1
    bbox = create_ops[0].bbox
    # The new box should align horizontally with the reference layout but be
    # positioned below it vertically.
    assert bbox[0] == base_bbox[0]
    assert bbox[2] == base_bbox[2]
    assert bbox[1] > base_bbox[1]


def test_create_boxes_stack_when_no_layouts_present():
    cfg = PlannerConfig(enable_splits=False)
    planner = AlignmentPlanner(cfg, lambda _a, _b: 0.0)
    transcripts = [
        _make_transcript(0, "l0"),
        _make_transcript(1, "l1"),
        _make_transcript(2, "l2"),
    ]
    block = Block(
        id="block:page-1:0",
        page_id="page-1",
        transcript_spans=transcripts,
        layout_spans=[],
        source="manual",
    )
    plan = planner.plan("page-1", [block])
    create_ops = [op for op in getattr(plan, "operations", []) if type(op).__name__ == "CreateBox"]
    assert len(create_ops) == len(transcripts)
    tops = [bbox[1] for bbox in [op.bbox for op in create_ops]]
    assert tops == sorted(tops)
    assert len(set(tops)) == len(tops)


def test_reflow_spreads_collapsed_layouts():
    cfg = PlannerConfig(
        enable_splits=False,
        reflow_min_lines=3,
        reflow_span_ratio=0.95,
        reflow_line_spacing_multiplier=1.0,
    )
    planner = AlignmentPlanner(cfg, lambda _a, _b: 0.0)
    transcripts = [_make_transcript(idx, f"line {idx}") for idx in range(6)]
    layouts = [
        _make_layout(idx, (0, 10 + (idx % 2), 100, 30 + (idx % 2)), f"ls:page-1:{idx}")
        for idx in range(6)
    ]
    block = Block(
        id="block:page-1:reflow",
        page_id="page-1",
        transcript_spans=transcripts,
        layout_spans=layouts,
        source="pagexml",
    )
    plan = planner.plan("page-1", [block])
    resize_ops = [op for op in plan.operations if isinstance(op, ResizeBox)]
    assert len(resize_ops) >= len(layouts)
    original_tops = [layout.bbox[1] for layout in layouts]
    tops = [op.new_bbox[1] for op in resize_ops]
    assert tops == sorted(tops)
    assert (max(tops) - min(tops)) > (max(original_tops) - min(original_tops))


def test_reflow_skips_manual_blocks():
    cfg = PlannerConfig(
        enable_splits=False,
        reflow_min_lines=3,
        reflow_span_ratio=0.95,
    )
    planner = AlignmentPlanner(cfg, lambda _a, _b: 0.0)
    transcripts = [_make_transcript(idx, f"line {idx}") for idx in range(6)]
    layouts = [
        _make_layout(idx, (0, 5, 100, 25), f"ls:page-1:manual:{idx}")
        for idx in range(6)
    ]
    block = Block(
        id="block:page-1:manual",
        page_id="page-1",
        transcript_spans=transcripts,
        layout_spans=layouts,
        source="manual",
    )
    plan = planner.plan("page-1", [block])
    resize_ops = [op for op in plan.operations if isinstance(op, ResizeBox)]
    assert not resize_ops
