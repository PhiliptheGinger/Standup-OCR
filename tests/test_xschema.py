import logging

import pytest

from xschema import (
    AlignmentPlanner,
    Block,
    LayoutSpan,
    MoveGroup,
    PlannerConfig,
    PlannerTelemetry,
    ResizeBox,
    TranscriptSpan,
)


def _make_transcript(page: str, idx: int, text: str) -> TranscriptSpan:
    return TranscriptSpan(
        id=f"ts:{page}:{idx}",
        page_id=page,
        text=text,
        line_index=idx,
        char_start=idx * 10,
        char_end=(idx * 10) + len(text),
        words=text.split() or None,
    )


def _make_layout(
    page: str,
    ident: str,
    top: int,
    bottom: int,
    *,
    text: str = "",
    manual: bool = False,
    order: int = 0,
) -> LayoutSpan:
    return LayoutSpan(
        id=f"ls:{page}:{ident}",
        page_id=page,
        bbox=(0, top, 50, bottom),
        baseline=[(0, top), (50, top)],
        order_key=(0, 0, 0, order, 0),
        text_hint=text,
        is_manual=manual,
    )


def test_planner_telemetry_logs_merge(caplog: pytest.LogCaptureFixture) -> None:
    page_id = "page"
    transcripts = [_make_transcript(page_id, 0, "alpha")]
    layouts = [
        _make_layout(page_id, "base", 0, 40, text="alpha", order=1),
        _make_layout(page_id, "extra", 5, 45, text="", order=2),
    ]
    block = Block(
        id="block:merge",
        page_id=page_id,
        transcript_spans=transcripts,
        layout_spans=layouts,
        source="manual",
    )
    config = PlannerConfig(merge_gap_ratio=5.0, merge_gain_threshold=0.0, split_height_threshold=10)
    planner = AlignmentPlanner(config, lambda a, b: 1.0 if a == b else 0.0)
    telemetry = PlannerTelemetry()

    with caplog.at_level(logging.DEBUG):
        plan = planner.plan(page_id, [block], telemetry=telemetry)

    assert plan.telemetry is telemetry
    assert telemetry.merges == 1
    assert telemetry.assignments == 1
    assert telemetry.links == 1
    assert f"merges={telemetry.merges}" in caplog.text


def test_planner_telemetry_tracks_all_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    page_id = "page"
    config = PlannerConfig(
        split_height_threshold=5,
        merge_gap_ratio=5.0,
        merge_gain_threshold=0.0,
        block_move_threshold=1,
        block_resize_slack=1,
    )
    planner = AlignmentPlanner(config, lambda a, b: 1.0 if a == b else 0.0)

    split_block = Block(
        id="block:split",
        page_id=page_id,
        transcript_spans=[
            _make_transcript(page_id, 0, "foo"),
            _make_transcript(page_id, 1, "bar"),
        ],
        layout_spans=[_make_layout(page_id, "split", 0, 200, text="foo", order=1)],
        source="manual",
    )

    create_block = Block(
        id="block:create",
        page_id=page_id,
        transcript_spans=[_make_transcript(page_id, 0, "lonely")],
        layout_spans=[],
        source="manual",
    )

    hide_block = Block(
        id="block:hide",
        page_id=page_id,
        transcript_spans=[],
        layout_spans=[_make_layout(page_id, "ghost", 0, 30, text="", order=1)],
        source="manual",
    )

    move_block = Block(
        id="block:move",
        page_id=page_id,
        transcript_spans=[_make_transcript(page_id, 0, "move")],
        layout_spans=[_make_layout(page_id, "move", 100, 140, text="move", order=1)],
        source="manual",
    )

    resize_block = Block(
        id="block:resize",
        page_id=page_id,
        transcript_spans=[],
        layout_spans=[_make_layout(page_id, "resize", 0, 120, text="", order=1)],
        source="manual",
    )

    def fake_move(self, block, links, min_center, span):
        if block.id != "block:move" or not links:
            return None
        layout_ids = [layout.id for link in links for layout in link.layouts]
        return MoveGroup(layout_ids=layout_ids, dx=0, dy=3)

    def fake_resize(self, block):
        if block.id != "block:resize" or not block.layout_spans:
            return []
        span = block.layout_spans[0]
        return [ResizeBox(layout_id=span.id, new_bbox=span.bbox)]

    monkeypatch.setattr(AlignmentPlanner, "_plan_move_group", fake_move, raising=False)
    monkeypatch.setattr(AlignmentPlanner, "_plan_resize_operations", fake_resize, raising=False)

    telemetry = PlannerTelemetry()
    plan = planner.plan(
        page_id,
        [split_block, create_block, hide_block, move_block, resize_block],
        telemetry=telemetry,
    )

    assert plan.telemetry is telemetry
    assert telemetry.splits == 1
    assert telemetry.created == 1
    assert telemetry.hidden >= 1
    assert telemetry.move_groups == 1
    assert telemetry.resize_boxes == 1
    assert telemetry.assignments >= 2