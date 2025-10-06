"""Unit tests for :mod:`src.line_store`."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "line_store.py"
SPEC = importlib.util.spec_from_file_location("line_store", MODULE_PATH)
assert SPEC and SPEC.loader
line_store = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = line_store
SPEC.loader.exec_module(line_store)

AddLine = line_store.AddLine
Line = line_store.Line
LineStore = line_store.LineStore
Point = line_store.Point
RemoveLines = line_store.RemoveLines
SetSelection = line_store.SetSelection
TokenOrder = line_store.TokenOrder
UpdateText = line_store.UpdateText


def _line(
    id_: int,
    baseline: List[Point],
    text: str,
    order_key: TokenOrder,
    *,
    selected: bool = False,
    is_manual: bool = False,
) -> Line:
    min_x = int(min(pt[0] for pt in baseline))
    min_y = int(min(pt[1] for pt in baseline))
    max_x = int(max(pt[0] for pt in baseline)) + 1
    max_y = int(max(pt[1] for pt in baseline)) + 1
    return Line(
        id=id_,
        baseline=list(baseline),
        bbox=(min_x, min_y, max_x, max_y),
        text=text,
        order_key=order_key,
        selected=selected,
        is_manual=is_manual,
    )


def test_set_lines_and_compose_text() -> None:
    store = LineStore()
    lines = [
        _line(1, [(0, 0), (10, 0)], "alpha", (0, 0, 0, 0, 0)),
        _line(2, [(0, 10), (10, 10)], "beta", (0, 0, 0, 0, 1)),
    ]
    store.set_lines(lines)

    listed = store.list()
    assert [line.id for line in listed] == [1, 2]
    assert store.compose_text() == "alpha\nbeta"


def test_add_line_assigns_id_and_bbox() -> None:
    store = LineStore()
    line_id = store.add_line([(5.0, 5.0), (15.0, 8.0)], is_manual=True)
    listed = store.list()
    assert line_id == 1
    assert len(listed) == 1
    line = listed[0]
    assert line.id == line_id
    assert line.bbox == (5, 5, 16, 9)
    assert line.is_manual is True


def test_selection_and_toggle() -> None:
    store = LineStore()
    store.set_lines([
        _line(1, [(0, 0), (10, 0)], "alpha", (0, 0, 0, 0, 0)),
        _line(2, [(0, 10), (10, 10)], "beta", (0, 0, 0, 0, 1)),
    ])

    store.select_only({2})
    assert store.selection() == {2}
    store.toggle(2)
    assert store.selection() == set()
    store.toggle(1)
    assert store.selection() == {1}


def test_remove_and_reinsert_via_command() -> None:
    store = LineStore()
    store.set_lines([
        _line(1, [(0, 0), (10, 0)], "alpha", (0, 0, 0, 0, 0)),
        _line(2, [(0, 10), (10, 10)], "beta", (0, 0, 0, 0, 1)),
    ])
    store.select_only({1, 2})

    cmd = RemoveLines({1})
    store.do(cmd)
    assert [line.id for line in store.list()] == [2]
    store.undo()
    assert [line.id for line in store.list()] == [1, 2]
    assert store.selection() == {1, 2}
    store.redo()
    assert [line.id for line in store.list()] == [2]


def test_update_text_command_tracks_history() -> None:
    store = LineStore()
    store.set_lines([
        _line(5, [(0, 0), (10, 0)], "", (0, 0, 0, 0, 0)),
    ])

    cmd = UpdateText(5, "hello")
    store.do(cmd)
    assert store.list()[0].text == "hello"
    store.undo()
    assert store.list()[0].text == ""


def test_hit_test_and_bbox_intersection() -> None:
    store = LineStore()
    store.set_lines([
        _line(1, [(0, 0), (10, 0)], "alpha", (0, 0, 0, 0, 0)),
        _line(2, [(0, 10), (10, 10)], "beta", (0, 0, 0, 0, 1)),
    ])

    hit = store.hit_test(3, 1, tol=2.0)
    assert hit == 1
    hits = store.bbox_intersect((0, 5, 12, 12))
    assert hits == {2}


def test_add_line_command_round_trip() -> None:
    store = LineStore()
    cmd = AddLine([(0.0, 0.0), (5.0, 0.0)])
    store.do(cmd)
    assert len(store.list()) == 1
    store.undo()
    assert len(store.list()) == 0
    store.redo()
    assert len(store.list()) == 1


def test_set_selection_command_additive() -> None:
    store = LineStore()
    store.set_lines([
        _line(1, [(0, 0), (10, 0)], "alpha", (0, 0, 0, 0, 0)),
        _line(2, [(0, 10), (10, 10)], "beta", (0, 0, 0, 0, 1)),
    ])

    cmd = SetSelection({1}, additive=False)
    store.do(cmd)
    assert store.selection() == {1}
    store.undo()
    assert store.selection() == set()

    cmd_toggle = SetSelection({1, 2}, additive=True)
    store.do(cmd_toggle)
    assert store.selection() == {1, 2}
    store.undo()
    assert store.selection() == set()


def test_add_line_requires_two_points() -> None:
    store = LineStore()
    with pytest.raises(ValueError):
        store.add_line([(0.0, 0.0)], is_manual=True)
