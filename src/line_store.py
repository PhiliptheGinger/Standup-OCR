"""State container for baseline-based line annotations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]
TokenOrder = Tuple[int, int, int, int, int]


@dataclass
class Line:
    """Representation of a segmented text line."""

    id: int
    baseline: List[Point]
    bbox: BBox
    text: str
    order_key: TokenOrder
    selected: bool = False
    is_manual: bool = False


class Command(Protocol):
    """Protocol describing undoable commands."""

    def do(self, store: "LineStore") -> None:
        ...

    def undo(self, store: "LineStore") -> None:
        ...


class LineStore:
    """Mutable store with undo/redo support for :class:`Line` objects."""

    def __init__(self) -> None:
        self._lines: Dict[int, Line] = {}
        self._order: List[int] = []
        self._selection: Set[int] = set()
        self._next_id: int = 1
        self._order_counter: int = 0
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def set_lines(self, lines: List[Line]) -> None:
        """Replace the store content with ``lines``.

        The provided line instances are copied to avoid outside mutation.
        """

        self._lines.clear()
        self._order.clear()
        self._selection.clear()
        for line in lines:
            line_copy = replace(line, baseline=list(line.baseline), selected=line.selected)
            self._lines[line_copy.id] = line_copy
            self._order.append(line_copy.id)
            self._selection.discard(line_copy.id)
            if line_copy.selected:
                self._selection.add(line_copy.id)
        if self._order:
            self._next_id = max(self._order) + 1
        else:
            self._next_id = 1
        if self._lines:
            self._order_counter = max(line.order_key[-1] for line in self._lines.values()) + 1
        else:
            self._order_counter = 0
        self._undo_stack.clear()
        self._redo_stack.clear()

    def _compute_bbox(self, baseline: Iterable[Point]) -> BBox:
        xs: List[float] = []
        ys: List[float] = []
        for x, y in baseline:
            xs.append(x)
            ys.append(y)
        if not xs or not ys:
            raise ValueError("Baseline must contain at least one point.")
        left = int(min(xs))
        top = int(min(ys))
        right = int(max(xs)) + 1
        bottom = int(max(ys)) + 1
        if left == right:
            right += 1
        if top == bottom:
            bottom += 1
        return left, top, right, bottom

    def add_line(self, baseline: List[Point], is_manual: bool) -> int:
        if len(baseline) < 2:
            raise ValueError("Baseline must have at least two points.")
        bbox = self._compute_bbox(baseline)
        line_id = self._next_id
        self._next_id += 1
        order = (0, 0, 0, 0, self._order_counter)
        self._order_counter += 1
        line = Line(
            id=line_id,
            baseline=list(baseline),
            bbox=bbox,
            text="",
            order_key=order,
            selected=False,
            is_manual=is_manual,
        )
        self._lines[line_id] = line
        self._order.append(line_id)
        return line_id

    def update_text(self, id_: int, text: str) -> None:
        if id_ not in self._lines:
            raise KeyError(id_)
        line = self._lines[id_]
        self._lines[id_] = replace(line, text=text)

    def remove(self, ids: Set[int]) -> List[Line]:
        removed: List[Line] = []
        remaining_order: List[int] = []
        for line_id in self._order:
            if line_id in ids:
                line = self._lines.pop(line_id, None)
                if line is not None:
                    removed.append(line)
                self._selection.discard(line_id)
            else:
                remaining_order.append(line_id)
        self._order = remaining_order
        return removed

    def reinsert(self, lines: Iterable[Line]) -> None:
        for line in lines:
            if line.id in self._lines:
                continue
            self._lines[line.id] = replace(
                line,
                baseline=list(line.baseline),
                selected=False,
            )
            self._order.append(line.id)
            if line.order_key[-1] >= self._order_counter:
                self._order_counter = line.order_key[-1] + 1
            if line.id >= self._next_id:
                self._next_id = line.id + 1
        self._order.sort(key=lambda idx: self._lines[idx].order_key)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def select_only(self, ids: Set[int]) -> None:
        valid = {line_id for line_id in ids if line_id in self._lines}
        self._selection = valid
        for line in self._lines.values():
            line.selected = line.id in valid

    def toggle(self, id_: int) -> None:
        if id_ not in self._lines:
            return
        if id_ in self._selection:
            self._selection.remove(id_)
            self._lines[id_].selected = False
        else:
            self._selection.add(id_)
            self._lines[id_].selected = True

    def hit_test(self, x: float, y: float, tol: float = 3.0) -> Optional[int]:
        best_id: Optional[int] = None
        best_distance = float("inf")
        for line_id in self._order:
            line = self._lines[line_id]
            distance = _distance_to_polyline(x, y, line.baseline)
            if distance < tol and distance < best_distance:
                best_distance = distance
                best_id = line_id
        return best_id

    def bbox_intersect(self, bbox: BBox) -> Set[int]:
        left, top, right, bottom = bbox
        hits: Set[int] = set()
        for line_id in self._order:
            lb, tb, rb, bb = self._lines[line_id].bbox
            if rb < left or lb > right or bb < top or tb > bottom:
                continue
            hits.add(line_id)
        return hits

    # ------------------------------------------------------------------
    # Information access
    # ------------------------------------------------------------------
    def compose_text(self) -> str:
        ordered = sorted(self._lines.values(), key=lambda line: line.order_key)
        return "\n".join(line.text for line in ordered)

    def list(self) -> List[Line]:
        return [self._lines[idx] for idx in sorted(self._order, key=lambda i: self._lines[i].order_key)]

    def selection(self) -> Set[int]:
        return set(self._selection)

    # ------------------------------------------------------------------
    # Undo/redo integration
    # ------------------------------------------------------------------
    def do(self, cmd: Command) -> None:
        cmd.do(self)
        self._undo_stack.append(cmd)
        self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        cmd = self._undo_stack.pop()
        cmd.undo(self)
        self._redo_stack.append(cmd)
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        cmd = self._redo_stack.pop()
        cmd.do(self)
        self._undo_stack.append(cmd)
        return True


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------


@dataclass
class AddLine(Command):
    baseline: List[Point]
    is_manual: bool = True
    new_id: Optional[int] = None

    def do(self, store: LineStore) -> None:
        self.new_id = store.add_line(self.baseline, self.is_manual)

    def undo(self, store: LineStore) -> None:
        if self.new_id is None:
            return
        store.remove({self.new_id})


@dataclass
class RemoveLines(Command):
    ids: Set[int]
    stash: Optional[List[Line]] = None
    previous_selection: Optional[Set[int]] = None

    def do(self, store: LineStore) -> None:
        self.previous_selection = store.selection()
        self.stash = store.remove(self.ids)

    def undo(self, store: LineStore) -> None:
        if not self.stash:
            return
        store.reinsert(self.stash)
        if self.previous_selection is not None:
            store.select_only(self.previous_selection)


@dataclass
class UpdateText(Command):
    id_: int
    new: str
    old: Optional[str] = None

    def do(self, store: LineStore) -> None:
        if self.old is None:
            line = next((line for line in store.list() if line.id == self.id_), None)
            if line is None:
                raise KeyError(self.id_)
            self.old = line.text
        store.update_text(self.id_, self.new)

    def undo(self, store: LineStore) -> None:
        if self.old is None:
            return
        store.update_text(self.id_, self.old)


@dataclass
class SetSelection(Command):
    ids: Set[int]
    additive: bool = False
    prev: Optional[Set[int]] = None

    def do(self, store: LineStore) -> None:
        current = store.selection()
        self.prev = current
        if self.additive:
            target = set(current)
            for line_id in self.ids:
                if line_id in target:
                    target.remove(line_id)
                else:
                    target.add(line_id)
            store.select_only(target)
        else:
            store.select_only(self.ids)

    def undo(self, store: LineStore) -> None:
        if self.prev is None:
            return
        store.select_only(self.prev)


def _distance_to_polyline(x: float, y: float, points: Sequence[Point]) -> float:
    if len(points) == 0:
        return float("inf")
    best = float("inf")
    px, py = points[0]
    for idx in range(1, len(points)):
        qx, qy = points[idx]
        distance = _distance_point_segment(x, y, px, py, qx, qy)
        if distance < best:
            best = distance
        px, py = qx, qy
    return best


def _distance_point_segment(
    x: float, y: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    # Based on projection of point onto segment
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return ((x - proj_x) ** 2 + (y - proj_y) ** 2) ** 0.5
