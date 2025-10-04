"""Overlay state and undo/redo management for the annotation UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple


BBox = Tuple[int, int, int, int]
TokenOrder = Tuple[int, int, int, int, int]
LineKey = Tuple[int, int, int]
Listener = Callable[[object], None]


@dataclass(slots=True)
class OcrToken:
    """OCR token recognised for the current image."""

    text: str
    bbox: BBox
    order_key: TokenOrder
    line_key: LineKey


@dataclass(slots=True)
class Overlay:
    """Overlay displayed on top of the image."""

    id: int
    bbox_base: BBox
    text: str
    is_manual: bool
    order_key: TokenOrder
    line_key: LineKey
    selected: bool = False


class Command(Protocol):
    """Reversible mutation applied to :class:`OverlayStore`."""

    def do(self, store: "OverlayStore") -> None:
        """Apply the command."""

    def undo(self, store: "OverlayStore") -> None:
        """Revert a previously applied command."""


@dataclass
class AddOverlay(Command):
    overlays: List[Overlay]
    index: Optional[int] = None
    _positions: List[int] = field(default_factory=list, init=False)

    def do(self, store: "OverlayStore") -> None:
        positions: List[int] = []
        insert_at = self.index
        for overlay in self.overlays:
            position = store._insert_overlay(overlay, insert_at)
            positions.append(position)
            if insert_at is not None:
                insert_at += 1
        self._positions = positions
        store._emit_overlays()

    def undo(self, store: "OverlayStore") -> None:
        store._remove_overlays([overlay.id for overlay in self.overlays])
        store._emit_overlays()


@dataclass
class RemoveOverlays(Command):
    overlay_ids: Sequence[int]
    _removed: List[Tuple[int, Overlay]] = field(default_factory=list, init=False)
    _previous_selection: Set[int] = field(default_factory=set, init=False)

    def do(self, store: "OverlayStore") -> None:
        self._previous_selection = set(store.selection)
        self._removed = store._remove_overlays(self.overlay_ids)
        store._emit_overlays()

    def undo(self, store: "OverlayStore") -> None:
        for index, overlay in sorted(self._removed, key=lambda item: item[0]):
            store._insert_overlay(overlay, index)
        store._apply_selection(self._previous_selection)
        store._emit_overlays()

    @property
    def removed(self) -> List[Overlay]:
        return [overlay for _, overlay in self._removed]


@dataclass
class UpdateOverlayText(Command):
    overlay_id: int
    new_text: str
    _previous: Optional[str] = field(default=None, init=False)

    def do(self, store: "OverlayStore") -> None:
        self._previous = store._set_overlay_text(self.overlay_id, self.new_text)
        store._emit_overlays()

    def undo(self, store: "OverlayStore") -> None:
        if self._previous is None:
            return
        store._set_overlay_text(self.overlay_id, self._previous)
        store._emit_overlays()


@dataclass
class SetSelection(Command):
    selection: Sequence[int]
    _previous: Set[int] = field(default_factory=set, init=False)

    def do(self, store: "OverlayStore") -> None:
        self._previous = set(store.selection)
        store._apply_selection(set(self.selection))

    def undo(self, store: "OverlayStore") -> None:
        store._apply_selection(self._previous)


class OverlayStore:
    """Single source of truth for overlays and selection state."""

    def __init__(self) -> None:
        self._overlays: Dict[int, Overlay] = {}
        self._order: List[int] = []
        self._selection: Set[int] = set()
        self._status: Optional[str] = None
        self._focus: Optional[int] = None
        self._next_id: int = 1

        self._listeners: Dict[str, List[Listener]] = {
            "overlays": [],
            "selection": [],
            "status": [],
            "focus": [],
        }

        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []

    # ------------------------------------------------------------------
    # Listener management
    # ------------------------------------------------------------------
    def subscribe(self, event: str, callback: Listener) -> Callable[[], None]:
        if event not in self._listeners:
            raise ValueError(f"Unknown event '{event}'.")
        listeners = self._listeners[event]
        listeners.append(callback)

        def unsubscribe() -> None:
            if callback in listeners:
                listeners.remove(callback)

        return unsubscribe

    def on_overlays(self, callback: Listener) -> Callable[[], None]:
        return self.subscribe("overlays", callback)

    def on_selection(self, callback: Listener) -> Callable[[], None]:
        return self.subscribe("selection", callback)

    def on_status(self, callback: Listener) -> Callable[[], None]:
        return self.subscribe("status", callback)

    def on_focus(self, callback: Listener) -> Callable[[], None]:
        return self.subscribe("focus", callback)

    def _emit(self, event: str, payload: object) -> None:
        for callback in tuple(self._listeners.get(event, [])):
            callback(payload)

    def _emit_overlays(self) -> None:
        self._emit("overlays", self.list_overlays())

    def _emit_selection(self) -> None:
        self._emit("selection", tuple(self._selection))

    def _emit_status(self) -> None:
        self._emit("status", self._status)

    def _emit_focus(self) -> None:
        self._emit("focus", self._focus)

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._overlays.clear()
        self._order.clear()
        self._selection.clear()
        self._status = None
        self._focus = None
        self._next_id = 1
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_overlays()
        self._emit_selection()
        self._emit_status()
        self._emit_focus()

    def set_tokens(self, tokens: Sequence[OcrToken]) -> None:
        self.reset()
        for token in tokens:
            overlay = Overlay(
                id=self._allocate_id(),
                bbox_base=token.bbox,
                text=token.text,
                is_manual=False,
                order_key=token.order_key,
                line_key=token.line_key,
            )
            self._insert_overlay(overlay)
        self._emit_overlays()

    def list_overlays(self) -> List[Overlay]:
        return [self._overlays[overlay_id] for overlay_id in self._order]

    def get_overlay(self, overlay_id: int) -> Optional[Overlay]:
        return self._overlays.get(overlay_id)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def select_click(self, overlay_id: int, *, additive: bool) -> None:
        if overlay_id not in self._overlays:
            return
        if additive:
            selection = set(self._selection)
            selection.add(overlay_id)
        else:
            selection = {overlay_id}
        self._apply_selection(selection)

    def select_set(self, ids: Set[int]) -> None:
        self._apply_selection(set(ids))

    def clear_selection(self) -> None:
        if not self._selection:
            return
        self._apply_selection(set())

    def ids_intersecting(self, bbox_base: BBox) -> Set[int]:
        x1, y1, x2, y2 = bbox_base
        results: Set[int] = set()
        for overlay in self.list_overlays():
            ox1, oy1, ox2, oy2 = overlay.bbox_base
            if not (x2 < ox1 or ox2 < x1 or y2 < oy1 or oy2 < y1):
                results.add(overlay.id)
        return results

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def add_manual(self, bbox_base: BBox) -> int:
        overlay_id = self._allocate_id()
        index = len(self._order)
        overlay = Overlay(
            id=overlay_id,
            bbox_base=bbox_base,
            text="",
            is_manual=True,
            order_key=(9999, 0, 0, overlay_id, overlay_id),
            line_key=(9999, 0, overlay_id),
        )
        command = AddOverlay([overlay], index=index)
        self.do(command)
        self._apply_selection({overlay_id})
        self.request_focus(overlay_id)
        return overlay_id

    def update_text(self, overlay_id: int, text: str) -> None:
        command = UpdateOverlayText(overlay_id, text)
        self.do(command)

    def remove_by_ids(self, ids: Sequence[int]) -> List[Overlay]:
        if not ids:
            return []
        command = RemoveOverlays(tuple(ids))
        self.do(command)
        removed = command.removed
        if removed and self._selection.intersection({overlay.id for overlay in removed}):
            self._apply_selection(self._selection - {overlay.id for overlay in removed})
        return removed

    def compose_text(self) -> str:
        lines: Dict[LineKey, List[Tuple[TokenOrder, str]]] = {}
        for overlay in self.list_overlays():
            text = overlay.text.strip()
            if not text:
                continue
            lines.setdefault(overlay.line_key, []).append((overlay.order_key, text))

        if not lines:
            return ""

        ordered_lines: List[str] = []
        for line_key in sorted(lines.keys()):
            words = [
                word
                for _, word in sorted(
                    lines[line_key],
                    key=lambda item: item[0],
                )
            ]
            ordered_lines.append(" ".join(words))
        return "\n".join(ordered_lines)

    # ------------------------------------------------------------------
    # Undo/redo
    # ------------------------------------------------------------------
    def do(self, command: Command) -> None:
        command.do(self)
        self._undo_stack.append(command)
        self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        command = self._undo_stack.pop()
        command.undo(self)
        self._redo_stack.append(command)
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        command = self._redo_stack.pop()
        command.do(self)
        self._undo_stack.append(command)
        return True

    # ------------------------------------------------------------------
    # Status/focus helpers
    # ------------------------------------------------------------------
    def set_status(self, message: Optional[str]) -> None:
        if self._status == message:
            return
        self._status = message
        self._emit_status()

    def request_focus(self, overlay_id: Optional[int]) -> None:
        if self._focus == overlay_id:
            return
        self._focus = overlay_id
        self._emit_focus()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _allocate_id(self) -> int:
        next_id = self._next_id
        self._next_id += 1
        return next_id

    def _insert_overlay(self, overlay: Overlay, index: Optional[int] = None) -> int:
        self._overlays[overlay.id] = overlay
        if index is None or index >= len(self._order):
            self._order.append(overlay.id)
            index = len(self._order) - 1
        else:
            self._order.insert(index, overlay.id)
        return index

    def _remove_overlays(self, ids: Sequence[int]) -> List[Tuple[int, Overlay]]:
        removed: List[Tuple[int, Overlay]] = []
        id_set = set(ids)
        new_order: List[int] = []
        for position, overlay_id in enumerate(self._order):
            if overlay_id in id_set:
                overlay = self._overlays.pop(overlay_id, None)
                if overlay is not None:
                    removed.append((position, overlay))
            else:
                new_order.append(overlay_id)
        self._order = new_order
        self._apply_selection(self._selection - id_set)
        return removed

    def _set_overlay_text(self, overlay_id: int, text: str) -> str:
        overlay = self._overlays.get(overlay_id)
        if overlay is None:
            raise KeyError(f"Overlay {overlay_id} not found")
        previous = overlay.text
        if previous == text:
            return previous
        overlay.text = text
        return previous

    def _apply_selection(self, selection: Set[int]) -> None:
        if selection == self._selection:
            return
        self._selection = {overlay_id for overlay_id in selection if overlay_id in self._overlays}
        for overlay in self._overlays.values():
            overlay.selected = overlay.id in self._selection
        self._emit_selection()
        self._emit_overlays()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def selection(self) -> Tuple[int, ...]:
        return tuple(self._selection)

    @property
    def status(self) -> Optional[str]:
        return self._status

    @property
    def focus_target(self) -> Optional[int]:
        return self._focus

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)


__all__ = [
    "BBox",
    "LineKey",
    "TokenOrder",
    "OcrToken",
    "Overlay",
    "OverlayStore",
    "Command",
    "AddOverlay",
    "RemoveOverlays",
    "UpdateOverlayText",
    "SetSelection",
    "UpdateOverlayBounds",
]
