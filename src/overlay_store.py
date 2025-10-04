"""Overlay state management for Standup-OCR UI components.

This module intentionally avoids any GUI/Tk specific imports so that it can be
unit tested in isolation and consumed by different front-ends.  It provides a
lightweight data model for OCR tokens/overlays as well as an observable store
with undo/redo support.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple


Bounds = Tuple[int, int, int, int]
Listener = Callable[[Any], None]


@dataclass(frozen=True)
class OcrToken:
    """Represents a single OCR token.

    Attributes
    ----------
    token_id:
        Identifier for the token as reported by the OCR engine.
    text:
        Raw text content recognised for the token.
    bounds:
        Bounding box for the token expressed as ``(x, y, width, height)`` in
        image coordinates.
    confidence:
        Optional confidence score produced by the OCR engine.  ``None`` is used
        when a score is not available.
    metadata:
        Free-form metadata describing token level information (e.g. language,
        font).  Consumers should treat the mapping as read-only.
    """

    token_id: str
    text: str
    bounds: Bounds
    confidence: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Overlay:
    """Metadata describing an annotation overlay rendered on top of the image."""

    overlay_id: str
    tokens: Tuple[OcrToken, ...]
    bounds: Bounds
    text: str = ""
    label: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_updated_text(self, text: str) -> "Overlay":
        """Return a copy of the overlay with updated text content."""

        return replace(self, text=text)


class Command(Protocol):
    """A reversible mutation on the :class:`OverlayStore`."""

    def apply(self, store: "OverlayStore") -> "Command":
        """Execute the command and return its inverse."""


@dataclass
class AddOverlay:
    """Command that adds one or more overlays to the store."""

    overlays: Tuple[Overlay, ...]

    def apply(self, store: "OverlayStore") -> "Command":
        for overlay in self.overlays:
            store._add_overlay_direct(overlay)
        return RemoveOverlays(tuple(o.overlay_id for o in self.overlays), overlays=self.overlays)


@dataclass
class RemoveOverlays:
    """Command that removes overlays by identifier."""

    overlay_ids: Tuple[str, ...]
    overlays: Optional[Tuple[Overlay, ...]] = None

    def apply(self, store: "OverlayStore") -> "Command":
        removed_overlays = store._remove_overlays_direct(self.overlay_ids)
        overlays_to_restore: Tuple[Overlay, ...]
        if self.overlays is None:
            overlays_to_restore = removed_overlays
        else:
            overlays_to_restore = self.overlays
        return AddOverlay(overlays_to_restore)


@dataclass
class UpdateOverlayText:
    """Command updating the text associated with an overlay."""

    overlay_id: str
    new_text: str

    def apply(self, store: "OverlayStore") -> "Command":
        previous_text = store._update_overlay_text_direct(self.overlay_id, self.new_text)
        return UpdateOverlayText(self.overlay_id, previous_text)


@dataclass
class SetSelection:
    """Command setting the current overlay selection."""

    selection: Tuple[str, ...]

    def apply(self, store: "OverlayStore") -> "Command":
        previous_selection = store._set_selection_direct(self.selection)
        return SetSelection(previous_selection)


class OverlayStore:
    """In-memory store for overlays with undo/redo semantics."""

    def __init__(self) -> None:
        self._overlays: Dict[str, Overlay] = {}
        self._selection: Tuple[str, ...] = tuple()
        self._status_message: Optional[str] = None
        self._focus_target: Optional[str] = None

        self._listeners: Dict[str, List[Listener]] = {
            "overlays": [],
            "selection": [],
            "status": [],
            "focus": [],
        }

        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []

    # ------------------------------------------------------------------
    # Listener registration & event dispatch
    # ------------------------------------------------------------------
    def subscribe(self, event: str, callback: Listener) -> Callable[[], None]:
        """Subscribe to an event emitted by the store.

        Parameters
        ----------
        event:
            Name of the event to subscribe to.  Valid events are ``"overlays"``,
            ``"selection"``, ``"status"`` and ``"focus"``.
        callback:
            Callable invoked with the current value when the event is emitted.

        Returns
        -------
        Callable[[], None]
            A function that removes the listener when called.
        """

        if event not in self._listeners:
            raise ValueError(f"Unknown event '{event}'.")
        listeners = self._listeners[event]
        listeners.append(callback)

        def unsubscribe() -> None:
            if callback in listeners:
                listeners.remove(callback)

        return unsubscribe

    def on_selection_changed(self, callback: Listener) -> Callable[[], None]:
        return self.subscribe("selection", callback)

    def on_status_changed(self, callback: Listener) -> Callable[[], None]:
        return self.subscribe("status", callback)

    def on_focus_requested(self, callback: Listener) -> Callable[[], None]:
        return self.subscribe("focus", callback)

    def on_overlays_changed(self, callback: Listener) -> Callable[[], None]:
        return self.subscribe("overlays", callback)

    def _emit(self, event: str, value: Any) -> None:
        for callback in tuple(self._listeners.get(event, [])):
            callback(value)

    # ------------------------------------------------------------------
    # Undo/redo orchestration
    # ------------------------------------------------------------------
    def _execute_user_command(self, command: Command) -> None:
        inverse = command.apply(self)
        self._undo_stack.append(inverse)
        self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        command = self._undo_stack.pop()
        inverse = command.apply(self)
        self._redo_stack.append(inverse)
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        command = self._redo_stack.pop()
        inverse = command.apply(self)
        self._undo_stack.append(inverse)
        return True

    # ------------------------------------------------------------------
    # Direct mutation helpers (do not push history)
    # ------------------------------------------------------------------
    def _add_overlay_direct(self, overlay: Overlay) -> None:
        self._overlays[overlay.overlay_id] = overlay
        self._emit("overlays", self.get_overlays())

    def _remove_overlays_direct(self, overlay_ids: Iterable[str]) -> Tuple[Overlay, ...]:
        removed: List[Overlay] = []
        for overlay_id in overlay_ids:
            overlay = self._overlays.pop(overlay_id, None)
            if overlay is not None:
                removed.append(overlay)
        if removed:
            self._emit("overlays", self.get_overlays())
        return tuple(removed)

    def _update_overlay_text_direct(self, overlay_id: str, new_text: str) -> str:
        overlay = self._overlays.get(overlay_id)
        if overlay is None:
            raise KeyError(f"Overlay '{overlay_id}' not found.")
        previous_text = overlay.text
        updated_overlay = overlay.with_updated_text(new_text)
        self._overlays[overlay_id] = updated_overlay
        self._emit("overlays", self.get_overlays())
        return previous_text

    def _set_selection_direct(self, selection: Iterable[str]) -> Tuple[str, ...]:
        previous_selection = self._selection
        normalized = tuple(dict.fromkeys(selection))
        if normalized != self._selection:
            self._selection = normalized
            self._emit("selection", self._selection)
        return previous_selection

    def _set_status_direct(self, message: Optional[str]) -> Optional[str]:
        previous = self._status_message
        if previous != message:
            self._status_message = message
            self._emit("status", self._status_message)
        return previous

    def _set_focus_direct(self, overlay_id: Optional[str]) -> Optional[str]:
        previous = self._focus_target
        if previous != overlay_id:
            self._focus_target = overlay_id
            self._emit("focus", self._focus_target)
        return previous

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_overlay(self, overlay: Overlay) -> None:
        self._execute_user_command(AddOverlay((overlay,)))

    def add_overlays(self, overlays: Sequence[Overlay]) -> None:
        if not overlays:
            return
        self._execute_user_command(AddOverlay(tuple(overlays)))

    def remove_overlays(self, overlay_ids: Iterable[str]) -> None:
        overlay_ids_tuple = tuple(overlay_ids)
        if not overlay_ids_tuple:
            return
        self._execute_user_command(RemoveOverlays(overlay_ids_tuple))

    def update_overlay_text(self, overlay_id: str, new_text: str) -> None:
        self._execute_user_command(UpdateOverlayText(overlay_id, new_text))

    def set_selection(self, overlay_ids: Iterable[str]) -> None:
        self._execute_user_command(SetSelection(tuple(overlay_ids)))

    def clear_selection(self) -> None:
        self.set_selection(())

    def set_status(self, message: Optional[str]) -> None:
        self._set_status_direct(message)

    def request_focus(self, overlay_id: Optional[str]) -> None:
        self._set_focus_direct(overlay_id)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_overlay(self, overlay_id: str) -> Optional[Overlay]:
        return self._overlays.get(overlay_id)

    def get_overlays(self) -> Tuple[Overlay, ...]:
        return tuple(self._overlays.values())

    def iter_overlays(self) -> Iterator[Overlay]:
        return iter(self._overlays.values())

    @property
    def selection(self) -> Tuple[str, ...]:
        return self._selection

    @property
    def status(self) -> Optional[str]:
        return self._status_message

    @property
    def focus_target(self) -> Optional[str]:
        return self._focus_target

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)


__all__ = [
    "Bounds",
    "Listener",
    "OcrToken",
    "Overlay",
    "OverlayStore",
    "AddOverlay",
    "RemoveOverlays",
    "UpdateOverlayText",
    "SetSelection",
]
