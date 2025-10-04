from __future__ import annotations

from typing import List

from src.overlay_store import OcrToken, OverlayStore


def make_token(
    text: str,
    bbox: tuple[int, int, int, int],
    order: tuple[int, int, int, int, int],
    line: tuple[int, int, int],
) -> OcrToken:
    return OcrToken(text=text, bbox=bbox, order_key=order, line_key=line)


def extract_ids(store: OverlayStore) -> List[int]:
    return [overlay.id for overlay in store.list_overlays()]


def test_set_tokens_and_selection() -> None:
    store = OverlayStore()
    tokens = [
        make_token("hello", (0, 0, 10, 10), (1, 1, 1, 1, 1), (1, 1, 1)),
        make_token("world", (12, 0, 22, 10), (1, 1, 1, 1, 2), (1, 1, 1)),
    ]

    store.set_tokens(tokens)
    overlays = store.list_overlays()
    assert [overlay.text for overlay in overlays] == ["hello", "world"]
    assert not store.selection

    store.select_click(overlays[0].id, additive=False)
    assert store.selection == (overlays[0].id,)
    assert store.list_overlays()[0].selected

    store.select_click(overlays[1].id, additive=True)
    assert set(store.selection) == {overlays[0].id, overlays[1].id}

    hits = store.ids_intersecting((5, 0, 18, 10))
    assert hits == {overlays[0].id, overlays[1].id}


def test_manual_add_remove_and_history() -> None:
    store = OverlayStore()
    store.set_tokens([])

    manual_id = store.add_manual((0, 0, 20, 20))
    assert manual_id in extract_ids(store)
    assert store.selection == (manual_id,)
    manual_overlay = store.get_overlay(manual_id)
    assert manual_overlay is not None and manual_overlay.is_manual

    removed = store.remove_by_ids([manual_id])
    assert [overlay.id for overlay in removed] == [manual_id]
    assert manual_id not in extract_ids(store)
    assert not store.selection

    assert store.undo() is True
    assert manual_id in extract_ids(store)
    assert store.selection == (manual_id,)

    assert store.redo() is True
    assert manual_id not in extract_ids(store)
    assert not store.selection


def test_compose_text_and_update_order() -> None:
    store = OverlayStore()
    tokens = [
        make_token("first", (0, 0, 10, 10), (1, 1, 1, 1, 1), (1, 1, 1)),
        make_token("third", (15, 0, 25, 10), (1, 1, 1, 1, 2), (1, 1, 1)),
        make_token("second", (0, 20, 10, 30), (1, 1, 1, 2, 1), (1, 1, 2)),
    ]
    store.set_tokens(tokens)
    overlays = store.list_overlays()
    assert [overlay.text for overlay in overlays] == ["first", "third", "second"]

    store.update_text(overlays[1].id, " third  ")
    store.update_text(overlays[2].id, "second")
    assert store.compose_text() == "first third\nsecond"


def test_redo_cleared_after_new_command() -> None:
    store = OverlayStore()
    tokens = [
        make_token("word", (0, 0, 10, 10), (1, 1, 1, 1, 1), (1, 1, 1)),
    ]
    store.set_tokens(tokens)
    overlay_id = store.list_overlays()[0].id

    store.update_text(overlay_id, "edited")
    store.undo()
    store.update_text(overlay_id, "fresh")
    assert store.redo() is False


def test_remove_returns_overlays_in_order() -> None:
    store = OverlayStore()
    tokens = [
        make_token("a", (0, 0, 5, 5), (1, 1, 1, 1, 1), (1, 1, 1)),
        make_token("b", (10, 0, 15, 5), (1, 1, 1, 1, 2), (1, 1, 1)),
    ]
    store.set_tokens(tokens)
    overlays = store.list_overlays()
    removed = store.remove_by_ids([overlays[0].id, overlays[1].id])
    assert [overlay.id for overlay in removed] == [overlays[0].id, overlays[1].id]
    assert not store.list_overlays()
    store.undo()
    assert extract_ids(store) == [overlay.id for overlay in overlays]
