"""Tests for the annotation helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import annotation
from annotation import AnnotationApp, AnnotationItem, _prepare_image


def test_prepare_image_applies_exif_orientation():
    """Images should be rotated according to their EXIF orientation."""

    image = Image.new("RGB", (10, 20), "red")
    exif = image.getexif()
    exif[274] = 6  # Orientation tag: rotate 270 degrees
    image.info["exif"] = exif.tobytes()

    prepared = _prepare_image(image)

    assert prepared.size == (20, 10)


def test_slugify_truncates_and_collapses_hyphens():
    """The slug should be limited in length and collapse repeated separators."""

    value = "abcde " * 20  # produces a string well over 60 characters once slugified
    expected = "-".join(["abcde"] * 20)
    app = AnnotationApp.__new__(AnnotationApp)

    slug = AnnotationApp._slugify(app, value)

    assert slug == expected[:60].rstrip("-")
    assert len(slug) <= 60
    assert "--" not in slug


def test_confirm_handles_oserror(monkeypatch):
    """Failures during saving should surface to the user without advancing."""

    app = AnnotationApp.__new__(AnnotationApp)
    app.items = [AnnotationItem(Path("image.png"))]
    app.index = 0
    app._get_transcription_text = lambda: "transcription"
    app.overlay_entries = []
    app.overlay_items = []
    app.rect_to_overlay = {}
    app.selected_rects = set()
    app.current_tokens = []

    def failing_save(*_args, **_kwargs):
        raise OSError("disk full")

    app._save_annotation = failing_save

    append_called = False

    def append_log(*_args, **_kwargs):
        nonlocal append_called
        append_called = True

    app._append_log = append_log

    advanced = False

    def advance():
        nonlocal advanced
        advanced = True

    app._advance = advance

    class DummyVar:
        def __init__(self):
            self.value = None

        def set(self, value):
            self.value = value

    app.status_var = DummyVar()

    errors: list[tuple[str, str]] = []

    def fake_showerror(title, message):
        errors.append((title, message))

    monkeypatch.setattr(annotation.messagebox, "showerror", fake_showerror)

    AnnotationApp.confirm(app)

    assert errors and "disk full" in errors[0][1]
    assert not append_called
    assert not advanced
    assert app.status_var.value is None


def test_back_rewinds_without_reappending_logs():
    """The back button should revisit the previous item without side effects."""

    app = AnnotationApp.__new__(AnnotationApp)
    app.items = [
        AnnotationItem(Path("first.png")),
        AnnotationItem(Path("second.png")),
    ]
    app.index = 1
    app._user_modified_transcription = False

    class DummyVar:
        def __init__(self, value: str = "") -> None:
            self.value = value

        def set(self, value: str) -> None:
            self.value = value

        def get(self) -> str:
            return self.value

    class DummyEntry:
        def focus_set(self) -> None:
            pass

        def delete(self, *_args, **_kwargs) -> None:  # pragma: no cover - unused here
            pass

        def insert(self, *_args, **_kwargs) -> None:  # pragma: no cover - unused here
            pass

        def get(self, *_args, **_kwargs) -> str:
            return ""

    class DummyButton:
        def __init__(self) -> None:
            self.state: Optional[str] = None

        def config(self, **kwargs) -> None:
            if "state" in kwargs:
                self.state = kwargs["state"]

    app.filename_var = DummyVar()
    app.status_var = DummyVar()
    app.entry_widget = DummyEntry()
    app.back_button = DummyButton()
    app._set_transcription = lambda value: None

    displayed_paths: list[Path] = []

    def fake_display_item(item: AnnotationItem) -> None:
        displayed_paths.append(item.path)
        app.status_var.set("Pre-filled transcription using OCR result.")

    app._display_item = fake_display_item

    app.back()

    assert app.index == 0
    assert displayed_paths == [Path("first.png")]
    assert app.back_button.state == annotation.tk.DISABLED
    status = app.status_var.get()
    assert "Returned to previous item" in status
    assert status.startswith("Pre-filled transcription using OCR result.")

    displayed_paths.clear()
    app.back_button.state = None

    app.back()

    assert app.index == 0
    assert displayed_paths == []
    assert app.back_button.state == annotation.tk.DISABLED
    assert app.status_var.get() == "Already at the first item."


def test_apply_transcription_clears_overlay_entries():
    app = AnnotationApp.__new__(AnnotationApp)

    class DummyEntry:
        def __init__(self, value: str) -> None:
            self.value = value

        def delete(self, *_args, **_kwargs) -> None:
            self.value = ""

        def insert(self, _index: int, value: str) -> None:  # pragma: no cover - not used
            self.value = value

    class DummyText:
        def __init__(self, value: str) -> None:
            self.value = value

        def get(self, *_args, **_kwargs) -> str:
            return self.value

        def delete(self, *_args, **_kwargs) -> None:
            self.value = ""

        def insert(self, *_args, **_kwargs) -> None:
            if len(_args) >= 2:
                self.value = _args[1]
            elif "chars" in _kwargs:
                self.value = _kwargs["chars"]

    entries = [DummyEntry("first"), DummyEntry("second")]
    app.overlay_entries = entries
    app.overlay_items = []
    app.rect_to_overlay = {}
    app.selected_rects = set()
    app.current_tokens = [
        annotation.OcrToken("first", (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0)),
        annotation.OcrToken("second", (0, 0, 0, 0), (0, 0, 0, 0, 1), (0, 0, 0)),
    ]
    app.entry_widget = DummyText("")
    app._setting_transcription = False
    app._user_modified_transcription = False

    AnnotationApp._apply_transcription_to_overlays(app)

    assert all(entry.value == "" for entry in entries)


def test_confirm_revisit_uses_persisted_metadata(tmp_path):
    app = AnnotationApp.__new__(AnnotationApp)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    item = AnnotationItem(image_path)
    app.items = [item]
    app.index = 0
    app.train_dir = tmp_path
    app.log_path = None
    app.overlay_entries = []
    app.overlay_items = []
    app.rect_to_overlay = {}
    app.selected_rects = set()
    app.current_tokens = []

    class DummyVar:
        def __init__(self, value: str = "") -> None:
            self.value = value

        def set(self, value: str) -> None:
            self.value = value

        def get(self) -> str:
            return self.value

    class DummyEntry:
        def __init__(self, text: str = "") -> None:
            self.text = text

        def focus_set(self) -> None:
            pass

        def delete(self, *_args, **_kwargs) -> None:
            self.text = ""

        def insert(self, *_args, **_kwargs) -> None:
            self.text = _args[1] if len(_args) > 1 else ""

        def get(self, *_args, **_kwargs) -> str:
            return self.text

    class DummyButton:
        def __init__(self) -> None:
            self.state = None

        def config(self, **kwargs) -> None:
            if "state" in kwargs:
                self.state = kwargs["state"]

    app.filename_var = DummyVar()
    app.status_var = DummyVar()
    app.entry_widget = DummyEntry("confirmed text")
    app.back_button = DummyButton()
    app._display_image = lambda *_args, **_kwargs: None
    app._clear_overlay_entries = lambda: None

    extract_called = False

    def fake_extract(_image):
        nonlocal extract_called
        extract_called = True
        return []

    app._extract_tokens = fake_extract
    app._compose_text_from_tokens = lambda _tokens: ""
    app._suggest_label = lambda _path: ""

    saved_destination = tmp_path / "train" / "saved.png"
    app._save_annotation = lambda *_args, **_kwargs: saved_destination
    app._append_log = lambda *_args, **_kwargs: None
    app._advance = lambda: None

    AnnotationApp.confirm(app)

    assert item.label == "confirmed text"
    assert item.status == "confirmed"
    assert item.saved_path == saved_destination

    app.entry_widget.text = ""

    AnnotationApp._show_current(app, revisit=True)

    assert AnnotationApp._get_transcription_text(app) == "confirmed text"
    status_message = app.status_var.get()
    assert "Previously saved to saved.png" in status_message
    assert not extract_called


def test_draw_select_delete_interactions(monkeypatch):
    """Drawing and selecting overlays should update the tracked structures."""

    class FakeEntry:
        def __init__(self, *_args, **_kwargs):
            self.text = ""
            self.focused = False
            self.destroyed = False
            self.bindings: dict[str, object] = {}

        def insert(self, _index, value):
            self.text = f"{value}{self.text}"

        def bind(self, sequence, callback):
            self.bindings[sequence] = callback

        def get(self):
            return self.text

        def focus_set(self):
            self.focused = True

        def destroy(self):
            self.destroyed = True

        def delete(self, *_args, **_kwargs):  # pragma: no cover - kept for interface parity
            self.text = ""

    class FakeCanvas:
        def __init__(self):
            self.next_id = 1
            self.rectangles: dict[int, list[float]] = {}
            self.windows: dict[int, list[float]] = {}
            self.focus_calls = 0

        def create_rectangle(self, x1, y1, x2, y2, **_kwargs):
            item_id = self.next_id
            self.next_id += 1
            self.rectangles[item_id] = [x1, y1, x2, y2]
            return item_id

        def create_window(self, x, y, **_kwargs):
            item_id = self.next_id
            self.next_id += 1
            self.windows[item_id] = [x, y]
            return item_id

        def coords(self, item_id, *coords):
            if coords:
                if item_id in self.rectangles:
                    self.rectangles[item_id] = list(coords)
                elif item_id in self.windows:
                    self.windows[item_id] = list(coords)
            else:
                if item_id in self.rectangles:
                    return self.rectangles[item_id]
                if item_id in self.windows:
                    return self.windows[item_id]
                return []

        def delete(self, item_id):
            if item_id == "all":
                self.rectangles.clear()
                self.windows.clear()
                return
            if isinstance(item_id, str):
                return
            self.rectangles.pop(item_id, None)
            self.windows.pop(item_id, None)

        def tag_raise(self, _item):
            pass

        def itemconfigure(self, _item, **_kwargs):
            pass

        def focus_set(self):
            self.focus_calls += 1

        def config(self, **_kwargs):  # pragma: no cover - not used in this test
            pass

    class DummyButton:
        def __init__(self):
            self.state = annotation.tk.DISABLED

        def config(self, **kwargs):
            if "state" in kwargs:
                self.state = kwargs["state"]

    class FakeEvent:
        def __init__(self, x, y, state=0):
            self.x = x
            self.y = y
            self.state = state

    monkeypatch.setattr(annotation.tk, "Entry", FakeEntry)

    app = AnnotationApp.__new__(AnnotationApp)
    app.canvas = FakeCanvas()
    app.overlay_items = []
    app.overlay_entries = []
    app.rect_to_overlay = {}
    app.selected_rects = set()
    app.current_tokens = []
    app.display_scale = (1.0, 1.0)
    app.manual_token_counter = 0
    app.mode_var = SimpleNamespace(get=lambda: "select")
    app.delete_button = DummyButton()
    updates: list[str] = []
    app._update_combined_transcription = lambda: updates.append("update")

    token_one = annotation.OcrToken(
        text="one",
        bbox=(0, 0, 10, 10),
        order_key=(0, 0, 0, 0, 0),
        line_key=(0, 0, 0),
    )
    token_two = annotation.OcrToken(
        text="two",
        bbox=(30, 0, 50, 10),
        order_key=(0, 0, 0, 0, 1),
        line_key=(0, 0, 0),
    )

    overlay_one = app._create_overlay_widget(token_one, 0, 0, 20, 20, preset_text="one", is_manual=False)
    overlay_two = app._create_overlay_widget(token_two, 40, 0, 70, 20, preset_text="two", is_manual=False)

    app._on_canvas_button_press(FakeEvent(5, 5))
    app._on_canvas_release(FakeEvent(5, 5))
    assert app.selected_rects == {overlay_one.rect_id}
    assert overlay_one.entry.focused
    assert app.delete_button.state == annotation.tk.NORMAL

    app._on_canvas_button_press(FakeEvent(60, 5, annotation.CONTROL_MASK))
    app._on_canvas_release(FakeEvent(60, 5, annotation.CONTROL_MASK))
    assert app.selected_rects == {overlay_one.rect_id, overlay_two.rect_id}

    app._clear_selection()
    app._on_canvas_button_press(FakeEvent(0, 0, annotation.SHIFT_MASK))
    app._on_canvas_drag(FakeEvent(80, 40, annotation.SHIFT_MASK))
    app._on_canvas_release(FakeEvent(80, 40, annotation.SHIFT_MASK))
    assert app.selected_rects == {overlay_one.rect_id, overlay_two.rect_id}

    app.mode_var = SimpleNamespace(get=lambda: "draw")
    app._on_canvas_button_press(FakeEvent(100, 100))
    app._on_canvas_drag(FakeEvent(140, 140))
    app._on_canvas_release(FakeEvent(140, 140))

    manual_overlay = app.overlay_items[-1]
    assert manual_overlay.is_manual
    assert manual_overlay.rect_id in app.selected_rects
    assert manual_overlay.entry.focused
    assert updates == ["update"]

    overlay_one.entry.text = "first"
    overlay_two.entry.text = "second"
    manual_overlay.entry.text = "manual"

    combined = app._compose_transcription()
    assert combined.splitlines() == ["first second", "manual"]

    app._delete_selected()
    assert manual_overlay.rect_id not in app.rect_to_overlay
    assert len(app.overlay_items) == 2
    assert len(app.current_tokens) == 2
    assert updates == ["update", "update"]
    assert manual_overlay.entry.destroyed
    assert app.delete_button.state == annotation.tk.DISABLED

    assert app._compose_transcription() == "first second"
