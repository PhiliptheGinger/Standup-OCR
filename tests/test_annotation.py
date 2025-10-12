"""Tests for the annotation helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

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


def test_confirm_updates_transcript_file(tmp_path):
    """Confirming an annotation should persist the transcription to disk."""

    source_path = tmp_path / "scan1.png"
    app = AnnotationApp.__new__(AnnotationApp)
    app.items = [AnnotationItem(source_path)]
    app.index = 0
    app.overlay_entries = []
    app.overlay_items = []
    app.rect_to_overlay = {}
    app.selected_rects = set()
    app.current_tokens = []
    app.transcripts_dir = tmp_path / "transcripts"
    app.transcripts_dir.mkdir()
    app._get_transcription_text = lambda: "Updated transcription"
    app._append_log = lambda *_args, **_kwargs: None
    app._advance = lambda: None
    app._on_sample_saved = None
    app.train_dir = tmp_path / "train"
    app.train_dir.mkdir()

    saved_path = app.train_dir / "saved.png"
    app._save_annotation = lambda *_args, **_kwargs: saved_path

    class DummyVar:
        def __init__(self) -> None:
            self.value: Optional[str] = None

        def set(self, value: str) -> None:
            self.value = value

    app.status_var = DummyVar()

    AnnotationApp.confirm(app)

    transcript_path = app.transcripts_dir / "scan1.txt"
    assert transcript_path.exists()
    assert transcript_path.read_text(encoding="utf8") == "Updated transcription"


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


@pytest.mark.parametrize(
    "coords, expected",
    [
        ((10.0, 20.0, 60.0, 70.0), (10, 20, 60, 70)),
        ((60.0, 20.0, 10.0, 70.0), (10, 20, 60, 70)),
        ((10.0, 70.0, 60.0, 20.0), (10, 20, 60, 70)),
        ((60.0, 70.0, 10.0, 20.0), (10, 20, 60, 70)),
    ],
)
def test_finish_manual_overlay_normalizes_coordinates(coords, expected):
    """Manual overlays should normalize drag direction before exporting."""

    app = AnnotationApp.__new__(AnnotationApp)
    app.display_scale = (1.0, 1.0)
    app.manual_token_counter = 0
    app.overlay_items = []
    app.rect_to_overlay = {}
    app.selected_rects = set()

    class DummyEntry:
        def focus_set(self) -> None:
            pass

    class DummyOverlay:
        def __init__(self) -> None:
            self.entry = DummyEntry()

    captured_bboxes = []

    def fake_create_overlay(text, bbox, order_key, token, *, is_manual, select):
        captured_bboxes.append(bbox)
        return DummyOverlay()

    app._create_overlay = fake_create_overlay  # type: ignore[assignment]
    app._update_transcription_from_overlays = lambda: None
    app._push_undo = lambda _callback: None

    class DummyVar:
        def __init__(self) -> None:
            self.value: Optional[str] = None

        def set(self, value: str) -> None:
            self.value = value

    app.status_var = DummyVar()

    class DummyCanvas:
        def __init__(self, coords):
            self._coords = coords
            self.deleted = None

        def coords(self, rect_id):
            assert rect_id == 1
            return self._coords

        def delete(self, rect_id):
            self.deleted = rect_id

    canvas = DummyCanvas(coords)
    app.canvas = canvas  # type: ignore[assignment]
    app._active_temp_rect = 1
    app._drag_start = (coords[0], coords[1])

    app._finish_manual_overlay(coords[2], coords[3])

    assert canvas.deleted == 1
    assert app._active_temp_rect is None
    assert captured_bboxes == [expected]
    assert app.manual_token_counter == 1


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
        annotation.OcrToken(
            "first",
            (0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        ),
        annotation.OcrToken(
            "second",
            (0, 0, 0, 0),
            (0, 0, 0, 0, 1),
            (0, 0, 0),
            (0, 0, 0),
        ),
    ]
    app.entry_widget = DummyText("")
    app._setting_transcription = False
    app._user_modified_transcription = False

    AnnotationApp._apply_transcription_to_overlays(app)

    assert all(entry.value == "" for entry in entries)


def test_apply_transcription_skips_extra_blank_lines():
    app = AnnotationApp.__new__(AnnotationApp)

    class DummyText:
        def __init__(self, value: str) -> None:
            self.value = value

        def get(self, *_args: object, **_kwargs: object) -> str:
            return self.value

    class DummyEntry:
        def __init__(self) -> None:
            self.value = ""

        def delete(self, *_args: object, **_kwargs: object) -> None:
            self.value = ""

        def insert(self, _index: object, text: str) -> None:
            self.value = text

    entries = [DummyEntry(), DummyEntry()]
    app.overlay_entries = entries
    app.overlay_items = [
        SimpleNamespace(
            entry=entries[0],
            bbox=(0, 0, 0, 0),
            order_key=(1, 1, 1, 1, 1),
            token=annotation.OcrToken("", (0, 0, 0, 0), (1, 1, 1, 1, 1), (0, 0, 0), (0, 0, 0)),
            is_manual=False,
        ),
        SimpleNamespace(
            entry=entries[1],
            bbox=(0, 0, 0, 0),
            order_key=(1, 1, 1, 2, 1),
            token=annotation.OcrToken("", (0, 0, 0, 0), (1, 1, 1, 2, 1), (0, 0, 0), (0, 0, 0)),
            is_manual=False,
        ),
    ]
    app.rect_to_overlay = {}
    app.selected_rects = set()
    app.current_tokens = []
    app.entry_widget = DummyText("first\n\nsecond")
    app._setting_transcription = False
    app._user_modified_transcription = False

    AnnotationApp._apply_transcription_to_overlays(app)

    assert [entry.value for entry in entries] == ["first", "second"]
    assert [token.text for token in app.current_tokens] == ["first", "second"]

def test_compose_text_from_tokens_groups_lines_and_paragraphs() -> None:
    app = AnnotationApp.__new__(AnnotationApp)
    tokens = [
        annotation.OcrToken(
            "hello",
            (0, 0, 0, 0),
            (1, 1, 1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        ),
        annotation.OcrToken(
            "there",
            (0, 0, 0, 0),
            (1, 1, 1, 1, 2),
            (1, 1, 1),
            (1, 1, 1),
        ),
        annotation.OcrToken(
            "general",
            (0, 0, 0, 0),
            (1, 1, 1, 2, 1),
            (1, 1, 1),
            (1, 1, 2),
        ),
        annotation.OcrToken(
            "kenobi",
            (0, 0, 0, 0),
            (1, 1, 1, 2, 2),
            (1, 1, 1),
            (1, 1, 2),
        ),
        annotation.OcrToken(
            "Another",
            (0, 0, 0, 0),
            (1, 1, 2, 1, 1),
            (1, 1, 2),
            (1, 1, 3),
        ),
        annotation.OcrToken(
            "paragraph",
            (0, 0, 0, 0),
            (1, 1, 2, 1, 2),
            (1, 1, 2),
            (1, 1, 3),
        ),
    ]

    assert (
        AnnotationApp._compose_text_from_tokens(app, tokens)
        == "hello there\ngeneral kenobi\n\nAnother paragraph"
    )


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


def test_confirm_triggers_background_training(monkeypatch, tmp_path):
    master_calls: list[tuple[int, object]] = []

    class StubMaster:
        def after(self, delay: int, callback):
            master_calls.append((delay, callback))
            callback()

    class ImmediateThread:
        def __init__(self, target, daemon: bool = False):
            self._target = target
            self.daemon = daemon

        def start(self) -> None:
            self._target()

    saved_dir = tmp_path / "train"
    saved_dir.mkdir()
    saved_one = saved_dir / "sample-one.png"
    saved_two = saved_dir / "sample-two.png"
    saved_one.touch()
    saved_two.touch()

    calls: list[tuple[Path, str, dict]] = []

    def fake_train(train_dir, output_model, **kwargs):
        calls.append((Path(train_dir), output_model, kwargs))
        return tmp_path / "models" / f"{output_model}.traineddata"

    monkeypatch.setattr(annotation, "_train_model", fake_train)
    monkeypatch.setattr(annotation.threading, "Thread", ImmediateThread)

    trainer = annotation.AnnotationTrainer(
        StubMaster(),
        train_dir=saved_dir,
        config=annotation.AnnotationAutoTrainConfig(
            auto_train=1,
            output_model="handwriting",
            model_dir=tmp_path / "models",
            base_lang="eng",
            max_iterations=100,
        ),
    )

    app = AnnotationApp.__new__(AnnotationApp)
    app.items = [
        AnnotationItem(Path("first.png")),
        AnnotationItem(Path("second.png")),
    ]
    app.index = 0
    app.train_dir = saved_dir
    app._get_transcription_text = lambda: "label"
    app._append_log = lambda *_args, **_kwargs: None
    app._advance = lambda: None
    app._save_annotation = lambda *_args, **_kwargs: saved_one
    app.status_var = SimpleNamespace(set=lambda _value: None)
    app.overlay_entries = []
    app.overlay_items = []
    app.rect_to_overlay = {}
    app.selected_rects = set()
    app.current_tokens = []
    app._on_sample_saved = trainer

    AnnotationApp.confirm(app)
    assert trainer.seen_samples == [saved_one]
    assert calls and calls[0][0] == saved_dir

    app._save_annotation = lambda *_args, **_kwargs: saved_two
    AnnotationApp.confirm(app)
    assert trainer.seen_samples == [saved_one, saved_two]
    assert len(calls) == 2
    assert len(master_calls) >= 2


def test_draw_select_delete_interactions(monkeypatch):
    """Selecting, drawing, and deleting overlays keeps the state consistent."""

    class DummyEntry:
        def __init__(self, *_args, width: int = 0, **_kwargs):  # pragma: no cover - width unused
            self.value = ""
            self.focused = False
            self.destroyed = False
            self.bindings: dict[str, object] = {}

        def insert(self, _index: int, value: str) -> None:
            self.value = value

        def delete(self, *_args, **_kwargs) -> None:
            self.value = ""

        def bind(self, sequence: str, handler) -> None:  # pragma: no cover - handler unused
            self.bindings[sequence] = handler

        def get(self) -> str:
            return self.value

        def focus_set(self) -> None:
            self.focused = True

        def destroy(self) -> None:
            self.destroyed = True

    class DummyText:
        def __init__(self) -> None:
            self.value = ""
            self.bindings: dict[str, object] = {}

        def delete(self, *_args, **_kwargs) -> None:
            self.value = ""

        def insert(self, *args, **kwargs) -> None:
            if args:
                self.value = args[-1]
            elif "value" in kwargs:
                self.value = kwargs["value"]
            else:  # pragma: no cover - default branch
                self.value = ""

        def get(self, *_args, **_kwargs) -> str:
            return self.value

        def bind(self, sequence: str, handler) -> None:  # pragma: no cover - handler unused
            self.bindings[sequence] = handler

    class DummyButton:
        def __init__(self) -> None:
            self.state = annotation.tk.DISABLED

        def config(self, **kwargs) -> None:
            if "state" in kwargs:
                self.state = kwargs["state"]

    class ModeVar:
        def __init__(self, value: str) -> None:
            self.value = value

        def get(self) -> str:
            return self.value

        def set(self, value: str) -> None:
            self.value = value

    class StubCanvas:
        def __init__(self) -> None:
            self._next_id = 1
            self.objects: dict[int, dict[str, object]] = {}
            self.bindings: dict[str, object] = {}
            self.config_values: dict[str, object] = {}

        def _allocate_id(self) -> int:
            obj_id = self._next_id
            self._next_id += 1
            return obj_id

        def bind(self, sequence: str, handler) -> None:  # pragma: no cover - not inspected
            self.bindings[sequence] = handler

        def create_image(self, x: float, y: float, **kwargs) -> int:
            obj_id = self._allocate_id()
            self.objects[obj_id] = {"coords": [x, y], "kwargs": kwargs}
            return obj_id

        def create_rectangle(self, x1: float, y1: float, x2: float, y2: float, **kwargs) -> int:
            obj_id = self._allocate_id()
            self.objects[obj_id] = {"coords": [x1, y1, x2, y2], "kwargs": kwargs}
            return obj_id

        def create_window(self, x: float, y: float, **kwargs) -> int:
            obj_id = self._allocate_id()
            self.objects[obj_id] = {"coords": [x, y], "kwargs": kwargs}
            return obj_id

        def tag_raise(self, *_args, **_kwargs) -> None:  # pragma: no cover - no ordering
            pass

        def coords(self, obj_id: int, *values: float) -> list[float]:
            if obj_id not in self.objects:
                return []
            if values:
                self.objects[obj_id]["coords"] = list(values)
            return list(self.objects[obj_id]["coords"])

        def delete(self, target) -> None:
            if target == "all":
                self.objects.clear()
                return
            if isinstance(target, str):
                for obj_id, data in list(self.objects.items()):
                    tags = data.get("kwargs", {}).get("tags")
                    if tags is None:
                        continue
                    if isinstance(tags, (tuple, list, set)):
                        tag_values = tuple(tags)
                    else:
                        tag_values = (tags,)
                    if target in tag_values:
                        self.objects.pop(obj_id, None)
                return
            self.objects.pop(target, None)

        def itemconfigure(self, obj_id: int, **kwargs) -> None:
            if obj_id in self.objects:
                self.objects[obj_id].setdefault("config", {}).update(kwargs)

        def focus_set(self) -> None:  # pragma: no cover - no focus behaviour
            pass

        def config(self, **kwargs) -> None:
            self.config_values.update(kwargs)

    class DummyPhotoImage:
        def __init__(self, image) -> None:
            self.image = image

    monkeypatch.setattr(annotation.tk, "Entry", DummyEntry)
    monkeypatch.setattr(annotation.ImageTk, "PhotoImage", DummyPhotoImage)

    app = AnnotationApp.__new__(AnnotationApp)
    app.canvas = StubCanvas()
    app.entry_widget = DummyText()
    app.delete_button = DummyButton()
    app.mode_var = ModeVar("select")
    app.overlay_items = []
    app.overlay_entries = []
    app.rect_to_overlay = {}
    app.selected_rects = set()
    app.current_tokens = []
    app.display_scale = (1.0, 1.0)
    app.manual_token_counter = 0
    app._drag_start = None
    app._active_temp_rect = None
    app._marquee_rect = None
    app._modifier_drag = False
    app._pressed_overlay = None
    app._user_modified_transcription = False
    app._setting_transcription = False

    image = Image.new("RGB", (100, 40), "white")
    tokens = [
        annotation.OcrToken(
            "alpha",
            (0, 0, 20, 10),
            (1, 1, 1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        ),
        annotation.OcrToken(
            "beta",
            (25, 0, 45, 10),
            (1, 1, 1, 1, 2),
            (1, 1, 1),
            (1, 1, 1),
        ),
    ]

    AnnotationApp._display_image(app, image, tokens)

    assert len(app.overlay_items) == 2
    assert [overlay.entry.get() for overlay in app.overlay_items] == ["alpha", "beta"]
    assert app.current_tokens and len(app.current_tokens) == 2
    assert app.entry_widget.get("1.0", annotation.tk.END) == "alpha beta"

    first_overlay = app.overlay_items[0]
    left, top, right, bottom = app.canvas.coords(first_overlay.rect_id)
    click_event = SimpleNamespace(x=(left + right) / 2, y=(top + bottom) / 2, state=0)

    AnnotationApp._on_canvas_button_press(app, click_event)
    AnnotationApp._on_canvas_release(app, click_event)

    assert app.selected_rects == {first_overlay.rect_id}
    assert first_overlay.selected is True
    assert app.delete_button.state == annotation.tk.NORMAL

    app.mode_var.set("draw")
    draw_start = SimpleNamespace(x=70, y=15, state=0)
    draw_end = SimpleNamespace(x=95, y=30, state=0)

    AnnotationApp._on_canvas_button_press(app, draw_start)
    AnnotationApp._on_canvas_drag(app, draw_end)
    AnnotationApp._on_canvas_release(app, draw_end)

    assert len(app.overlay_items) == 3
    manual_overlay = app.overlay_items[-1]
    assert manual_overlay.is_manual is True
    assert manual_overlay.selected is True

    manual_overlay.entry.insert(0, "gamma")
    AnnotationApp._on_overlay_modified(app, None)
    assert "gamma" in app.entry_widget.get("1.0", annotation.tk.END)

    app.mode_var.set("select")
    AnnotationApp._delete_selected(app)

    assert len(app.overlay_items) == 2
    assert all(not overlay.is_manual for overlay in app.overlay_items)
    assert app.current_tokens and len(app.current_tokens) == 2
    assert app.entry_widget.get("1.0", annotation.tk.END) == "alpha beta"
    assert not app.selected_rects
    assert manual_overlay.entry.destroyed
    assert app.delete_button.state == annotation.tk.DISABLED

    second_overlay = app.overlay_items[1]
    s_left, s_top, s_right, s_bottom = app.canvas.coords(second_overlay.rect_id)
    ctrl_event = SimpleNamespace(
        x=(s_left + s_right) / 2,
        y=(s_top + s_bottom) / 2,
        state=annotation.CONTROL_MASK,
    )

    AnnotationApp._on_canvas_button_press(app, click_event)
    AnnotationApp._on_canvas_release(app, click_event)
    AnnotationApp._on_canvas_button_press(app, ctrl_event)
    AnnotationApp._on_canvas_release(app, ctrl_event)

    assert app.selected_rects == {first_overlay.rect_id, second_overlay.rect_id}

    AnnotationApp._delete_selected(app)

    assert not app.overlay_items
    assert not app.rect_to_overlay
    assert not app.current_tokens
    assert not app.selected_rects
    assert app.entry_widget.get("1.0", annotation.tk.END) == ""
    assert app.delete_button.state == annotation.tk.DISABLED
