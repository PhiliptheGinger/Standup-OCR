"""Tests for the annotation helpers."""

import sys
from pathlib import Path

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
