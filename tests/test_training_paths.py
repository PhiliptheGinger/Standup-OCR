"""Tests for tessdata discovery helpers and training directory bootstrapping."""

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - test shim
    sys.path.insert(0, str(ROOT))

if "cv2" not in sys.modules:  # pragma: no cover - shim when OpenCV is unavailable
    sys.modules["cv2"] = types.SimpleNamespace(imwrite=lambda *_args, **_kwargs: True)

from src import training


def test_resolve_uses_environment(monkeypatch, tmp_path):
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    monkeypatch.setenv("TESSDATA_PREFIX", str(tessdata))

    result = training._resolve_tessdata_dir(None)

    assert result == tessdata


def test_resolve_uses_tesseract_binary_location(monkeypatch, tmp_path):
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)

    install_root = tmp_path / "scoop" / "apps" / "tesseract" / "current"
    tessdata = install_root / "tessdata"
    tessdata.mkdir(parents=True)

    exe_path = install_root / "tesseract.exe"
    exe_path.write_text("", encoding="utf-8")

    def fake_run(*args, **kwargs):  # pragma: no cover - guardrail
        raise FileNotFoundError

    monkeypatch.setattr(training.subprocess, "run", fake_run)
    monkeypatch.setattr(training.shutil, "which", lambda _: str(exe_path))

    # Avoid accidentally matching an existing tessdata folder on the test machine.
    for env_var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA", "HOME", "USERPROFILE"):
        monkeypatch.setenv(env_var, str(tmp_path / "missing"))

    result = training._resolve_tessdata_dir(None)

    assert result == tessdata


def test_discover_images_creates_sample(tmp_path):
    train_dir = tmp_path / "train"

    images = training._discover_images(train_dir)

    assert len(images) == 1
    sample = images[0]
    assert sample.name == "word_sample.png"
    assert sample.exists()


def test_discover_images_prefers_lines_directory(tmp_path):
    train_dir = tmp_path / "train"
    lines_dir = train_dir / "lines"
    lines_dir.mkdir(parents=True)

    handwriting = lines_dir / "my_sample.png"
    handwriting.write_bytes(b"not a real image")

    images = training._discover_images(train_dir)

    assert images == [handwriting]
    assert not (train_dir / "word_sample.png").exists()


def test_discover_images_ignores_bootstrap_when_real_samples_exist(tmp_path):
    train_dir = tmp_path / "train"
    lines_dir = train_dir / "lines"
    lines_dir.mkdir(parents=True)

    bootstrap = train_dir / "word_sample.png"
    bootstrap.write_bytes(b"placeholder")

    handwriting = lines_dir / "another_sample.png"
    handwriting.write_bytes(b"content")

    extra = train_dir / "root_sample.png"
    extra.write_bytes(b"root content")

    images = training._discover_images(train_dir)

    assert handwriting in images
    assert extra in images
    assert bootstrap not in images
