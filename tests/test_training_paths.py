"""Tests for tessdata discovery helpers and training directory bootstrapping."""

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
