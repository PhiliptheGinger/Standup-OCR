"""Tests for Kraken CLI integration helpers."""

from __future__ import annotations

import logging
import shutil
import sys
import types
from pathlib import Path

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import kraken_adapter
from PIL import Image


def _setup_monkeypatch(monkeypatch, help_text: str | None):
    """Prepare common monkeypatches for ``kraken_adapter.train`` tests."""

    monkeypatch.setattr(kraken_adapter, "_require_kraken", lambda: None)
    monkeypatch.setattr(kraken_adapter.shutil, "which", lambda name: "/usr/bin/ketos" if name == "ketos" else None)

    captured_commands: list[list[str]] = []

    def fake_run(cmd, *_, **kwargs):
        captured_commands.append(cmd)
        if cmd[:3] == ["/usr/bin/ketos", "train", "--help"]:
            if help_text is None:
                raise RuntimeError("help not available")
            return types.SimpleNamespace(stdout=help_text, stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(kraken_adapter.subprocess, "run", fake_run)
    kraken_adapter._ketos_train_validation_flag.cache_clear()

    return captured_commands


def test_train_uses_partition_when_supported(monkeypatch, tmp_path):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --partition FLOAT"
    captured = _setup_monkeypatch(monkeypatch, help_text)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    lines_dir = dataset_dir / "lines"
    lines_dir.mkdir()
    line_image = lines_dir / "sample.png"
    line_image.write_bytes(b"")
    model_out = tmp_path / "model.mlmodel"

    result = kraken_adapter.train(dataset_dir, model_out, epochs=10, val_split=0.2)

    assert result == model_out
    assert captured[0][:3] == ["/usr/bin/ketos", "train", "--help"]
    assert captured[-1] == [
        "/usr/bin/ketos",
        "train",
        "--output",
        str(model_out),
        "--epochs",
        "10",
        "--partition",
        "0.2",
        str(line_image),
    ]


def test_train_falls_back_to_validation(monkeypatch, tmp_path):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --validation FLOAT"
    captured = _setup_monkeypatch(monkeypatch, help_text)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    lines_dir = dataset_dir / "lines"
    lines_dir.mkdir()
    line_image = lines_dir / "sample.png"
    line_image.write_bytes(b"")
    model_out = tmp_path / "model.mlmodel"

    kraken_adapter.train(dataset_dir, model_out, epochs=5, val_split=0.3)

    assert captured[-1] == [
        "/usr/bin/ketos",
        "train",
        "--output",
        str(model_out),
        "--epochs",
        "5",
        "--validation",
        "0.3",
        str(line_image),
    ]


def test_train_skips_validation_flag_when_disabled(monkeypatch, tmp_path):
    captured = _setup_monkeypatch(monkeypatch, help_text="Usage: ketos train [OPTIONS]")

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    lines_dir = dataset_dir / "lines"
    lines_dir.mkdir()
    line_image = lines_dir / "sample.png"
    line_image.write_bytes(b"")
    model_out = tmp_path / "model.mlmodel"

    kraken_adapter.train(dataset_dir, model_out, epochs=2, val_split=0.0)

    assert captured == [
        [
            "/usr/bin/ketos",
            "train",
            "--output",
            str(model_out),
            "--epochs",
            "2",
            str(line_image),
        ]
    ]


def test_train_prefers_pagexml_when_available(monkeypatch, tmp_path):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --partition FLOAT"
    captured = _setup_monkeypatch(monkeypatch, help_text)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    lines_dir = dataset_dir / "lines"
    lines_dir.mkdir()
    (lines_dir / "line.png").write_bytes(b"")

    pagexml_dir = dataset_dir / "pagexml"
    pagexml_dir.mkdir()
    pagexml_file = pagexml_dir / "page.xml"
    pagexml_file.write_text("<PcGts></PcGts>", encoding="utf-8")

    model_out = tmp_path / "model.mlmodel"

    kraken_adapter.train(dataset_dir, model_out, epochs=3, val_split=0.4)

    assert captured[-1][-1] == str(pagexml_file)


def test_train_pads_problematic_line_images(monkeypatch, tmp_path, caplog):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --partition FLOAT"
    captured = _setup_monkeypatch(monkeypatch, help_text)

    dataset_dir = tmp_path / "dataset"
    lines_dir = dataset_dir / "lines"
    lines_dir.mkdir(parents=True)

    line_image = lines_dir / "problem.png"
    Image.new("L", (10, 10), color=200).save(line_image)
    label_path = lines_dir / "problem.gt.txt"
    label_path.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(
        kraken_adapter,
        "_line_needs_center_padding",
        lambda path, target_height=120: Path(path) == line_image,
    )

    tempdirs: list[object] = []

    class DummyTempDir:
        def __init__(self, prefix=None):
            self.path = tmp_path / "sanitized"
            self.path.mkdir()
            self.name = str(self.path)
            self.cleaned = False

        def cleanup(self):
            self.cleaned = True

    def fake_tempdir(prefix=None):
        tmp = DummyTempDir(prefix)
        tempdirs.append(tmp)
        return tmp

    monkeypatch.setattr(kraken_adapter.tempfile, "TemporaryDirectory", fake_tempdir)

    caplog.set_level(logging.WARNING)

    model_out = tmp_path / "model.mlmodel"
    kraken_adapter.train(dataset_dir, model_out, epochs=1, val_split=0.2)

    assert tempdirs and tempdirs[0].cleaned

    sanitized_path = Path(captured[-1][-1])
    assert sanitized_path.is_relative_to(tempdirs[0].path)
    sanitized_label = sanitized_path.with_name(f"{sanitized_path.stem}.gt.txt")
    assert sanitized_label.exists()
    assert sanitized_label.read_text(encoding="utf-8") == "hello"
    assert "Applied additional padding" in caplog.text

    shutil.rmtree(tempdirs[0].path)
