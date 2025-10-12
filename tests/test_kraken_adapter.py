"""Tests for Kraken CLI integration helpers."""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from src import kraken_adapter


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
