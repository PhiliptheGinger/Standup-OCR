"""Tests for Kraken CLI integration helpers."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

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
    captured_envs: list[dict[str, str] | None] = []

    def fake_run(cmd, *_, **kwargs):
        captured_commands.append(cmd)
        if cmd[:3] == ["/usr/bin/ketos", "train", "--help"]:
            if help_text is None:
                raise RuntimeError("help not available")
            return types.SimpleNamespace(stdout=help_text, stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(kraken_adapter.subprocess, "run", fake_run)

    def fake_live_output(cmd, *, env=None):
        captured_commands.append(cmd)
        captured_envs.append(env)
        return "", ""

    monkeypatch.setattr(kraken_adapter, "_run_with_live_output", fake_live_output)
    kraken_adapter._ketos_train_validation_flag.cache_clear()

    return captured_commands, captured_envs


def _dataset_with_line(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    lines_dir = dataset_dir / "lines"
    lines_dir.mkdir()
    line_image = lines_dir / "sample.png"
    line_image.write_bytes(b"")
    return dataset_dir


def test_train_uses_partition_when_supported(monkeypatch, tmp_path):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --partition FLOAT"
    captured, _ = _setup_monkeypatch(monkeypatch, help_text)

    dataset_dir = _dataset_with_line(tmp_path)
    line_image = dataset_dir / "lines" / "sample.png"
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
    captured, _ = _setup_monkeypatch(monkeypatch, help_text)

    dataset_dir = _dataset_with_line(tmp_path)
    line_image = dataset_dir / "lines" / "sample.png"
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
    captured, _ = _setup_monkeypatch(monkeypatch, help_text="Usage: ketos train [OPTIONS]")

    dataset_dir = _dataset_with_line(tmp_path)
    line_image = dataset_dir / "lines" / "sample.png"
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
    captured, _ = _setup_monkeypatch(monkeypatch, help_text)

    dataset_dir = _dataset_with_line(tmp_path)

    pagexml_dir = dataset_dir / "pagexml"
    pagexml_dir.mkdir()
    pagexml_file = pagexml_dir / "page.xml"
    pagexml_file.write_text("<PcGts></PcGts>", encoding="utf-8")

    model_out = tmp_path / "model.mlmodel"

    kraken_adapter.train(dataset_dir, model_out, epochs=3, val_split=0.4)

    assert captured[-1][-1] == str(pagexml_file)


def test_train_pads_problematic_line_images(monkeypatch, tmp_path, caplog):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --partition FLOAT"
    captured, _ = _setup_monkeypatch(monkeypatch, help_text)

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


def test_train_surfaces_model_not_improving_error(monkeypatch, tmp_path):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --partition FLOAT"
    captured = []
    captured_envs: list[dict[str, str] | None] = []

    def fake_run(cmd, *_, **kwargs):
        if cmd[:3] == ["/usr/bin/ketos", "train", "--help"]:
            return types.SimpleNamespace(stdout=help_text, stderr="")
        raise AssertionError("unexpected subprocess.run invocation")

    def fake_live_output(cmd, *, env=None):
        captured.append(cmd)
        captured_envs.append(env)
        raise subprocess.CalledProcessError(
            1,
            cmd,
            output=(
                "[10/12/25 19:47:08] WARNING  Model did not improve during    "
                "recognition.py:321\nSeed set to 42"
            ),
            stderr="",
        )

    monkeypatch.setattr(kraken_adapter, "_require_kraken", lambda: None)
    monkeypatch.setattr(kraken_adapter.shutil, "which", lambda name: "/usr/bin/ketos" if name == "ketos" else None)
    monkeypatch.setattr(kraken_adapter.subprocess, "run", fake_run)
    monkeypatch.setattr(kraken_adapter, "_run_with_live_output", fake_live_output)
    kraken_adapter._ketos_train_validation_flag.cache_clear()

    dataset_dir = _dataset_with_line(tmp_path)
    model_out = tmp_path / "model.mlmodel"

    with pytest.raises(RuntimeError) as excinfo:
        kraken_adapter.train(dataset_dir, model_out, epochs=1, val_split=0.1)

    assert "Kraken aborted training" in str(excinfo.value)
    assert "exit code 1" in str(excinfo.value)
    assert captured[-1][0] == "/usr/bin/ketos"


def test_train_forces_utf8_stdio(monkeypatch, tmp_path):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --partition FLOAT"
    captured, envs = _setup_monkeypatch(monkeypatch, help_text)

    dataset_dir = _dataset_with_line(tmp_path)
    model_out = tmp_path / "model.mlmodel"

    kraken_adapter.train(dataset_dir, model_out, epochs=1, val_split=0.1)

    assert envs[-1] is not None
    assert envs[-1]["PYTHONIOENCODING"] == "utf-8"
    assert envs[-1]["PYTHONUTF8"] == "1"


def test_train_sets_requested_progress_env(monkeypatch, tmp_path):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --partition FLOAT"
    _, envs = _setup_monkeypatch(monkeypatch, help_text)

    dataset_dir = _dataset_with_line(tmp_path)
    model_out = tmp_path / "model.mlmodel"

    kraken_adapter.train(dataset_dir, model_out, epochs=1, val_split=0.1, progress="plain")

    assert envs[-1]["KRAKEN_PROGRESS"] == "plain"


def test_train_defaults_progress_when_helper_requests_it(monkeypatch, tmp_path):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --partition FLOAT"
    _, envs = _setup_monkeypatch(monkeypatch, help_text)

    monkeypatch.setattr(kraken_adapter, "_default_kraken_progress", lambda: "plain")
    monkeypatch.delenv("KRAKEN_PROGRESS", raising=False)

    dataset_dir = _dataset_with_line(tmp_path)
    model_out = tmp_path / "model.mlmodel"

    kraken_adapter.train(dataset_dir, model_out, epochs=1, val_split=0.1)

    assert envs[-1]["KRAKEN_PROGRESS"] == "plain"


def test_train_respects_none_progress_override(monkeypatch, tmp_path):
    help_text = "Usage: ketos train [OPTIONS]\n\nOptions:\n  --partition FLOAT"
    _, envs = _setup_monkeypatch(monkeypatch, help_text)

    monkeypatch.setenv("KRAKEN_PROGRESS", "rich")

    dataset_dir = _dataset_with_line(tmp_path)
    model_out = tmp_path / "model.mlmodel"

    kraken_adapter.train(dataset_dir, model_out, epochs=1, val_split=0.1, progress="none")

    assert "KRAKEN_PROGRESS" not in envs[-1]


def test_segment_pages_with_kraken_exports_crops(monkeypatch, tmp_path):
    page = tmp_path / "page.png"
    Image.new("L", (100, 80), color=200).save(page)

    captured_pagexml: list[Path | None] = []

    def fake_get_segmentation(
        image_path,
        *,
        model=None,
        out_pagexml=None,
        global_cli_args=None,
        segment_cli_args=None,
    ):
        captured_pagexml.append(out_pagexml)
        return {
            "lines": [
                {
                    "id": "l1",
                    "boundary": [[10, 10], [90, 10], [90, 30], [10, 30]],
                },
                {
                    "id": "l2",
                    "baseline": [[20, 50], [80, 52]],
                },
            ]
        }

    monkeypatch.setattr(kraken_adapter, "_get_segmentation", fake_get_segmentation)

    out_dir = tmp_path / "lines"
    xml_dir = tmp_path / "pagexml"
    stats = kraken_adapter.segment_pages_with_kraken(
        [page],
        out_dir,
        pagexml_dir=xml_dir,
        padding=5,
        min_width=10,
        min_height=10,
    )

    assert stats.pages == 1
    assert stats.lines == 2
    assert stats.errors == 0
    assert stats.skipped == 0

    first_line = out_dir / "page_line001.png"
    assert first_line.exists()
    first_gt = first_line.with_suffix(".gt.txt")
    assert first_gt.exists()
    assert first_gt.read_text(encoding="utf8") == ""
    metadata = json.loads(first_line.with_suffix(".boxes.json").read_text(encoding="utf8"))
    assert metadata["line_index"] == 1
    assert metadata["id"] == "l1"

    second_line = out_dir / "page_line002.png"
    assert second_line.exists()

    pagexml_file = xml_dir / "page.xml"
    assert captured_pagexml[-1] == pagexml_file


def test_segment_via_cli_uses_correct_command(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"data")

    tmp_root = tmp_path / "cli-tmp"

    class DummyTempDir:
        def __enter__(self):
            tmp_root.mkdir(parents=True, exist_ok=True)
            return str(tmp_root)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(kraken_adapter.tempfile, "TemporaryDirectory", lambda: DummyTempDir())
    monkeypatch.setattr(kraken_adapter.shutil, "which", lambda name: "/usr/bin/kraken" if name == "kraken" else None)

    captured_cmd: list[str] = []

    def fake_run(cmd, check):
        captured_cmd.extend(cmd)
        (tmp_root / "segmentation.json").write_text(json.dumps({"lines": []}), encoding="utf-8")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(kraken_adapter.subprocess, "run", fake_run)

    result = kraken_adapter._segment_via_cli(
        image,
        model="seg.mlmodel",
        global_cli_args=["--device", "cuda:0", "--threads", "2"],
        segment_cli_args=["--baseline", "--text-direction", "vertical-rl"],
    )

    expected_json = tmp_root / "segmentation.json"
    assert result == {"lines": []}
    assert captured_cmd == [
        "/usr/bin/kraken",
        "--device",
        "cuda:0",
        "--threads",
        "2",
        "-i",
        str(image),
        str(expected_json),
        "binarize",
        "segment",
        "--model",
        "seg.mlmodel",
        "--baseline",
        "--text-direction",
        "vertical-rl",
    ]
