import io
from pathlib import Path

from src.training import _split_boxes_to_chars, _validate_lstmf


def test_split_boxes_to_chars_simple():
    # one box covering width 0..90 for word 'abc' should split into 3 boxes
    box_lines = ["word 0 30 90 0 0"]
    raw_gt = "abc\n"
    out = _split_boxes_to_chars(box_lines, raw_gt)
    assert out is not None
    assert len(out) == 3
    assert out[0].startswith("a ") and out[1].startswith("b ") and out[2].startswith("c ")


def test_split_boxes_to_chars_with_floats_and_multiple_boxes():
    box_lines = ["x 0.0 10.0 50.5 0 0", "x 50.6 10 100.0 0 0"]
    raw_gt = "hello\n"
    out = _split_boxes_to_chars(box_lines, raw_gt)
    assert out is not None
    assert len(out) == 5
    # characters must match the GT characters
    gt_chars = [c for c in list(raw_gt.rstrip("\n")) if c not in (" ", "\t", "\n")]
    for line, ch in zip(out, gt_chars):
        assert line.startswith(f"{ch} ")


def test_validate_lstmf_sizes(tmp_path: Path):
    small = tmp_path / "small.lstmf"
    small.write_bytes(b"\x00" * 100)
    assert _validate_lstmf(small) is False

    ok = tmp_path / "ok.lstmf"
    ok.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024)
    assert _validate_lstmf(ok) is True
