from pathlib import Path

from PIL import Image

from src.kraken_dataset import sanitize_line_dataset


def _create_sample_line(folder: Path, stem: str, *, text: str, suffix: str = ".gt.txt") -> None:
    image_path = folder / f"{stem}.png"
    Image.new("RGB", (32, 12), color="white").save(image_path)
    label_path = folder / f"{stem}{suffix}"
    label_path.write_text(text, encoding="utf8")


def test_sanitize_line_dataset_writes_clean_images(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _create_sample_line(source, "sample", text="Hello Kraken")

    output = tmp_path / "clean"
    stats = sanitize_line_dataset(source, output, resize_width=0)

    assert stats.processed == 1
    cleaned_image = output / "sample.png"
    cleaned_label = output / "sample.gt.txt"
    assert cleaned_image.exists()
    assert cleaned_label.exists()
    assert cleaned_label.read_text(encoding="utf8") == "Hello Kraken"


def test_sanitize_accepts_plain_txt_labels(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    _create_sample_line(source, "plain", text="Alt suffix", suffix=".txt")

    output = tmp_path / "dst"
    stats = sanitize_line_dataset(source, output, resize_width=0)

    assert stats.processed == 1
    assert not stats.missing_gt
    assert (output / "plain.gt.txt").read_text(encoding="utf8") == "Alt suffix"
