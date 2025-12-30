from __future__ import annotations

import json
from pathlib import Path
import textwrap

from PIL import ExifTags, Image, ImageOps

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from exporters import load_pagexml, save_line_crops, save_pagexml  # type: ignore
from line_store import Line  # type: ignore


def test_save_line_crops_writes_metadata(tmp_path):
    image_path = tmp_path / "page.png"
    Image.new("L", (40, 20), color=255).save(image_path)

    out_dir = tmp_path / "train"
    lines = [
        Line(
            id=1,
            baseline=[(5, 10), (25, 10)],
            bbox=(4, 6, 26, 12),
            text="Sample text",
            order_key=(0, 0, 0, 1, 1),
            selected=False,
            is_manual=True,
        ),
        Line(
            id=2,
            baseline=[(5, 15), (30, 15)],
            bbox=(4, 13, 30, 18),
            text="Second line",
            order_key=(0, 0, 0, 2, 1),
            selected=False,
            is_manual=False,
        ),
    ]

    save_line_crops(image_path, lines, out_dir)

    metadata_path = out_dir / f"{image_path.stem}.boxes.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    assert len(metadata) == 2

    first = metadata[0]
    assert first["image"].endswith("_line01.png")
    assert first["text_file"].endswith("_line01.gt.txt")
    assert first["text"] == "Sample text"
    assert first["bbox"] == {"left": 4, "top": 6, "right": 26, "bottom": 12}
    assert first["is_manual"] is True

    second = metadata[1]
    assert second["is_manual"] is False
    assert second["text"] == "Second line"


def _image_description_tag() -> int:
    for tag, name in ExifTags.TAGS.items():
        if name == "ImageDescription":
            return tag
    raise AssertionError("ImageDescription tag missing from ExifTags")


def _orientation_tag() -> int:
    for tag, name in ExifTags.TAGS.items():
        if name == "Orientation":
            return tag
    raise AssertionError("Orientation tag missing from ExifTags")


def test_save_line_crops_embeds_exif_metadata(tmp_path):
    image_path = tmp_path / "page.png"
    Image.new("L", (40, 20), color=255).save(image_path)

    out_dir = tmp_path / "train"
    line = Line(
        id=3,
        baseline=[(5, 10), (25, 10)],
        bbox=(4, 6, 26, 12),
        text="Deskew",
        order_key=(0, 0, 0, 1, 1),
        selected=False,
        is_manual=False,
    )
    save_line_crops(image_path, [line], out_dir)

    crop_path = out_dir / f"{image_path.stem}_line01.png"
    assert crop_path.exists()

    with Image.open(crop_path) as image:
        exif = image.getexif()
    raw = exif.get(_image_description_tag())
    assert raw
    metadata = json.loads(raw)
    assert metadata["signature"] == "standup-ocr"
    payload = metadata["payload"]
    assert payload["line_id"] == 3
    assert payload["bbox_page"] == {"left": 4, "top": 6, "right": 26, "bottom": 12}
    assert payload["bbox_local"] == {"left": 4, "top": 4, "right": 26, "bottom": 10}
    assert payload["baseline_local"][0] == [5.0, 8.0]


def test_save_pagexml_uses_oriented_dimensions(tmp_path):
        image_path = tmp_path / "portrait.jpg"
        raw = Image.new("RGB", (30, 60), color="white")
        exif = raw.getexif()
        exif[_orientation_tag()] = 6
        raw.save(image_path, exif=exif.tobytes())

        line = Line(
                id=1,
                baseline=[(2, 4), (25, 4)],
                bbox=(2, 2, 25, 12),
                text="Hello",
                order_key=(0, 0, 0, 1, 1),
                selected=False,
                is_manual=False,
        )
        xml_path = tmp_path / "portrait.xml"
        save_pagexml(image_path, [line], xml_path)

        content = xml_path.read_text(encoding="utf8")
        assert 'imageWidth="60"' in content
        assert 'imageHeight="30"' in content


def test_load_pagexml_sorts_lines_by_geometry(tmp_path):
        xml_path = tmp_path / "order.xml"
        xml_path.write_text(
                textwrap.dedent(
                        """
                        <?xml version="1.0" encoding="UTF-8"?>
                        <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
                            <Metadata><Creator>test</Creator></Metadata>
                            <Page imageFilename="page.png" imageWidth="400" imageHeight="200">
                                <TextRegion id="r1">
                                    <TextLine id="l2">
                                        <Coords points="80,120 200,120 200,160 80,160" />
                                        <Baseline points="80,160 200,160" />
                                        <TextEquiv><Unicode>Lower line</Unicode></TextEquiv>
                                    </TextLine>
                                    <TextLine id="l1">
                                        <Coords points="60,20 220,20 220,60 60,60" />
                                        <Baseline points="60,60 220,60" />
                                        <TextEquiv><Unicode>Upper line</Unicode></TextEquiv>
                                    </TextLine>
                                </TextRegion>
                            </Page>
                        </PcGts>
                        """
                ),
                encoding="utf8",
        )

        lines = load_pagexml(xml_path)
        assert [line.text for line in lines] == ["Upper line", "Lower line"]
        assert [line.order_key for line in lines] == [(1, 1, 1, 1, 1), (2, 1, 1, 1, 1)]


def test_load_pagexml_reprojects_with_exif_orientation(tmp_path):
        image_path = tmp_path / "rotated.jpg"
        raw = Image.new("RGB", (30, 60), color="white")
        exif = raw.getexif()
        exif[_orientation_tag()] = 6
        raw.save(image_path, exif=exif.tobytes())

        with Image.open(image_path) as image:
                prepared_size = ImageOps.exif_transpose(image).size

        xml_path = tmp_path / "rotated.xml"
        xml_path.write_text(
                textwrap.dedent(
                        """
                        <?xml version="1.0" encoding="UTF-8"?>
                        <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
                            <Metadata><Creator>test</Creator></Metadata>
                            <Page imageFilename="rotated.jpg" imageWidth="30" imageHeight="60">
                                <TextRegion id="r1">
                                    <TextLine id="l1">
                                        <Coords points="5,40 15,40 15,55 5,55" />
                                        <Baseline points="5,55 15,55" />
                                        <TextEquiv><Unicode>Rotated</Unicode></TextEquiv>
                                    </TextLine>
                                </TextRegion>
                            </Page>
                        </PcGts>
                        """
                ),
                encoding="utf8",
        )

        lines = load_pagexml(xml_path, image_path=image_path, prepared_size=prepared_size)
        assert len(lines) == 1
        bbox = lines[0].bbox
        assert bbox == (4, 5, 19, 15)
        assert lines[0].id == 1
        assert lines[0].order_key == (1, 1, 1, 1, 1)


def test_load_pagexml_reprojects_with_exif_orientation_even_when_in_bounds(tmp_path):
        image_path = tmp_path / "rotated.jpg"
        raw = Image.new("RGB", (30, 60), color="white")
        exif = raw.getexif()
        exif[_orientation_tag()] = 6
        raw.save(image_path, exif=exif.tobytes())

        with Image.open(image_path) as image:
                prepared_size = ImageOps.exif_transpose(image).size

        # This box is in-bounds for both (30x60) and (60x30), so the old
        # out-of-bounds heuristic would incorrectly skip reprojection.
        xml_path = tmp_path / "rotated_in_bounds.xml"
        xml_path.write_text(
                textwrap.dedent(
                        """
                        <?xml version="1.0" encoding="UTF-8"?>
                        <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
                            <Metadata><Creator>test</Creator></Metadata>
                            <Page imageFilename="rotated.jpg" imageWidth="30" imageHeight="60">
                                <TextRegion id="r1">
                                    <TextLine id="l1">
                                        <Coords points="5,5 15,5 15,15 5,15" />
                                        <Baseline points="5,15 15,15" />
                                        <TextEquiv><Unicode>Rotated</Unicode></TextEquiv>
                                    </TextLine>
                                </TextRegion>
                            </Page>
                        </PcGts>
                        """
                ),
                encoding="utf8",
        )

        lines = load_pagexml(xml_path, image_path=image_path, prepared_size=prepared_size)
        assert len(lines) == 1
        assert lines[0].bbox == (44, 5, 54, 15)
