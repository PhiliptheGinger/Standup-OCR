from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src import refine
import main as cli_main


class StubResponse:
    def __init__(self, text: str):
        self.output_text = text


class StubTranscriber:
    def __init__(self) -> None:
        self.calls = []
        self.max_output_tokens = 256
        self.prompt = "prompt"

    def generate(self, image_path: Path, **kwargs):  # noqa: D401 - signature controlled by tests
        self.calls.append(kwargs)
        return StubResponse('{"corrected_text":"Fixed text","confidence":"75%","notes":"ok"}')


def _sample_tokens() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": "Hello",
                "left": 10,
                "top": 5,
                "right": 30,
                "bottom": 25,
                "confidence": 85,
                "page_num": 1,
                "block_num": 1,
                "par_num": 1,
                "line_num": 1,
                "word_num": 1,
            },
            {
                "text": "world",
                "left": 35,
                "top": 6,
                "right": 70,
                "bottom": 26,
                "confidence": 90,
                "page_num": 1,
                "block_num": 1,
                "par_num": 1,
                "line_num": 1,
                "word_num": 2,
            },
        ]
    )


def test_token_payload_groups_words():
    payload = refine._token_payload(_sample_tokens())
    assert len(payload) == 1
    first_line = payload[0]
    assert first_line["bbox"] == [10, 5, 70, 26]
    words = first_line["words"]
    assert [w["text"] for w in words] == ["Hello", "world"]
    assert words[0]["bbox"] == [10, 5, 30, 25]


@pytest.mark.parametrize(
    "value,expected",
    [(0.9, 0.9), ("0.75", 0.75), ("75%", 0.75), (75, 0.75), (-5, 0.0), (200, 1.0)],
)
def test_normalise_confidence(value, expected):
    assert refine._normalise_confidence(value) == pytest.approx(expected)


def test_run_refinement_uses_hint(monkeypatch):
    stub = StubTranscriber()
    monkeypatch.setattr(refine, "ocr_detailed", lambda *args, **kwargs: _sample_tokens())
    results = refine.run_refinement(
        [Path("image.png")],
        transcriber=stub,
        engine="tesseract",
        tesseract_model=None,
        tessdata_dir=None,
        psm=6,
        temperature=0.2,
        max_output_tokens=123,
    )
    assert len(results) == 1
    result = results[0]
    assert result.corrected_text == "Fixed text"
    assert result.confidence == pytest.approx(0.75)
    assert result.rough_text == "Hello world"
    assert result.tokens
    assert stub.calls
    call = stub.calls[0]
    assert call["temperature"] == 0.2
    assert call["max_output_tokens"] == 123
    hint = call["hint_text"]
    payload = json.loads(hint)
    assert payload["rough_text"] == "Hello world"
    assert payload["lines"][0]["words"][0]["text"] == "Hello"


def test_handle_refine_writes_outputs(monkeypatch, tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"fake")

    class DummyTranscriber:
        def __init__(self, *_, **__):
            pass

    monkeypatch.setattr(cli_main, "GPTTranscriber", DummyTranscriber)

    def fake_run_refinement(images, **kwargs):
        assert images == [image]
        return [
            refine.RefinementResult(
                image=image,
                engine="tesseract",
                corrected_text="Clean text",
                confidence=0.9,
                rough_text="rough",
                tokens=[{"line_index": 0, "bbox": [0, 0, 10, 10], "words": []}],
                notes="checked",
            )
        ]

    monkeypatch.setattr(cli_main, "run_refinement", fake_run_refinement)

    args = argparse.Namespace(
        source=image,
        output_dir=tmp_path / "out",
        engine="tesseract",
        tesseract_model=None,
        tessdata_dir=None,
        psm=6,
        kraken_model=None,
        gpt_model="gpt",
        gpt_prompt=None,
        gpt_cache_dir=None,
        gpt_max_output_tokens=256,
        gpt_temperature=None,
    )

    cli_main.handle_refine(args)

    json_path = args.output_dir / "scan.json"
    text_path = args.output_dir / "scan.txt"
    assert json_path.exists()
    assert text_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["corrected_text"] == "Clean text"
    assert payload["confidence"] == pytest.approx(0.9)
    assert payload["notes"] == "checked"
    assert text_path.read_text(encoding="utf-8").strip() == "Clean text"
