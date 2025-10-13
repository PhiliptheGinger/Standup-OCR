"""Pipeline utilities for GPT-assisted OCR refinement."""
from __future__ import annotations

import json
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .gpt_ocr import GPTTranscriber, GPTTranscriptionError
from .ocr import ocr_detailed

try:  # pragma: no cover - optional dependency
    from .kraken_adapter import is_available as kraken_available, ocr as kraken_ocr
except Exception:  # pragma: no cover - defensive import guard
    kraken_available = lambda: False  # type: ignore
    kraken_ocr = None  # type: ignore

log = logging.getLogger(__name__)

DEFAULT_REFINE_PROMPT = (
    "You are an expert at correcting handwritten OCR. "
    "You receive the original image together with a rough OCR attempt and bounding boxes. "
    "Return corrected text that restores the intended meaning. "
    "Respond in JSON with fields 'corrected_text' and 'confidence' (0-1). "
    "Include an optional 'notes' string when helpful. Preserve line breaks where appropriate."
)

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ocr_refinement_result",
        "schema": {
            "type": "object",
            "properties": {
                "corrected_text": {"type": "string"},
                "confidence": {"type": ["number", "string"]},
                "notes": {"type": "string"},
            },
            "required": ["corrected_text", "confidence"],
            "additionalProperties": False,
        },
    },
}

_TOKEN_GROUP_COLS = ["page_num", "block_num", "par_num", "line_num"]
_WORD_SORT_COLS = _TOKEN_GROUP_COLS + ["word_num"]


@dataclass
class RefinementResult:
    """Container describing a single GPT-backed refinement."""

    image: Path
    engine: str
    corrected_text: str
    confidence: float
    rough_text: str
    tokens: List[dict]
    notes: Optional[str] = None


def _normalise_confidence(value: object) -> float:
    """Return ``value`` coerced to a 0-1 float."""

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        try:
            numeric = float(cleaned)
        except ValueError as exc:  # pragma: no cover - defensive
            raise GPTTranscriptionError(f"Unable to parse confidence value: {value!r}") from exc
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:  # pragma: no cover - defensive
        raise GPTTranscriptionError(f"Unsupported confidence type: {type(value)!r}")

    if math.isnan(numeric):
        raise GPTTranscriptionError("Confidence value was NaN")

    if numeric > 1:
        numeric /= 100.0
    numeric = max(0.0, min(1.0, numeric))
    return numeric


def _extract_text(tokens: pd.DataFrame) -> str:
    """Collapse recognised tokens into a rough text transcription."""

    if tokens.empty:
        return ""
    data = tokens.copy()
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data = data[data["text"].ne("")]
    if data.empty:
        return ""
    data = data.sort_values(_WORD_SORT_COLS)
    grouped = data.groupby(_TOKEN_GROUP_COLS, sort=True)["text"].apply(lambda words: " ".join(words))
    return "\n".join(grouped.tolist()).strip()


def _token_payload(tokens: pd.DataFrame) -> List[dict]:
    """Return simplified token metadata for prompts and logs."""

    if tokens.empty:
        return []

    data = tokens.copy()
    data["text"] = data["text"].fillna("").astype(str)
    rows: List[dict] = []
    grouped = data.groupby(_TOKEN_GROUP_COLS, sort=True)
    for index, group in enumerate(grouped):
        _, frame = group
        frame = frame.sort_values("word_num")
        words = []
        lefts: List[float] = []
        tops: List[float] = []
        rights: List[float] = []
        bottoms: List[float] = []
        for _, row in frame.iterrows():
            text = row.get("text", "").strip()
            if not text:
                continue
            left = float(row.get("left", 0) or 0)
            top = float(row.get("top", 0) or 0)
            right = float(row.get("right", left))
            bottom = float(row.get("bottom", top))
            bbox = [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))]
            word_conf = row.get("confidence")
            if isinstance(word_conf, (int, float)) and not math.isnan(word_conf):
                confidence = float(word_conf)
            else:
                confidence = None
            words.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )
            lefts.append(left)
            tops.append(top)
            rights.append(right)
            bottoms.append(bottom)
        if not words:
            continue
        line_bbox = [
            int(round(min(lefts))),
            int(round(min(tops))),
            int(round(max(rights))),
            int(round(max(bottoms))),
        ]
        rows.append({
            "line_index": index,
            "bbox": line_bbox,
            "words": words,
        })
    return rows


def _build_hint(engine: str, rough_text: str, tokens: List[dict]) -> str:
    payload = {
        "engine": engine,
        "rough_text": rough_text,
        "bbox_format": "[left, top, right, bottom] pixels",
        "lines": tokens,
    }
    return json.dumps(payload, ensure_ascii=False)


def _run_kraken_ocr(image_path: Path, model_path: Path) -> str:
    if not kraken_available():
        raise RuntimeError(
            "Kraken is not installed. Install it with 'pip install kraken[serve]' to use --engine kraken."
        )
    if kraken_ocr is None:  # pragma: no cover - defensive
        raise RuntimeError("Kraken OCR helper was not initialised")
    with tempfile.TemporaryDirectory() as tmp:
        out_txt = Path(tmp) / "ocr.txt"
        kraken_ocr(image_path, model_path, out_txt)
        return out_txt.read_text(encoding="utf-8").strip()


def run_refinement(
    image_paths: Iterable[Path],
    *,
    transcriber: GPTTranscriber,
    engine: str,
    tesseract_model: Optional[Path] = None,
    tessdata_dir: Optional[Path] = None,
    psm: int = 6,
    kraken_model: Optional[Path] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> List[RefinementResult]:
    """Refine ``image_paths`` using baseline OCR plus GPT cleanup."""

    results: List[RefinementResult] = []
    for image_path in image_paths:
        log.debug("Preparing baseline OCR for %s", image_path)
        tokens = ocr_detailed(
            image_path,
            model_path=tesseract_model,
            tessdata_dir=tessdata_dir,
            psm=psm,
        )
        tokens_payload = _token_payload(tokens)
        if engine == "kraken":
            if kraken_model is None:
                raise RuntimeError("--kraken-model is required when --engine kraken is selected")
            rough_text = _run_kraken_ocr(image_path, Path(kraken_model))
        else:
            rough_text = _extract_text(tokens)

        hint_text = _build_hint(engine, rough_text, tokens_payload)
        response = transcriber.generate(
            image_path,
            hint_text=hint_text,
            response_format=_RESPONSE_FORMAT,
            temperature=temperature,
            max_output_tokens=max_output_tokens or transcriber.max_output_tokens,
        )
        raw = response.output_text.strip()
        if not raw:
            raise GPTTranscriptionError("ChatGPT returned an empty refinement response")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GPTTranscriptionError("Failed to parse ChatGPT refinement JSON") from exc

        corrected = str(parsed.get("corrected_text", "")).strip()
        if not corrected:
            raise GPTTranscriptionError("ChatGPT refinement did not include corrected_text")
        confidence = _normalise_confidence(parsed.get("confidence"))
        notes = parsed.get("notes")

        results.append(
            RefinementResult(
                image=Path(image_path),
                engine=engine,
                corrected_text=corrected,
                confidence=confidence,
                rough_text=rough_text,
                tokens=tokens_payload,
                notes=notes if isinstance(notes, str) and notes.strip() else None,
            )
        )
    return results


__all__ = [
    "DEFAULT_REFINE_PROMPT",
    "RefinementResult",
    "run_refinement",
]
