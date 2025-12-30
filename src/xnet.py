"""Goal-oriented XNet controller for OCR pipeline orchestration.

This module implements a minimal, non-redundant controller that sequences
PREPROCESS → SEGMENT → RECOGNIZE → CHECK → REPAIR with bounded retries.

It reuses existing components:
- Preprocessing: src.preprocessing
- Segmentation: src.kraken_adapter (writes line crops and optional PAGE-XML)
- Recognition: src.ocr (Tesseract) and optional GPT fallback via src.gpt_ocr
- Export/line structures: src.exporters and src.line_store

The controller is designed to be pluggable: segmenters/recognizers/checkers
are defined via simple Protocols so future U-Net segmenters can drop in
without changing orchestration logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
import difflib
import json
import statistics
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

from PIL import Image

from .preprocessing import load_normalized_image
from .exporters import save_line_crops
from .line_store import Line
from .ocr import ocr_detailed

try:
    from .gpt_ocr import GPTTranscriber  # optional
except Exception:  # pragma: no cover - allow running without OpenAI deps
    GPTTranscriber = None  # type: ignore

from .kraken_adapter import (
    DeskewConfig,
    is_available as kraken_available,
    ocr_line_to_string as kraken_ocr_line_to_string,
    segment_pages_with_kraken,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols (extensible adapters)
# ---------------------------------------------------------------------------


class Segmenter(Protocol):
    def segment(self, image_path: Path, output_dir: Path) -> List[Line]:
        ...


class RecognizedLine:
    def __init__(
        self,
        line: Line,
        text: str,
        confidence: float,
        crop_path: Optional[Path] = None,
        page_bbox: Optional[Tuple[int, int, int, int]] = None,
        tokens: Optional[List[dict]] = None,
        member_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> None:
        self.line = line
        self.text = text
        self.confidence = confidence
        self.crop_path = crop_path
        self.page_bbox = page_bbox
        self.tokens = tokens or []
        self.member_bboxes = member_bboxes


class Recognizer(Protocol):
    def recognize(self, crops_dir: Path) -> List[RecognizedLine]:
        ...


class Checker(Protocol):
    def check(self, lines: Sequence[RecognizedLine], language_hint: Optional[str] = None) -> float:
        ...


# ---------------------------------------------------------------------------
# Default adapters (Kraken + Tesseract)
# ---------------------------------------------------------------------------


@dataclass
class KrakenSegmenter:
    model: Optional[str] = None
    pagexml_dir: Optional[Path] = None
    padding: int = 12
    min_width: int = 24
    min_height: int = 16
    deskew: bool = True
    deskew_max_angle: float = 25.0
    force_landscape: bool = True
    force_upright: bool = True

    def segment(self, image_path: Path, output_dir: Path) -> List[Line]:
        output_dir.mkdir(parents=True, exist_ok=True)

        deskew_config = None
        if self.deskew:
            deskew_config = DeskewConfig(
                enabled=True,
                max_skew=self.deskew_max_angle,
                force_landscape=self.force_landscape,
                force_upright=self.force_upright,
            )

        # Run Kraken to write line crops; returns stats but lines are in files.
        segment_pages_with_kraken(
            images=[image_path],
            output_dir=output_dir,
            model=self.model,
            pagexml_dir=self.pagexml_dir,
            padding=self.padding,
            min_width=self.min_width,
            min_height=self.min_height,
            global_cli_args=None,
            segment_cli_args=None,
            filter_config=None,
            filter_use_gpt=False,
            filter_gpt_model=None,
            filter_gpt_prompt=None,
            filter_gpt_cache_dir=None,
            deskew_config=deskew_config,
        )

        # Convert created crops to Line stubs using EXIF metadata when available
        lines: List[Line] = []
        index = 1
        for png in sorted(output_dir.glob(f"{image_path.stem}_line*.png")):
            try:
                with Image.open(png) as im:
                    exif = im.getexif()
                # Minimal fallback bbox derives from crop size when EXIF payload missing
                width, height = Image.open(png).size
                bbox = (0, 0, width, height)
            except Exception:
                width, height = 0, 0
                bbox = (0, 0, 1, 1)

            line = Line(
                id=index,
                baseline=[],
                bbox=bbox,
                text="",
                order_key=(index, 1, 1, 1, 1),
                selected=False,
                is_manual=False,
            )
            lines.append(line)
            index += 1

        return lines


@dataclass
class TesseractRecognizer:
    model_path: Optional[Path] = None
    tessdata_dir: Optional[Path] = None
    psm: int = 7
    adaptive: bool = True
    resize_width: int = 1800
    use_gpt_fallback: bool = False
    gpt_model: Optional[str] = None
    gpt_cache_dir: Optional[Path] = None
    max_gpt_calls_per_page: int = 10
    _warned_no_openai: bool = False

    def _avg_confidence(self, detailed_df) -> float:
        if detailed_df is None or detailed_df.empty:
            return 0.0
        confidences = detailed_df.get("confidence")
        if confidences is None:
            return 0.0
        values = confidences.dropna().astype(float).tolist()
        return float(sum(values) / len(values)) if values else 0.0

    def recognize(self, crops_dir: Path) -> List[RecognizedLine]:
        results: List[RecognizedLine] = []
        gpt_calls = 0
        transcriber: Optional[object] = None

        if self.use_gpt_fallback and GPTTranscriber is None and not self._warned_no_openai:
            self._warned_no_openai = True
            log.warning("--gpt-fallback requested but OpenAI client is unavailable (install 'openai').")

        if self.use_gpt_fallback and GPTTranscriber is not None:
            try:
                transcriber = GPTTranscriber(model=self.gpt_model or "gpt-4o-mini", cache_dir=self.gpt_cache_dir)
            except Exception as exc:  # pragma: no cover - GPT optional
                log.info("GPT transcriber init failed: %s", exc)
                transcriber = None

        crop_index = 0
        for crop_png in sorted(crops_dir.glob("*_line*.png")):
            crop_index += 1

            page_bbox: Optional[Tuple[int, int, int, int]] = None
            try:
                meta_path = crop_png.with_suffix(".boxes.json")
                if meta_path.exists():
                    import json

                    data = json.loads(meta_path.read_text(encoding="utf8"))
                    bbox_src = data.get("bbox_original") or data.get("bbox")
                    if isinstance(bbox_src, dict):
                        left = int(bbox_src.get("left"))
                        top = int(bbox_src.get("top"))
                        right = int(bbox_src.get("right"))
                        bottom = int(bbox_src.get("bottom"))
                        page_bbox = (left, top, right, bottom)
            except Exception:
                page_bbox = None

            detailed = ocr_detailed(
                crop_png,
                model_path=self.model_path,
                tessdata_dir=self.tessdata_dir,
                psm=self.psm,
                force_landscape=False,
                adaptive=self.adaptive,
                resize_width=self.resize_width,
            )
            avg_conf = self._avg_confidence(detailed)

            token_geoms: List[dict] = []
            try:
                if detailed is not None and not detailed.empty:
                    # Persist lightweight token geometry for downstream repair decisions.
                    token_df = detailed.copy()
                    token_df["text"] = token_df["text"].fillna("").astype(str).str.strip()
                    token_df = token_df[token_df["text"].ne("")]
                    for _, row in token_df.iterrows():
                        token_geoms.append(
                            {
                                "text": str(row.get("text", "")),
                                "conf": float(row.get("confidence", 0.0)),
                                "left": int(row.get("left", 0)),
                                "top": int(row.get("top", 0)),
                                "width": int(row.get("width", 0)),
                                "height": int(row.get("height", 0)),
                                "line_num": int(row.get("line_num", 0)),
                                "word_num": int(row.get("word_num", 0)),
                            }
                        )
            except Exception:
                token_geoms = []
            text = ""
            if not detailed.empty:
                # Group by line to form text for this crop
                cols = ["page_num", "block_num", "par_num", "line_num", "word_num"]
                token_text_df = detailed.copy()
                token_text_df["text"] = token_text_df["text"].fillna("").astype(str).str.strip()
                token_text_df = token_text_df[token_text_df["text"].ne("")]
                if not token_text_df.empty:
                    sort_cols = cols
                    token_text_df = token_text_df.sort_values(sort_cols)
                    grouped = token_text_df.groupby(sort_cols[:-1], sort=True)["text"].apply(lambda words: " ".join(words))
                    text = "\n".join(grouped.tolist()).strip()

            # Optional GPT fallback for stubborn lines
            if (
                self.use_gpt_fallback
                and transcriber is not None
                and avg_conf < 50.0
                and gpt_calls < self.max_gpt_calls_per_page
            ):
                try:
                    log.info("GPT fallback (line) for %s (conf=%.1f)", crop_png.name, avg_conf)
                    gpt_calls += 1
                    gpt_text = transcriber.transcribe(crop_png, hint_text=text or None)
                    if gpt_text and (not text or len(gpt_text) > len(text)):
                        text = gpt_text.strip()
                        avg_conf = max(avg_conf, 55.0)  # heuristic floor when GPT intervenes
                except Exception as exc:  # pragma: no cover - GPT optional
                    log.info("GPT fallback (line) failed for %s: %s", crop_png.name, exc)

            # Construct a Line placeholder corresponding to this crop
            width, height = Image.open(crop_png).size
            line = Line(
                id=crop_index,
                baseline=[],
                bbox=(0, 0, width, height),
                text=text,
                order_key=(crop_index, 1, 1, 1, 1),
                selected=False,
                is_manual=False,
            )
            results.append(
                RecognizedLine(
                    line=line,
                    text=text,
                    confidence=avg_conf,
                    crop_path=crop_png,
                    page_bbox=page_bbox,
                    tokens=token_geoms,
                    member_bboxes=[page_bbox] if page_bbox is not None else None,
                )
            )

        return results


@dataclass
class KrakenRecognizer:
    model_path: Path
    confidence_floor: float = 60.0

    def recognize(self, crops_dir: Path) -> List[RecognizedLine]:
        if not kraken_available():
            raise RuntimeError(
                "Kraken is not installed/available. Install it with 'pip install kraken[serve]' and ensure the 'kraken' command is on PATH."
            )

        results: List[RecognizedLine] = []
        crop_index = 0
        for crop_png in sorted(crops_dir.glob("*_line*.png")):
            crop_index += 1

            page_bbox: Optional[Tuple[int, int, int, int]] = None
            try:
                meta_path = crop_png.with_suffix(".boxes.json")
                if meta_path.exists():
                    data = json.loads(meta_path.read_text(encoding="utf8"))
                    bbox_src = data.get("bbox_original") or data.get("bbox")
                    if isinstance(bbox_src, dict):
                        page_bbox = (
                            int(bbox_src.get("left")),
                            int(bbox_src.get("top")),
                            int(bbox_src.get("right")),
                            int(bbox_src.get("bottom")),
                        )
            except Exception:
                page_bbox = None

            try:
                # Prefer in-process Kraken API to avoid intermittent Windows
                # temp-file cleanup failures when invoking kraken.exe.
                text = kraken_ocr_line_to_string(crop_png, self.model_path).strip()
            except Exception as exc:
                log.info("Kraken OCR failed for %s: %s", crop_png.name, exc)
                text = ""

            try:
                width, height = Image.open(crop_png).size
            except Exception:
                width, height = 0, 0

            line = Line(
                id=crop_index,
                baseline=[],
                bbox=(0, 0, width, height),
                text=text,
                order_key=(crop_index, 1, 1, 1, 1),
                selected=False,
                is_manual=False,
            )
            results.append(
                RecognizedLine(
                    line=line,
                    text=text,
                    confidence=float(self.confidence_floor),
                    crop_path=crop_png,
                    page_bbox=page_bbox,
                    tokens=[],
                    member_bboxes=[page_bbox] if page_bbox is not None else None,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Simple language/coherence checker
# ---------------------------------------------------------------------------


class SimpleChecker:
    def __init__(self, min_alpha_ratio: float = 0.6, min_vowel_ratio: float = 0.15) -> None:
        self.min_alpha_ratio = min_alpha_ratio
        self.min_vowel_ratio = min_vowel_ratio

    def check(self, lines: Sequence[RecognizedLine], language_hint: Optional[str] = None) -> float:
        text = "\n".join(line.text for line in lines if line.text)
        if not text:
            return 0.0
        total = len(text)
        letters = sum(1 for ch in text if ch.isalpha())
        spaces = sum(1 for ch in text if ch.isspace())
        vowels = sum(1 for ch in text.lower() if ch in "aeiou")
        alpha_ratio = letters / max(1, total)
        vowel_ratio = vowels / max(1, letters)
        space_ratio = spaces / max(1, total)
        score = 0.0
        if alpha_ratio >= self.min_alpha_ratio:
            score += 40.0
        if vowel_ratio >= self.min_vowel_ratio:
            score += 30.0
        if 0.05 <= space_ratio <= 0.3:
            score += 30.0
        return score


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@dataclass
class XNetConfig:
    min_confidence: float = 55.0
    min_coherence: float = 60.0
    min_similarity: float = 0.6
    max_retries: int = 2
    auto_tune: bool = True
    # segmentation/recognition adapters
    segmenter: Optional[Segmenter] = None
    recognizer: Optional[Recognizer] = None
    checker: Optional[Checker] = None
    # Optional page-level ground truth directory: expects <stem>.gt.txt files
    ground_truth_page_dir: Optional[Path] = None
    # Write per-page report.json/report.txt to output_dir
    write_report: bool = False


@dataclass
class XNetResult:
    image: Path
    lines: List[RecognizedLine]
    coherence: float
    avg_confidence: float
    similarity: Optional[float]
    accepted: bool
    attempts: int


class XNetController:
    def __init__(self, config: XNetConfig) -> None:
        self.config = config
        # Default adapters if not provided
        self.segmenter = config.segmenter or KrakenSegmenter()
        self.recognizer = config.recognizer or TesseractRecognizer()
        self.checker = config.checker or SimpleChecker()

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split()).strip().lower()

    def _load_gt_lines(self, page_stem: str) -> Optional[List[str]]:
        gt_dir = self.config.ground_truth_page_dir
        if gt_dir is None:
            return None
        gt_path = Path(gt_dir) / f"{page_stem}.gt.txt"
        if not gt_path.exists():
            return None
        try:
            raw = gt_path.read_text(encoding="utf8", errors="ignore")
        except Exception:
            return None
        # One physical line per newline. Drop empty lines, strip trailing whitespace.
        lines = [ln.rstrip() for ln in raw.replace("\r\n", "\n").split("\n")]
        lines = [ln for ln in lines if ln.strip()]
        return lines

    def _line_similarity(self, a: str, b: str) -> float:
        a_norm = self._normalize_text(a)
        b_norm = self._normalize_text(b)
        if not a_norm or not b_norm:
            return 0.0
        return float(difflib.SequenceMatcher(None, a_norm, b_norm).ratio())

    def _read_page_bbox_for_crop(self, crop_path: Path) -> Optional[Tuple[int, int, int, int]]:
        try:
            meta_path = crop_path.with_suffix(".boxes.json")
            if not meta_path.exists():
                return None
            import json

            data = json.loads(meta_path.read_text(encoding="utf8"))
            bbox_src = data.get("bbox_original") or data.get("bbox")
            if not isinstance(bbox_src, dict):
                return None
            left = int(bbox_src.get("left"))
            top = int(bbox_src.get("top"))
            right = int(bbox_src.get("right"))
            bottom = int(bbox_src.get("bottom"))
            return (left, top, right, bottom)
        except Exception:
            return None

    def _union_bbox(self, boxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        left = min(b[0] for b in boxes)
        top = min(b[1] for b in boxes)
        right = max(b[2] for b in boxes)
        bottom = max(b[3] for b in boxes)
        return (left, top, right, bottom)

    def _partition_boxes_by_y(
        self,
        boxes: Sequence[Tuple[int, int, int, int]],
        k: int,
    ) -> List[List[Tuple[int, int, int, int]]]:
        if k <= 1:
            return [list(boxes)]
        ordered = sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2.0, b[0]))
        n = len(ordered)
        if n == 0:
            return []
        if n < k:
            # Can't create more non-empty groups than items.
            return [list(ordered)]
        # Split into k contiguous groups in y-order.
        groups: List[List[Tuple[int, int, int, int]]] = []
        for i in range(k):
            start = (n * i) // k
            end = (n * (i + 1)) // k
            if start == end:
                continue
            groups.append(ordered[start:end])
        return groups if groups else [list(ordered)]

    def _best_text_line_for_gt(self, gt_line: str, text: str) -> str:
        if not text:
            return ""
        parts = [p.strip() for p in str(text).splitlines() if p.strip()]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return max(parts, key=lambda p: self._line_similarity(gt_line, p))

    def _reorder_by_gt_similarity(
        self,
        gt_lines: Sequence[str],
        recognized: List[RecognizedLine],
        *,
        max_passes: int = 3,
        min_gain: float = 0.01,
    ) -> List[RecognizedLine]:
        if len(gt_lines) != len(recognized) or len(gt_lines) < 2:
            return recognized

        def sim(i: int, j: int) -> float:
            return self._line_similarity(gt_lines[i], recognized[j].text)

        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(len(recognized) - 1):
                s1 = sim(i, i) + sim(i + 1, i + 1)
                s2 = sim(i, i + 1) + sim(i + 1, i)
                if s2 > s1 + min_gain:
                    recognized[i], recognized[i + 1] = recognized[i + 1], recognized[i]
                    improved = True

        # Re-assign sequential ids/order_keys after reordering.
        for i, rl in enumerate(recognized, start=1):
            rl.line.id = i
            rl.line.order_key = (i, 1, 1, 1, 1)
        return recognized

    def _refine_line_by_split_drop(
        self,
        *,
        gt_line: str,
        recognized_line: RecognizedLine,
        page_image_path: Path,
        crops_dir: Path,
        name_prefix: str,
        baseline_sim: float,
        min_gain: float = 0.03,
        max_splits: int = 3,
        min_boxes: int = 6,
    ) -> RecognizedLine:
        boxes: Optional[List[Tuple[int, int, int, int]]] = None
        if recognized_line.member_bboxes:
            boxes = list(recognized_line.member_bboxes)
        elif recognized_line.page_bbox is not None:
            boxes = [recognized_line.page_bbox]

        if not boxes or len(boxes) < min_boxes:
            return recognized_line

        best_sim = baseline_sim
        best_rl = recognized_line
        best_boxes: Optional[List[Tuple[int, int, int, int]]] = None

        for k in range(2, max_splits + 1):
            groups = self._partition_boxes_by_y(boxes, k=k)
            if len(groups) <= 1:
                continue
            for g_idx, group in enumerate(groups, start=1):
                if len(group) < max(2, min_boxes // 3):
                    continue
                union = self._robust_union_bbox(group, image_size=None, pad=12, trim_ratio=0.12)
                candidate = self._ocr_page_bbox(
                    page_image_path=page_image_path,
                    bbox=union,
                    crops_dir=crops_dir,
                    name=f"{name_prefix}.split{k}.{g_idx:02d}.png",
                )
                best_text = self._best_text_line_for_gt(gt_line, candidate.text)
                candidate.text = best_text
                candidate.line.text = best_text
                s = self._line_similarity(gt_line, best_text)
                if s > best_sim + min_gain:
                    best_sim = s
                    best_rl = candidate
                    best_boxes = list(group)

        if best_rl is recognized_line:
            return recognized_line

        # Preserve identity/order and store retained sub-boxes.
        best_rl.line.id = recognized_line.line.id
        best_rl.line.order_key = recognized_line.line.order_key
        best_rl.member_bboxes = best_boxes
        best_rl.page_bbox = recognized_line.page_bbox  # keep original union bbox for context
        return best_rl

    def _robust_union_bbox(
        self,
        boxes: Sequence[Tuple[int, int, int, int]],
        *,
        image_size: Optional[Tuple[int, int]] = None,
        pad: int = 10,
        trim_ratio: float = 0.1,
    ) -> Tuple[int, int, int, int]:
        """Union bbox with trimmed top/bottom to reduce outlier influence.

        Kraken crops can include occasional tall/shifted boxes that cause a naive
        min(top)/max(bottom) union to bleed into neighboring lines. We keep
        min(left)/max(right) (safe for text width), but trim top/bottom.
        """

        if not boxes:
            return (0, 0, 1, 1)

        lefts = [b[0] for b in boxes]
        rights = [b[2] for b in boxes]
        tops = sorted([b[1] for b in boxes])
        bottoms = sorted([b[3] for b in boxes])

        left = min(lefts)
        right = max(rights)

        n = len(boxes)
        trim = float(trim_ratio)
        if trim < 0.0:
            trim = 0.0
        if trim > 0.45:
            trim = 0.45
        lo = int(n * trim)
        hi = int(n * (1.0 - trim)) - 1
        lo = max(0, min(lo, n - 1))
        hi = max(0, min(hi, n - 1))
        if lo > hi:
            lo, hi = 0, n - 1

        top = tops[lo]
        bottom = bottoms[hi]
        if top >= bottom:
            # Fallback to naive union if trimming becomes degenerate.
            top = min(tops)
            bottom = max(bottoms)

        left -= pad
        top -= pad
        right += pad
        bottom += pad

        if image_size is not None:
            w, h = image_size
            left = max(0, min(left, w - 1))
            top = max(0, min(top, h - 1))
            right = max(1, min(right, w))
            bottom = max(1, min(bottom, h))
            if right <= left:
                right = min(w, left + 1)
            if bottom <= top:
                bottom = min(h, top + 1)

        return (int(left), int(top), int(right), int(bottom))

    def _cluster_by_y(
        self,
        items: Sequence[RecognizedLine],
        *,
        y_tol: Optional[int] = None,
        min_vertical_overlap: float = 0.25,
        target_clusters: Optional[int] = None,
    ) -> List[List[RecognizedLine]]:
        with_bbox = [rl for rl in items if rl.page_bbox is not None]
        if not with_bbox:
            return [list(items)]

        def mid_y(rl: RecognizedLine) -> float:
            left, top, right, bottom = rl.page_bbox  # type: ignore[misc]
            return (top + bottom) / 2.0

        def height(rl: RecognizedLine) -> int:
            left, top, right, bottom = rl.page_bbox  # type: ignore[misc]
            return max(1, int(bottom) - int(top))

        # Choose a tolerance scaled to the typical crop height.
        # Kraken can produce many narrow/fragment crops whose mid_y varies
        # more than a fixed small pixel threshold, but we must avoid
        # accidentally chaining clusters across multiple physical lines.
        if y_tol is None:
            try:
                med_h = int(statistics.median([height(rl) for rl in with_bbox]))
            except Exception:
                med_h = 0
            # Empirically, ~0.35× crop height works better than a constant.
            y_tol = max(22, int(med_h * 0.35) if med_h > 0 else 22)

        def vertical_overlap_ratio(a: RecognizedLine, b: RecognizedLine) -> float:
            a_left, a_top, a_right, a_bottom = a.page_bbox  # type: ignore[misc]
            b_left, b_top, b_right, b_bottom = b.page_bbox  # type: ignore[misc]
            overlap = max(0, min(a_bottom, b_bottom) - max(a_top, b_top))
            denom = max(1, min(a_bottom - a_top, b_bottom - b_top))
            return float(overlap) / float(denom)

        ordered = sorted(with_bbox, key=mid_y)
        clusters: List[List[RecognizedLine]] = []
        current: List[RecognizedLine] = []
        current_mid: Optional[float] = None
        for rl in ordered:
            y = mid_y(rl)
            if not current:
                current = [rl]
                current_mid = y
                continue
            # Midpoint closeness is the primary signal.
            # Vertical overlap is only allowed as a weak fallback when y is
            # not too far away, to avoid chain-merging distinct lines.
            close_enough = current_mid is not None and abs(y - current_mid) <= y_tol
            overlaps_enough = (
                current_mid is not None
                and abs(y - current_mid) <= (y_tol * 2)
                and vertical_overlap_ratio(current[-1], rl) >= min_vertical_overlap
            )
            if close_enough or overlaps_enough:
                current.append(rl)
                current_mid = (current_mid * (len(current) - 1) + y) / len(current)
            else:
                clusters.append(current)
                current = [rl]
                current_mid = y
        if current:
            clusters.append(current)

        # If GT is available, we can often do better than a fixed tolerance by
        # collapsing oversplit clusters down to the expected number of physical
        # lines. This only merges adjacent clusters (never reorders or splits),
        # so it stays consistent with top-to-bottom reading order.
        if target_clusters is not None:
            try:
                target = int(target_clusters)
            except Exception:
                target = 0
            if target > 0 and len(clusters) > target:
                def cluster_mid(c: Sequence[RecognizedLine]) -> float:
                    ys = [mid_y(rl) for rl in c]
                    return float(statistics.median(ys)) if ys else 0.0

                # Iteratively merge the closest adjacent pair by y-midpoint.
                while len(clusters) > target and len(clusters) >= 2:
                    mids = [cluster_mid(c) for c in clusters]
                    gaps = [abs(mids[i + 1] - mids[i]) for i in range(len(mids) - 1)]
                    if not gaps:
                        break
                    i = int(min(range(len(gaps)), key=lambda k: gaps[k]))
                    merged = clusters[i] + clusters[i + 1]
                    clusters[i] = merged
                    del clusters[i + 1]

        return clusters

    def _text_from_detailed(self, detailed) -> str:
        if detailed is None or getattr(detailed, "empty", True):
            return ""
        cols = ["page_num", "block_num", "par_num", "line_num", "word_num"]
        tokens = detailed.copy()
        tokens["text"] = tokens["text"].fillna("").astype(str).str.strip()
        tokens = tokens[tokens["text"].ne("")]
        if tokens.empty:
            return ""
        tokens = tokens.sort_values(cols)
        grouped = tokens.groupby(cols[:-1], sort=True)["text"].apply(lambda words: " ".join(words))
        return "\n".join(grouped.tolist()).strip()

    def _tokens_from_detailed(self, detailed) -> List[dict]:
        if detailed is None or getattr(detailed, "empty", True):
            return []
        tokens: List[dict] = []
        try:
            df = detailed.copy()
            df["text"] = df["text"].fillna("").astype(str).str.strip()
            df = df[df["text"].ne("")]
            if df.empty:
                return []
            for _, row in df.iterrows():
                tokens.append(
                    {
                        "text": str(row.get("text", "")),
                        "conf": float(row.get("confidence", 0.0)),
                        "left": int(row.get("left", 0)),
                        "top": int(row.get("top", 0)),
                        "width": int(row.get("width", 0)),
                        "height": int(row.get("height", 0)),
                        "line_num": int(row.get("line_num", 0)),
                        "word_num": int(row.get("word_num", 0)),
                    }
                )
        except Exception:
            return []
        return tokens

    def _token_line_bboxes_in_crop(self, tokens: Sequence[dict], *, min_tokens_per_line: int = 2) -> List[Tuple[int, int, int, int]]:
        by_line: dict[int, List[dict]] = {}
        for t in tokens:
            try:
                ln = int(t.get("line_num", 0))
            except Exception:
                ln = 0
            if ln <= 0:
                continue
            by_line.setdefault(ln, []).append(t)

        bboxes: List[Tuple[int, int, int, int]] = []
        for ln in sorted(by_line.keys()):
            items = by_line[ln]
            if len(items) < min_tokens_per_line:
                continue
            left = min(int(it.get("left", 0)) for it in items)
            top = min(int(it.get("top", 0)) for it in items)
            right = max(int(it.get("left", 0)) + int(it.get("width", 0)) for it in items)
            bottom = max(int(it.get("top", 0)) + int(it.get("height", 0)) for it in items)
            if right > left and bottom > top:
                bboxes.append((left, top, right, bottom))

        # Sort top-to-bottom
        bboxes.sort(key=lambda b: ((b[1] + b[3]) / 2.0, b[0]))
        return bboxes

    def _split_line_by_token_lines(
        self,
        *,
        rl: RecognizedLine,
        page_image_path: Path,
        crops_dir: Path,
        name_prefix: str,
        pad: int = 10,
        min_line_gap: int = 18,
    ) -> List[RecognizedLine]:
        if not rl.tokens or rl.page_bbox is None:
            return [rl]

        crop_line_boxes = self._token_line_bboxes_in_crop(rl.tokens)
        if len(crop_line_boxes) <= 1:
            return [rl]

        # Require a meaningful vertical separation between detected token-lines.
        gaps = [crop_line_boxes[i + 1][1] - crop_line_boxes[i][3] for i in range(len(crop_line_boxes) - 1)]
        if not gaps or max(gaps) < min_line_gap:
            return [rl]

        page_left, page_top, _, _ = rl.page_bbox
        out: List[RecognizedLine] = []
        for idx, (l, t, r, b) in enumerate(crop_line_boxes, start=1):
            sub = (page_left + l - pad, page_top + t - pad, page_left + r + pad, page_top + b + pad)
            split_rl = self._ocr_page_bbox(
                page_image_path=page_image_path,
                bbox=sub,
                crops_dir=crops_dir,
                name=f"{name_prefix}.tokline{idx:02d}.png",
            )
            out.append(split_rl)

        # Keep reading order
        out.sort(key=lambda x: ((x.page_bbox[1] + x.page_bbox[3]) / 2.0, x.page_bbox[0]) if x.page_bbox else (0.0, 0.0))
        return out if out else [rl]

    def _avg_conf_from_detailed(self, detailed) -> float:
        if detailed is None or getattr(detailed, "empty", True):
            return 0.0
        confidences = detailed.get("confidence")
        if confidences is None:
            return 0.0
        values = confidences.dropna().astype(float).tolist()
        return float(sum(values) / len(values)) if values else 0.0

    def _ocr_page_bbox(
        self,
        *,
        page_image_path: Path,
        bbox: Tuple[int, int, int, int],
        crops_dir: Path,
        name: str,
    ) -> RecognizedLine:
        left, top, right, bottom = bbox
        with Image.open(page_image_path) as im:
            w, h = im.size
            left = max(0, min(int(left), w - 1))
            top = max(0, min(int(top), h - 1))
            right = max(1, min(int(right), w))
            bottom = max(1, min(int(bottom), h))
            if right <= left:
                right = min(w, left + 1)
            if bottom <= top:
                bottom = min(h, top + 1)
            crop = im.crop((left, top, right, bottom))

        out_path = crops_dir / name
        crop.save(out_path)

        if not isinstance(self.recognizer, TesseractRecognizer):
            text = ""
            conf = 0.0
        else:
            detailed = ocr_detailed(
                out_path,
                model_path=self.recognizer.model_path,
                tessdata_dir=self.recognizer.tessdata_dir,
                psm=self.recognizer.psm,
                force_landscape=False,
                adaptive=self.recognizer.adaptive,
                resize_width=self.recognizer.resize_width,
            )
            text = self._text_from_detailed(detailed)
            conf = self._avg_conf_from_detailed(detailed)
            tokens = self._tokens_from_detailed(detailed)

        line = Line(
            id=1,
            baseline=[],
            bbox=(0, 0, max(1, right - left), max(1, bottom - top)),
            text=text,
            order_key=(1, 1, 1, 1, 1),
            selected=False,
            is_manual=False,
        )
        return RecognizedLine(
            line=line,
            text=text,
            confidence=conf,
            crop_path=out_path,
            page_bbox=bbox,
            tokens=tokens if isinstance(self.recognizer, TesseractRecognizer) else None,
            member_bboxes=[bbox],
        )

    def _align_gt_to_ocr_groups(
        self,
        *,
        gt_lines: Sequence[str],
        ocr_lines: Sequence[str],
        max_merge: int = 4,
    ) -> List[Tuple[int, int]]:
        """Return list of (start,end) spans into ocr_lines for each gt line."""

        m = len(gt_lines)
        n = len(ocr_lines)
        if m == 0 or n == 0:
            return []

        # dp[i][j] = best score aligning first i gt lines using first j ocr lines
        neg_inf = -1e9
        dp = [[neg_inf] * (n + 1) for _ in range(m + 1)]
        back: List[List[Optional[Tuple[int, int]]]] = [[None] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0

        def merge_text(span: Sequence[str]) -> str:
            return " ".join([s.strip() for s in span if s.strip()]).strip()

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                best = neg_inf
                best_k: Optional[Tuple[int, int]] = None
                for k in range(1, min(max_merge, j) + 1):
                    prev = dp[i - 1][j - k]
                    if prev <= neg_inf / 2:
                        continue
                    merged = merge_text(ocr_lines[j - k : j])
                    sim = self._line_similarity(gt_lines[i - 1], merged)
                    penalty = 0.02 * float(k - 1)
                    score = prev + sim - penalty
                    if score > best:
                        best = score
                        best_k = (j - k, j)
                dp[i][j] = best
                back[i][j] = best_k

        # Choose best ending j (allow leftover OCR lines)
        end_j = max(range(n + 1), key=lambda j: dp[m][j])
        spans: List[Tuple[int, int]] = []
        i = m
        j = end_j
        while i > 0 and j >= 0:
            step = back[i][j]
            if step is None:
                break
            start, end = step
            spans.append((start, end))
            i -= 1
            j = start
        spans.reverse()
        return spans

    def _page_similarity(self, page_text: str, page_stem: str) -> Optional[float]:
        gt_dir = self.config.ground_truth_page_dir
        if gt_dir is None:
            return None
        gt_path = Path(gt_dir) / f"{page_stem}.gt.txt"
        if not gt_path.exists():
            return None
        try:
            gt_text = gt_path.read_text(encoding="utf8", errors="ignore")
        except Exception:
            return None
        a = self._normalize_text(gt_text)
        b = self._normalize_text(page_text)
        if not a or not b:
            return 0.0
        return float(difflib.SequenceMatcher(None, a, b).ratio())

    def run(self, image_path: Path, *, output_dir: Path) -> XNetResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        # START + PREPROCESS: normalize orientation to avoid EXIF surprises
        normalized_image, meta = load_normalized_image(image_path)
        normalized_path = output_dir / f"{image_path.stem}.normalized.png"
        try:
            normalized_image.save(normalized_path)
        except Exception:
            # If saving fails, fall back to the original path
            normalized_path = image_path

        attempts = 0
        final_lines: List[RecognizedLine] = []
        coherence_score: float = 0.0
        final_avg_conf: float = 0.0
        final_similarity: Optional[float] = None
        accepted: bool = False

        attempt_reports: List[dict] = []

        # Candidate OCR variants explored across attempts.
        # For line crops, PSM=7 (single line) is usually best; we try fallbacks across retries.
        ocr_variants: List[tuple[int, bool, int]] = [
            # Single-line focus
            (7, True, 1800),
            (7, False, 1800),
            (7, True, 2200),
            (7, False, 2200),
            (7, True, 2600),

            # Multi-line crop fallback / weak segmentation
            (6, True, 1800),
            (6, False, 1800),
            (6, True, 2200),
            (6, False, 2200),
            (6, True, 2600),

            # Sparse text / isolated words (titles, “love”, “happy”, bullets)
            (11, True, 1800),
            (11, False, 1800),
            (12, True, 1800),
            (12, False, 1800),

            # Single text line (sometimes helps vs 7 depending on handwriting)
            (13, True, 1800),
            (13, False, 1800),
        ]

        best_candidate: tuple[float, float, Optional[float], str, List[RecognizedLine]] | None = None

        max_attempts = max(1, self.config.max_retries + 1)
        while attempts < max_attempts:
            attempts += 1
            # SEGMENT: write line crops to a temporary directory under output_dir
            crops_root = Path("data") / "train" / "lines" / image_path.stem / "crops"
            crops_dir = crops_root / f"attempt{attempts:02d}"
            crops_dir.mkdir(parents=True, exist_ok=True)

            segmented_lines = self.segmenter.segment(normalized_path, crops_dir)
            if not segmented_lines:
                log.info("No lines detected; falling back to whole-page OCR.")
                # Save a single full-page crop to reuse recognizer logic
                with Image.open(normalized_path) as im:
                    im.convert("L").save(crops_dir / f"{image_path.stem}_line01.png")

            # Optional: write metadata EXIF + .gt.txt (empty) for downstream tools
            try:
                save_line_crops(normalized_path, segmented_lines or [], crops_dir)
            except Exception:
                # Non-critical; crops already exist from Kraken
                pass

            # RECOGNIZE: OCR with average confidence
            variant_index: Optional[int] = None
            variant_cycle: Optional[int] = None
            if self.config.auto_tune and isinstance(self.recognizer, TesseractRecognizer):
                variant_index = (attempts - 1) % len(ocr_variants)
                variant_cycle = (attempts - 1) // len(ocr_variants)
                variant = ocr_variants[variant_index]
                self.recognizer.psm, self.recognizer.adaptive, self.recognizer.resize_width = variant
                log.info(
                    "XNet attempt %d/%d using Tesseract psm=%d adaptive=%s resize_width=%d",
                    attempts,
                    max_attempts,
                    self.recognizer.psm,
                    self.recognizer.adaptive,
                    self.recognizer.resize_width,
                )

            attempt_meta: dict = {
                "attempt": attempts,
                "max_attempts": max_attempts,
                "tesseract": {
                    "psm": int(self.recognizer.psm) if isinstance(self.recognizer, TesseractRecognizer) else None,
                    "adaptive": bool(self.recognizer.adaptive) if isinstance(self.recognizer, TesseractRecognizer) else None,
                    "resize_width": int(self.recognizer.resize_width) if isinstance(self.recognizer, TesseractRecognizer) else None,
                },
            }
            attempt_meta["tesseract"]["variant_index"] = int(variant_index) if variant_index is not None else None
            attempt_meta["tesseract"]["variant_cycle"] = int(variant_cycle) if variant_cycle is not None else None

            crop_paths = sorted(crops_dir.glob("*_line*.png"))
            gt_lines = self._load_gt_lines(image_path.stem)
            attempt_meta["num_crops"] = len(crop_paths)
            attempt_meta["gt_lines"] = len(gt_lines) if gt_lines is not None else None

            # Fast path: if Kraken produced many tiny crops, don't OCR each.
            # Instead, cluster by y-position (from .boxes.json) into physical lines and OCR those.
            if (
                isinstance(self.recognizer, TesseractRecognizer)
                and gt_lines is not None
                and len(crop_paths) > max(60, len(gt_lines) * 5)
            ):
                pseudo: List[RecognizedLine] = []
                for idx, crop_path in enumerate(crop_paths, start=1):
                    bbox = self._read_page_bbox_for_crop(crop_path)
                    if bbox is None:
                        continue
                    line = Line(
                        id=idx,
                        baseline=[],
                        bbox=(0, 0, 1, 1),
                        text="",
                        order_key=(idx, 1, 1, 1, 1),
                        selected=False,
                        is_manual=False,
                    )
                    pseudo.append(RecognizedLine(line=line, text="", confidence=0.0, crop_path=crop_path, page_bbox=bbox))

                clusters = self._cluster_by_y(pseudo, target_clusters=len(gt_lines) if gt_lines else None)
                attempt_meta["clustered"] = True
                attempt_meta["num_clusters"] = len(clusters)
                log.info(
                    "Clustering %d crops into %d line clusters (target=%s)",
                    len(pseudo),
                    len(clusters),
                    str(len(gt_lines)) if gt_lines else "n/a",
                )

                recognized = []
                for c_idx, cluster in enumerate(clusters, start=1):
                    boxes = [rl.page_bbox for rl in cluster if rl.page_bbox is not None]
                    if not boxes:
                        continue
                    union = self._robust_union_bbox(
                        boxes,  # type: ignore[arg-type]
                        image_size=normalized_image.size,
                        pad=12,
                        trim_ratio=0.12,
                    )
                    rl = self._ocr_page_bbox(
                        page_image_path=normalized_path,
                        bbox=union,
                        crops_dir=crops_dir,
                        name=f"{image_path.stem}.cluster{c_idx:03d}.png",
                    )
                    rl.line.id = c_idx
                    rl.line.order_key = (c_idx, 1, 1, 1, 1)
                    rl.member_bboxes = list(boxes)  # type: ignore[list-item]

                    # SPLIT: if this merged crop contains multiple token-level lines, split into real sub-crops.
                    split = self._split_line_by_token_lines(
                        rl=rl,
                        page_image_path=normalized_path,
                        crops_dir=crops_dir,
                        name_prefix=f"{image_path.stem}.cluster{c_idx:03d}",
                    )
                    recognized.extend(split)

                # Re-number after splits to keep stable ordering.
                recognized = sorted(
                    recognized,
                    key=lambda x: ((x.page_bbox[1] + x.page_bbox[3]) / 2.0, x.page_bbox[0]) if x.page_bbox else (0.0, 0.0),
                )
                for i, rl in enumerate(recognized, start=1):
                    rl.line.id = i
                    rl.line.order_key = (i, 1, 1, 1, 1)
            else:
                attempt_meta["clustered"] = False
                recognized = self.recognizer.recognize(crops_dir)
            # Order lines top-to-bottom. Prefer explicit lineNN index in filename; fallback to geometry.
            def _parse_index(path: Optional[Path]) -> Optional[int]:
                if not path:
                    return None
                name = path.stem
                # Expect patterns like <stem>_lineNN or <stem>.normalized_lineNN
                for token in name.split("_"):
                    if token.startswith("line"):
                        try:
                            return int(token.replace("line", ""))
                        except ValueError:
                            continue
                return None

            def _order_key(rl: RecognizedLine) -> Tuple[float, float, int]:
                # 1) Prefer explicit index parsed from filename
                idx = _parse_index(rl.crop_path)
                if idx is not None:
                    return (float(idx), 0.0, idx)

                # 2) Try sidecar JSON metadata from Kraken with original/page bboxes
                try:
                    if rl.crop_path:
                        meta_path = rl.crop_path.with_suffix(".boxes.json")
                        if meta_path.exists():
                            import json
                            data = json.loads(meta_path.read_text(encoding="utf8"))
                            bbox_src = data.get("bbox_original") or data.get("bbox")
                            if isinstance(bbox_src, dict):
                                left = int(bbox_src.get("left", rl.line.bbox[0]))
                                top = int(bbox_src.get("top", rl.line.bbox[1]))
                                right = int(bbox_src.get("right", rl.line.bbox[2]))
                                bottom = int(bbox_src.get("bottom", rl.line.bbox[3]))
                                mid_y = (top + bottom) / 2.0
                                return (mid_y, float(left), rl.line.id)
                except Exception:
                    pass

                # 3) Fallback: local geometry mid_y then left
                left, top, right, bottom = rl.line.bbox
                mid_y = (top + bottom) / 2.0
                return (mid_y, float(left), rl.line.id)

            recognized = sorted(recognized, key=_order_key)
            # Refresh sequential order_key after sorting
            for i, rl in enumerate(recognized, start=1):
                rl.line.order_key = (i, 1, 1, 1, 1)

            # ALIGN/REPAIR: if GT is available, align OCR lines to GT lines and merge/re-OCR where needed.
            if gt_lines is not None and recognized and isinstance(self.recognizer, TesseractRecognizer):
                ocr_texts = [rl.text for rl in recognized]
                spans = self._align_gt_to_ocr_groups(gt_lines=gt_lines, ocr_lines=ocr_texts, max_merge=4)
                if spans and len(spans) == len(gt_lines):
                    repaired: List[RecognizedLine] = []
                    for gt_idx, (start, end) in enumerate(spans, start=1):
                        group = recognized[start:end]
                        member_boxes: List[Tuple[int, int, int, int]] = []
                        for rl in group:
                            if rl.member_bboxes:
                                member_boxes.extend(rl.member_bboxes)
                            elif rl.page_bbox is not None:
                                member_boxes.append(rl.page_bbox)

                        if member_boxes:
                            union = self._robust_union_bbox(
                                member_boxes,
                                image_size=normalized_image.size,
                                pad=12,
                                trim_ratio=0.12,
                            )
                            merged = self._ocr_page_bbox(
                                page_image_path=normalized_path,
                                bbox=union,
                                crops_dir=crops_dir,
                                name=f"{image_path.stem}.aligned{gt_idx:03d}.png",
                            )
                            best_text = self._best_text_line_for_gt(gt_lines[gt_idx - 1], merged.text)
                            merged.text = best_text
                            merged.line.text = best_text
                            merged.line.id = gt_idx
                            merged.line.order_key = (gt_idx, 1, 1, 1, 1)
                            merged.member_bboxes = member_boxes
                            repaired.append(merged)
                        else:
                            # Fallback: keep first element if bbox missing
                            keep = group[0]
                            keep.line.id = gt_idx
                            keep.line.order_key = (gt_idx, 1, 1, 1, 1)
                            repaired.append(keep)
                    recognized = repaired

                    # REPAIR: optional local reorder + split/drop noise using GT feedback.
                    # 1) Try greedy adjacent swaps if they improve GT similarity.
                    recognized = self._reorder_by_gt_similarity(gt_lines, recognized)

                    # 2) For the worst-matching lines, try splitting their member boxes by y
                    # and keeping only the best-matching subcluster (drops outlier boxes).
                    base_sims = [self._line_similarity(gt_lines[i], recognized[i].text) for i in range(len(gt_lines))]
                    worst = sorted(range(len(gt_lines)), key=lambda i: base_sims[i])[: min(4, len(gt_lines))]
                    refined = 0
                    for i in worst:
                        before = base_sims[i]
                        updated = self._refine_line_by_split_drop(
                            gt_line=gt_lines[i],
                            recognized_line=recognized[i],
                            page_image_path=normalized_path,
                            crops_dir=crops_dir,
                            name_prefix=f"{image_path.stem}.refine{i+1:03d}",
                            baseline_sim=before,
                        )
                        after = self._line_similarity(gt_lines[i], updated.text)
                        if after > before + 0.03:
                            recognized[i] = updated
                            refined += 1
                    if refined:
                        log.info("Refined %d line(s) via split/drop repair", refined)
                    attempt_meta["refined_lines"] = refined

                    avg_line_sim = sum(
                        self._line_similarity(gt_lines[i], recognized[i].text)
                        for i in range(min(len(gt_lines), len(recognized)))
                    ) / max(1, min(len(gt_lines), len(recognized)))
                    log.info(
                        "Aligned to %d GT lines; avg line similarity=%.3f",
                        len(gt_lines),
                        avg_line_sim,
                    )
            if not recognized:
                log.warning("Recognition returned no lines for %s", image_path)
                final_lines = []
                coherence_score = 0.0
                break

            # CONTEXT CHECK: simple coherence gate
            coherence = self.checker.check(recognized)
            avg_conf = sum(r.confidence for r in recognized) / max(1, len(recognized))
            page_text = "\n".join(r.text for r in recognized if r.text).strip()

            page_similarity = self._page_similarity(page_text, image_path.stem)
            line_similarity: Optional[float] = None
            if gt_lines is not None and len(gt_lines) == len(recognized):
                sims = [self._line_similarity(gt_lines[i], recognized[i].text) for i in range(len(gt_lines))]
                line_similarity = float(sum(sims) / len(sims)) if sims else 0.0

            # Prefer line-level similarity when GT is line-based; fall back to page similarity.
            similarity = line_similarity if line_similarity is not None else page_similarity

            attempt_meta["metrics"] = {
                "avg_confidence": float(avg_conf),
                "coherence": float(coherence),
                "similarity_used": float(similarity) if similarity is not None else None,
                "similarity_page": float(page_similarity) if page_similarity is not None else None,
                "similarity_line": float(line_similarity) if line_similarity is not None else None,
                "chars": int(len(page_text)),
                "num_lines": int(len(recognized)),
            }
            if gt_lines is not None:
                per_line: List[dict] = []
                for i, rl in enumerate(recognized):
                    gt = gt_lines[i] if i < len(gt_lines) else None
                    per_line.append(
                        {
                            "index": i + 1,
                            "text": rl.text,
                            "confidence": float(rl.confidence),
                            "crop": str(rl.crop_path.name) if rl.crop_path else None,
                            "page_bbox": list(rl.page_bbox) if rl.page_bbox else None,
                            "gt": gt,
                            "line_similarity": float(self._line_similarity(gt, rl.text)) if gt is not None else None,
                        }
                    )
                attempt_meta["lines"] = per_line

            attempt_reports.append(attempt_meta)

            # Track best candidate even if it doesn't meet thresholds.
            if page_text:
                candidate_score = avg_conf + coherence + (100.0 * similarity if similarity is not None else 0.0)
                if best_candidate is None or candidate_score > best_candidate[0]:
                    best_candidate = (candidate_score, avg_conf, similarity, page_text, recognized)

            log.info(
                "Attempt metrics for %s: conf=%.1f coh=%.1f sim=%s (page=%s line=%s) chars=%d",
                image_path.name,
                avg_conf,
                coherence,
                f"{similarity:.3f}" if similarity is not None else "n/a",
                f"{page_similarity:.3f}" if page_similarity is not None else "n/a",
                f"{line_similarity:.3f}" if line_similarity is not None else "n/a",
                len(page_text),
            )

            if (
                page_text
                and avg_conf >= self.config.min_confidence
                and coherence >= self.config.min_coherence
                and (similarity is None or similarity >= self.config.min_similarity)
            ):
                final_lines = recognized
                coherence_score = coherence
                final_avg_conf = avg_conf
                final_similarity = similarity
                accepted = True
                break

            # REPAIR LOOP: adjust parameters and retry.
            if attempts < max_attempts:
                log.info(
                    "Repair loop: retrying after low metrics (conf=%.1f, coh=%.1f).",
                    avg_conf,
                    coherence,
                )
                if isinstance(self.segmenter, KrakenSegmenter):
                    self.segmenter.deskew_max_angle = min(45.0, self.segmenter.deskew_max_angle + 10.0)

        # Optional page-level GPT fallback when all attempts exhausted.
        if (
            (not final_lines)
            and isinstance(self.recognizer, TesseractRecognizer)
            and self.recognizer.use_gpt_fallback
            and GPTTranscriber is not None
        ):
            try:
                log.info("GPT fallback (page) for %s", image_path.name)
                transcriber = GPTTranscriber(
                    model=self.recognizer.gpt_model or "gpt-4o-mini",
                    cache_dir=self.recognizer.gpt_cache_dir,
                )
                hint = best_candidate[3] if best_candidate is not None else None
                gpt_text = (transcriber.transcribe(normalized_path, hint_text=hint) or "").strip()
                if gpt_text:
                    sim = self._page_similarity(gpt_text, image_path.stem)
                    if sim is None or sim >= self.config.min_similarity:
                        line = Line(
                            id=1,
                            baseline=[],
                            bbox=(0, 0, normalized_image.size[0], normalized_image.size[1]),
                            text=gpt_text,
                            order_key=(1, 1, 1, 1, 1),
                            selected=False,
                            is_manual=False,
                        )
                        final_lines = [RecognizedLine(line=line, text=gpt_text, confidence=60.0)]
                        coherence_score = self.checker.check(final_lines)
                        final_avg_conf = 60.0
                        final_similarity = sim
                        accepted = True
                    else:
                        log.info(
                            "GPT page-level text rejected by similarity (sim=%.3f < %.3f)",
                            sim,
                            self.config.min_similarity,
                        )
            except Exception as exc:  # pragma: no cover - GPT optional
                log.info("GPT page-level fallback failed for %s: %s", image_path.name, exc)

        # If nothing passed thresholds but we found a best candidate, keep it for inspection.
        if not final_lines and best_candidate is not None:
            _, _, _, best_page_text, best_lines = best_candidate
            final_lines = best_lines
            coherence_score = self.checker.check(final_lines)
            final_avg_conf = sum(r.confidence for r in final_lines) / max(1, len(final_lines))
            page_similarity = self._page_similarity(best_page_text, image_path.stem)
            line_similarity: Optional[float] = None
            if gt_lines is not None and len(gt_lines) == len(final_lines):
                sims = [self._line_similarity(gt_lines[i], final_lines[i].text) for i in range(len(gt_lines))]
                line_similarity = float(sum(sims) / len(sims)) if sims else 0.0
            final_similarity = line_similarity if line_similarity is not None else page_similarity
            accepted = False

        if self.config.write_report:
            try:
                report = {
                    "image": str(image_path),
                    "accepted": bool(accepted),
                    "attempts": int(attempts),
                    "final": {
                        "avg_confidence": float(final_avg_conf),
                        "coherence": float(coherence_score),
                        "similarity": float(final_similarity) if final_similarity is not None else None,
                        "num_lines": int(len(final_lines)),
                    },
                    "attempts_report": attempt_reports,
                }
                (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")

                summary_lines: List[str] = []
                summary_lines.append(f"image: {image_path}")
                summary_lines.append(f"accepted: {accepted}")
                summary_lines.append(f"attempts: {attempts}")
                summary_lines.append(
                    "final: conf={:.1f} coh={:.1f} sim={}".format(
                        final_avg_conf,
                        coherence_score,
                        final_similarity if final_similarity is not None else "n/a",
                    )
                )
                summary_lines.append("")
                summary_lines.append("attempts:")
                for a in attempt_reports:
                    m = a.get("metrics") or {}
                    t = a.get("tesseract") or {}
                    summary_lines.append(
                        "  - attempt {attempt}: psm={psm} adaptive={adaptive} resize={resize} conf={conf} coh={coh} sim={sim} (page={page} line={line}) lines={lines}".format(
                            attempt=a.get("attempt"),
                            psm=t.get("psm"),
                            adaptive=t.get("adaptive"),
                            resize=t.get("resize_width"),
                            conf=m.get("avg_confidence"),
                            coh=m.get("coherence"),
                            sim=m.get("similarity_used"),
                            page=m.get("similarity_page"),
                            line=m.get("similarity_line"),
                            lines=m.get("num_lines"),
                        )
                    )
                summary_lines.append("")
                summary_lines.append("final_text:")
                summary_lines.append("\n".join(rl.text for rl in final_lines if rl.text).strip())
                (output_dir / "report.txt").write_text("\n".join(summary_lines).strip() + "\n", encoding="utf8")
            except Exception as exc:  # pragma: no cover
                log.info("Failed to write report for %s: %s", image_path.name, exc)

        return XNetResult(
            image=image_path,
            lines=final_lines,
            coherence=coherence_score,
            avg_confidence=final_avg_conf,
            similarity=final_similarity,
            accepted=accepted,
            attempts=attempts,
        )


__all__ = [
    "Segmenter",
    "Recognizer",
    "Checker",
    "KrakenSegmenter",
    "TesseractRecognizer",
    "KrakenRecognizer",
    "SimpleChecker",
    "XNetConfig",
    "XNetController",
    "XNetResult",
]
