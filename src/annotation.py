"""Tkinter application for creating OCR training annotations."""

from __future__ import annotations

import csv
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox

try:  # pragma: no cover - optional dependency for auto-segmentation
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

import numpy as np
from PIL import Image, ImageOps, ImageTk

try:  # pragma: no cover - allow running as a package or script
    from .exporters import save_line_crops, save_pagexml
    from .kraken_adapter import is_available as kraken_available, segment_lines
    from .line_store import Line
except ImportError:  # pragma: no cover
    from exporters import save_line_crops, save_pagexml
    from kraken_adapter import is_available as kraken_available, segment_lines
    from line_store import Line


_prefill_ocr_cached = None
_prefill_ocr_failed = False


def _get_prefill_ocr():
    global _prefill_ocr_cached, _prefill_ocr_failed
    if _prefill_ocr_failed:
        return None
    if _prefill_ocr_cached is not None:
        return _prefill_ocr_cached
    try:
        from .ocr import ocr_image as impl  # type: ignore
    except ImportError:
        try:
            from ocr import ocr_image as impl  # type: ignore
        except ImportError:
            import importlib
            import sys

            root = Path(__file__).resolve().parent.parent
            if str(root) not in sys.path:
                sys.path.append(str(root))
            try:
                impl = importlib.import_module("src.ocr").ocr_image
            except Exception:  # pragma: no cover - optional dependency missing
                _prefill_ocr_failed = True
                return None
    _prefill_ocr_cached = impl
    return impl

CONTROL_MASK = 0x0004
SHIFT_MASK = 0x0001


def _prepare_image(image: Image.Image) -> Image.Image:
    """Return ``image`` rotated according to its EXIF orientation."""

    return ImageOps.exif_transpose(image)


def prepare_image(path: Path) -> Image.Image:
    """Load and prepare an image from ``path``."""

    with Image.open(path) as src:
        prepared = _prepare_image(src)
        return prepared.copy()


@dataclass
class AnnotationItem:
    path: Path
    label: Optional[str] = None
    status: Optional[str] = None
    saved_path: Optional[Path] = None
    prefill: Optional[str] = None


@dataclass
class AnnotationOptions:
    engine: str = "kraken"
    segmentation: str = "auto"
    export_format: str = "lines"
    prefill_enabled: bool = True
    prefill_model: Optional[Path] = None
    prefill_tessdata: Optional[Path] = None
    prefill_psm: int = 6


@dataclass
class OcrToken:
    text: str
    bbox: Tuple[int, int, int, int]
    order_key: Tuple[int, int, int, int, int]
    baseline: Tuple[int, int, int]
    origin: Tuple[int, int, int]


@dataclass
class OverlayItem:
    rect_id: int
    entry: tk.Entry
    bbox: Tuple[int, int, int, int]
    order_key: Tuple[int, int, int, int, int]
    token: Optional[OcrToken] = None
    is_manual: bool = False
    selected: bool = False

    @property
    def text(self) -> str:
        try:
            return self.entry.get()
        except tk.TclError:  # pragma: no cover - defensive for destroyed entries
            return ""


class AnnotationApp:
    """Tkinter-based interface for collecting OCR training data."""

    MAX_SIZE = (900, 700)
    MIN_ZOOM = 0.5
    MAX_ZOOM = 3.0
    ZOOM_STEP = 1.25

    def __init__(
        self,
        master: tk.Tk,
        items: Iterable[AnnotationItem],
        train_dir: Path,
        *,
        options: Optional[AnnotationOptions] = None,
        log_path: Optional[Path] = None,
        on_sample_saved: Optional[callable[[Path], None]] = None,
        transcripts_dir: Optional[Path] = None,
    ) -> None:
        self.master = master
        self.items: List[AnnotationItem] = list(items)
        if not self.items:
            raise ValueError("No images were provided for annotation.")
        self.index = 0
        self.train_dir = Path(train_dir)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.options = options or AnnotationOptions()
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir = Path(transcripts_dir) if transcripts_dir is not None else None
        if self.transcripts_dir is not None:
            self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self._on_sample_saved = on_sample_saved

        self.overlay_entries: List[tk.Entry] = []
        self.overlay_items: List[OverlayItem] = []
        self.rect_to_overlay: Dict[int, OverlayItem] = {}
        self.selected_rects: set[int] = set()
        self.current_tokens: List[OcrToken] = []
        self.manual_token_counter = 0

        self._pressed_overlay: Optional[OverlayItem] = None
        self._drag_start: Optional[Tuple[float, float]] = None
        self._active_temp_rect: Optional[int] = None
        self._modifier_drag = False
        self._marquee_rect: Optional[int] = None
        self._user_modified_transcription = False
        self._setting_transcription = False

        self.display_scale: Tuple[float, float] = (1.0, 1.0)
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.canvas_image_id: Optional[int] = None
        self._base_image: Optional[Image.Image] = None
        self._fit_scale: float = 1.0
        self._zoom_level: float = 1.0
        self._undo_stack: List[Callable[[], None]] = []

        self.mode_var = tk.StringVar(value="select")
        self.status_var = tk.StringVar()
        self.filename_var = tk.StringVar()

        self.canvas: tk.Canvas
        self.entry_widget: tk.Text
        self.delete_button: tk.Button
        self.back_button: tk.Button
        self.lines_frame: tk.Frame

        self._build_ui()
        self._show_current()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.master.title("Standup-OCR Annotation")
        self.master.geometry("1024x840")

        container = tk.Frame(self.master, padx=12, pady=12)
        container.pack(fill="both", expand=True)

        header = tk.Label(container, textvariable=self.filename_var, font=("TkDefaultFont", 14, "bold"))
        header.pack(anchor="w")

        toolbar = tk.Frame(container)
        toolbar.pack(anchor="w", pady=(12, 8))
        tk.Radiobutton(toolbar, text="Select", variable=self.mode_var, value="select").pack(side="left")
        tk.Radiobutton(toolbar, text="Draw", variable=self.mode_var, value="draw").pack(side="left", padx=(8, 0))

        delete_btn = tk.Button(toolbar, text="Delete Selected", command=self._delete_selected, state=tk.DISABLED)
        delete_btn.pack(side="left", padx=(16, 0))
        self.delete_button = delete_btn

        tk.Button(toolbar, text="Zoom In", command=lambda: self._adjust_zoom(self.ZOOM_STEP)).pack(side="left", padx=(16, 0))
        tk.Button(toolbar, text="Zoom Out", command=lambda: self._adjust_zoom(1 / self.ZOOM_STEP)).pack(side="left", padx=(4, 0))
        tk.Button(toolbar, text="Reset Zoom", command=self._reset_zoom).pack(side="left", padx=(4, 0))

        content = tk.Frame(container)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        canvas_frame = tk.Frame(content)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bd=1, relief="sunken", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        side_panel = tk.Frame(content)
        side_panel.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(0, weight=1)
        side_panel.rowconfigure(1, weight=1)

        lines_container = tk.LabelFrame(side_panel, text="Lines")
        lines_container.grid(row=0, column=0, sticky="nsew")
        self.lines_frame = tk.Frame(lines_container)
        self.lines_frame.pack(fill="both", expand=True)

        transcription_container = tk.LabelFrame(side_panel, text="Transcription")
        transcription_container.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        transcription_container.rowconfigure(0, weight=1)
        transcription_container.columnconfigure(0, weight=1)
        text_widget = tk.Text(transcription_container, height=6, wrap="word")
        text_widget.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        text_widget.bind("<Key>", self._on_transcription_modified)
        text_widget.bind("<Control-Return>", self._on_confirm)
        text_widget.bind("<Command-Return>", self._on_confirm)
        self.entry_widget = text_widget

        buttons = tk.Frame(side_panel)
        buttons.grid(row=2, column=0, sticky="ew", pady=(12, 8))
        back_btn = tk.Button(buttons, text="Back", command=self.back)
        back_btn.pack(side="left", padx=4)
        self.back_button = back_btn
        tk.Button(buttons, text="Skip", command=self.skip).pack(side="left", padx=4)
        confirm_btn = tk.Button(buttons, text="Confirm", command=self.confirm, default=tk.ACTIVE)
        confirm_btn.pack(side="left", padx=4)
        tk.Button(buttons, text="Unsure", command=self.unsure).pack(side="left", padx=4)

        status_label = tk.Label(side_panel, textvariable=self.status_var, fg="gray")
        status_label.grid(row=3, column=0, sticky="w")

        self.canvas.bind("<ButtonPress-1>", self._on_canvas_button_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Control-MouseWheel>", self._on_mouse_zoom)
        self.canvas.bind("<Control-Button-4>", lambda event: self._adjust_zoom(self.ZOOM_STEP))
        self.canvas.bind("<Control-Button-5>", lambda event: self._adjust_zoom(1 / self.ZOOM_STEP))

        self.master.bind("<Escape>", self._on_escape)
        self.master.bind("<Delete>", self._on_delete_selected)
        self.master.bind("<BackSpace>", self._on_delete_selected)
        self.master.bind("<Control-z>", self._on_undo)
        self.master.bind("<Command-z>", self._on_undo)
        self.master.bind("<Control-Z>", self._on_undo)
        self.master.bind("<Control-plus>", self._on_zoom_in)
        self.master.bind("<Control-equal>", self._on_zoom_in)
        self.master.bind("<Control-KP_Add>", self._on_zoom_in)
        self.master.bind("<Control-minus>", self._on_zoom_out)
        self.master.bind("<Control-underscore>", self._on_zoom_out)
        self.master.bind("<Control-KP_Subtract>", self._on_zoom_out)
        self.master.bind("<Control-0>", self._on_zoom_reset)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _slugify(self, value: str) -> str:
        slug = "-".join(value.strip().lower().split())
        slug = slug.replace("_", "-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug[:60].rstrip("-")

    def _on_confirm(self, _event: Optional[tk.Event]) -> None:
        self.confirm()

    def _on_escape(self, _event: Optional[tk.Event]) -> None:
        if self._active_temp_rect is not None:
            self.canvas.delete(self._active_temp_rect)
            self._active_temp_rect = None
            self._drag_start = None
        else:
            self.master.destroy()

    def _get_transcription_text(self) -> str:
        return self.entry_widget.get("1.0", tk.END).rstrip("\n")

    def _set_transcription(self, value: str) -> None:
        self._setting_transcription = True
        self.entry_widget.delete("1.0", tk.END)
        if value:
            self.entry_widget.insert("1.0", value)
        self._setting_transcription = False

    def _apply_transcription_to_overlays(self) -> None:
        for entry in list(self.overlay_entries):
            try:
                entry.delete(0, tk.END)
            except Exception:  # pragma: no cover - defensive for stub entries
                pass

        if not self.overlay_items:
            self.current_tokens = []
            return

        raw_text = self.entry_widget.get("1.0", tk.END)
        normalized = raw_text.replace("\r\n", "\n")

        overlays = sorted(self.overlay_items, key=lambda item: item.order_key)
        segments = normalized.split("\n")
        segment_index = 0
        updated_tokens: List[OcrToken] = []

        for position, overlay in enumerate(overlays):
            remaining_overlays = len(overlays) - position
            while (
                segment_index < len(segments)
                and segments[segment_index] == ""
                and (len(segments) - segment_index) > remaining_overlays
            ):
                segment_index += 1

            if segment_index < len(segments):
                line_text = segments[segment_index]
                segment_index += 1
            else:
                line_text = ""

            overlay.entry.delete(0, tk.END)
            overlay.entry.insert(0, line_text)

            if overlay.token is not None:
                overlay.token.text = line_text
                baseline = overlay.token.baseline
                origin = overlay.token.origin
            else:
                baseline = (0, 0, 0)
                origin = (0, 0, 0)

            updated_tokens.append(
                OcrToken(
                    text=line_text,
                    bbox=overlay.bbox,
                    order_key=overlay.order_key,
                    baseline=baseline,
                    origin=origin,
                )
            )

        self.current_tokens = updated_tokens

    def _on_transcription_modified(self, _event: Optional[tk.Event]) -> None:
        if self._setting_transcription:
            return
        self._user_modified_transcription = True
        self._apply_transcription_to_overlays()

    def _compose_text_from_tokens(self, tokens: Sequence[OcrToken]) -> str:
        pieces: List[str] = []
        previous_paragraph: Optional[Tuple[int, int, int]] = None
        previous_line: Optional[Tuple[int, int, int, int]] = None
        for token in tokens:
            paragraph_key = token.order_key[:3]
            line_key = token.order_key[:4]
            if not pieces:
                pieces.append(token.text)
            else:
                if paragraph_key != previous_paragraph:
                    pieces.append("\n\n" + token.text)
                elif line_key != previous_line:
                    pieces.append("\n" + token.text)
                else:
                    pieces.append(" " + token.text)
            previous_paragraph = paragraph_key
            previous_line = line_key
        return "".join(pieces)

    def _scale_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        sx, sy = self.display_scale
        x1, y1, x2, y2 = bbox
        return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

    def _to_base_bbox(self, coords: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        sx, sy = self.display_scale
        x1, y1, x2, y2 = coords
        if sx == 0 or sy == 0:
            return int(x1), int(y1), int(x2), int(y2)
        return int(x1 / sx), int(y1 / sy), int(x2 / sx), int(y2 / sy)

    def _safe_messagebox(self, name: str, *args, **kwargs):
        func = getattr(messagebox, name)
        try:
            return func(*args, **kwargs)
        except tk.TclError:
            return None

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def confirm(self) -> None:
        text = self._get_transcription_text()
        if not self.overlay_items and not text:
            self._safe_messagebox("showinfo", "No lines", "Create at least one line before confirming.")
            return

        if not text:
            if not self._safe_messagebox(
                "askyesno",
                "Empty transcription",
                "No text was entered for the selected lines. Export anyway?",
            ):
                return

        item = self.items[self.index]
        try:
            saved_path = self._save_annotation(item, text)
        except OSError as exc:
            self._safe_messagebox("showerror", "Export failed", f"Could not export training data: {exc}")
            return
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.exception("Export failed")
            self._safe_messagebox("showerror", "Export failed", str(exc))
            return

        item.label = text
        item.status = "confirmed"
        item.saved_path = saved_path
        self._append_log(item.path, text, "confirmed", saved_path)
        self._update_transcript(item, text)
        self.status_var.set("Exported training data")
        callback = getattr(self, "_on_sample_saved", None)
        if callback is not None and saved_path is not None:
            callback(saved_path)
        self._advance()

    def skip(self) -> None:
        item = self.items[self.index]
        item.status = "skipped"
        item.label = ""
        item.saved_path = None
        self._append_log(item.path, "", "skipped", None)
        self.status_var.set("Skipped")
        self._advance()

    def unsure(self) -> None:
        item = self.items[self.index]
        item.status = "unsure"
        item.label = self._get_transcription_text()
        item.saved_path = None
        self._append_log(item.path, item.label or "", "unsure", None)
        self.status_var.set("Marked as unsure")
        self._advance()

    def back(self) -> None:
        if self.index == 0:
            if hasattr(self, "back_button"):
                self.back_button.config(state=tk.DISABLED)
            self.status_var.set("Already at the first item.")
            return
        self.index -= 1
        self._show_current(revisit=True)

    def _advance(self) -> None:
        self.index += 1
        if self.index >= len(self.items):
            self._safe_messagebox("showinfo", "Complete", "All images have been processed.")
            self.master.destroy()
            return
        self._show_current()

    def _show_current(self, *, revisit: bool = False) -> None:
        item = self.items[self.index]
        self.filename_var.set(f"{item.path.name} ({self.index + 1}/{len(self.items)})")
        self.back_button.config(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self._display_item(item)
        self.entry_widget.focus_set()
        if revisit:
            status = self.status_var.get()
            extra = []
            if item.saved_path is not None:
                extra.append(f"Previously saved to {item.saved_path.name}.")
            extra.append("Returned to previous item.")
            prefix = f"{status} " if status else ""
            self.status_var.set(prefix + " ".join(extra))

    # ------------------------------------------------------------------
    # Item display and token management
    # ------------------------------------------------------------------
    def _display_item(self, item: AnnotationItem) -> None:
        try:
            with Image.open(item.path) as image:
                base = _prepare_image(image).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open {item.path.name}: {exc}")
            self.skip()
            return

        if base.width == 0 or base.height == 0:
            self.status_var.set("Image has invalid dimensions.")
            return

        fit_scale = min(
            self.MAX_SIZE[0] / base.width if base.width else 1.0,
            self.MAX_SIZE[1] / base.height if base.height else 1.0,
            1.0,
        )
        display_size = (
            max(1, int(base.width * fit_scale)),
            max(1, int(base.height * fit_scale)),
        )
        display = base.resize(display_size, Image.LANCZOS)
        sx = display.width / base.width if base.width else 1.0
        sy = display.height / base.height if base.height else 1.0
        self.display_scale = (sx, sy)
        self._base_image = base
        self._fit_scale = fit_scale
        self._zoom_level = 1.0
        stack = getattr(self, "_undo_stack", None)
        if stack is not None:
            stack.clear()
        else:
            self._undo_stack = []

        tokens: List[OcrToken]
        if item.label:
            tokens = []
        elif self.options.segmentation == "auto" and self.options.engine == "kraken" and kraken_available():
            try:
                baselines = segment_lines(item.path)
            except RuntimeError as exc:
                self.status_var.set(str(exc))
                baselines = []
            tokens = [
                OcrToken(
                    text="",
                    bbox=self._baseline_to_bbox(baseline),
                    order_key=(1, 1, 1, index + 1, 1),
                    baseline=(0, 0, 0),
                    origin=(0, 0, 0),
                )
                for index, baseline in enumerate(baselines)
            ]
        else:
            tokens = self._extract_tokens(base)

        self._display_image(display, tokens)

        if item.label:
            self._set_transcription(item.label)
            self._apply_transcription_to_overlays()
            self.status_var.set("Loaded existing transcription.")
            return

        suggestion = self._get_prefill(item)
        if suggestion:
            self._set_transcription(suggestion)
            self._apply_transcription_to_overlays()
            self.status_var.set("Pre-filled transcription using OCR result.")
            return

        fallback = self._compose_text_from_tokens(tokens)
        self._set_transcription(fallback)
        self._apply_transcription_to_overlays()
        if tokens and self.options.engine == "kraken" and kraken_available():
            self.status_var.set("Kraken returned baseline segmentation.")
        elif tokens:
            self.status_var.set(f"Detected {len(tokens)} line(s) automatically.")
        else:
            self.status_var.set("Draw baselines manually using the canvas.")

    def _get_prefill(self, item: AnnotationItem) -> str:
        options = getattr(self, "options", AnnotationOptions())
        if not options.prefill_enabled:
            return ""
        if item.prefill is not None:
            return item.prefill
        suggestion = self._suggest_label(item.path)
        item.prefill = suggestion
        return suggestion

    def _display_image(self, image: Image.Image, tokens: Sequence[OcrToken]) -> None:
        self.canvas.delete("all")
        self._clear_overlay_entries()
        self.overlay_items.clear()
        self.rect_to_overlay.clear()
        self.selected_rects.clear()
        self.current_tokens = list(tokens)

        self.current_photo = ImageTk.PhotoImage(image)
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.current_photo)
        if hasattr(self.canvas, "config"):
            self.canvas.config(scrollregion=(0, 0, image.width, image.height))

        for token in tokens:
            self._create_overlay(token.text, token.bbox, token.order_key, token)

        self._update_transcription_from_overlays()
        manual_keys = [item.order_key[3] for item in self.overlay_items if item.is_manual]
        if manual_keys:
            self.manual_token_counter = max(manual_keys)
        else:
            self.manual_token_counter = 0

    def _baseline_to_bbox(self, baseline: Sequence[Tuple[float, float]]) -> Tuple[int, int, int, int]:
        xs = [point[0] for point in baseline]
        ys = [point[1] for point in baseline]
        left = int(min(xs))
        right = int(max(xs))
        top = int(min(ys))
        bottom = int(max(ys)) + 10
        return left, top, right, bottom

    def _create_overlay(
        self,
        text: str,
        bbox: Tuple[int, int, int, int],
        order_key: Tuple[int, int, int, int, int],
        token: Optional[OcrToken] = None,
        *,
        is_manual: bool = False,
        select: bool = False,
    ) -> OverlayItem:
        scaled = self._scale_bbox(bbox)
        rect_id = self.canvas.create_rectangle(*scaled, outline="#f97316", width=2, tags=("overlay",))
        entry_parent = getattr(self, "lines_frame", None)
        entry = tk.Entry(entry_parent, width=80)
        entry.insert(0, text)
        entry.bind("<Key>", self._on_overlay_modified)
        if hasattr(entry, "pack"):
            entry.pack(fill="x", pady=2)
        overlay = OverlayItem(rect_id=rect_id, entry=entry, bbox=bbox, order_key=order_key, token=token, is_manual=is_manual)
        self.overlay_items.append(overlay)
        self.overlay_entries.append(entry)
        self.rect_to_overlay[rect_id] = overlay
        if select:
            self._select_overlay(overlay, additive=False)
        return overlay

    def _clear_overlay_entries(self) -> None:
        for entry in list(self.overlay_entries):
            try:
                entry.destroy()
            except Exception:  # pragma: no cover - defensive
                pass
        self.overlay_entries.clear()

    def _suggest_label(self, path: Path) -> str:
        options = getattr(self, "options", AnnotationOptions())
        if not options.prefill_enabled:
            return ""
        ocr_impl = _get_prefill_ocr()
        if ocr_impl is None:
            return ""
        try:
            text = ocr_impl(
                path,
                model_path=options.prefill_model,
                tessdata_dir=options.prefill_tessdata,
                psm=options.prefill_psm,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.debug("Prefill OCR failed for %s: %s", path, exc)
            return ""
        return text.strip()

    def _extract_tokens(self, image: Image.Image) -> List[OcrToken]:
        if image.width == 0 or image.height == 0:
            return []

        gray = np.array(image.convert("L"))
        if gray.size == 0:
            return []

        if cv2 is not None:
            return self._extract_tokens_with_cv2(gray)
        return self._extract_tokens_without_cv2(gray)

    def _extract_tokens_with_cv2(self, gray: np.ndarray) -> List[OcrToken]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = 255 - thresh

        non_zero = cv2.findNonZero(binary)
        if non_zero is not None:
            x, y, w, h = cv2.boundingRect(non_zero)
            roi = binary[y : y + h, x : x + w]
        else:
            x = y = 0
            h, w = binary.shape
            roi = binary

        kernel_width = max(15, roi.shape[1] // 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
        connected = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            if ch < 12 or cw < 20:
                continue
            boxes.append((x + cx, y + cy, x + cx + cw, y + cy + ch))

        if not boxes:
            return []

        boxes.sort(key=lambda box: (box[1], box[0]))
        heights = [bottom - top for _, top, _, bottom in boxes]
        median_height = float(np.median(heights)) if heights else 0.0
        line_threshold = median_height * 0.6 if median_height else 20.0

        lines: List[List[Tuple[int, int, int, int]]] = []
        for box in boxes:
            placed = False
            for line in lines:
                reference_top = float(np.mean([b[1] for b in line])) if line else box[1]
                if abs(box[1] - reference_top) <= line_threshold:
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])

        tokens: List[OcrToken] = []
        for line_index, line_boxes in enumerate(lines, start=1):
            for column_index, (left, top, right, bottom) in enumerate(sorted(line_boxes, key=lambda item: item[0]), start=1):
                baseline = (left, bottom - 1, right)
                origin = (left, top, bottom)
                order_key = (2, line_index, 1, column_index, 1)
                tokens.append(
                    OcrToken(
                        text="",
                        bbox=(left, top, right, bottom),
                        order_key=order_key,
                        baseline=baseline,
                        origin=origin,
                    )
                )

        tokens.sort(key=lambda token: token.order_key)
        return tokens

    def _extract_tokens_without_cv2(self, gray: np.ndarray) -> List[OcrToken]:
        threshold = self._otsu_threshold(gray)
        mask = gray <= threshold
        if not np.any(mask):
            return []

        row_activity = mask.sum(axis=1)
        min_line_pixels = max(8, int(mask.shape[1] * 0.01))
        active_rows = row_activity >= min_line_pixels
        line_runs = self._find_runs(active_rows)

        min_line_height = max(6, int(mask.shape[0] * 0.005))
        tokens: List[OcrToken] = []
        for line_index, (start, end) in enumerate(line_runs, start=1):
            if end - start < min_line_height:
                continue

            line_slice = mask[start:end, :]
            column_activity = line_slice.sum(axis=0)
            min_col_pixels = max(4, int((end - start) * 0.3))
            active_cols = np.where(column_activity >= min_col_pixels)[0]
            if active_cols.size == 0:
                active_cols = np.where(column_activity > 0)[0]
                if active_cols.size == 0:
                    continue

            left = int(max(0, active_cols[0] - 2))
            right = int(min(mask.shape[1], active_cols[-1] + 3))
            top = max(0, start - 2)
            bottom = min(mask.shape[0], end + 2)

            baseline = (left, bottom - 1, right)
            origin = (left, top, bottom)
            order_key = (4, line_index, 1, 1, 1)
            tokens.append(
                OcrToken(
                    text="",
                    bbox=(left, top, right, bottom),
                    order_key=order_key,
                    baseline=baseline,
                    origin=origin,
                )
            )

        tokens.sort(key=lambda token: token.order_key)
        return tokens

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> int:
        histogram, _ = np.histogram(gray, bins=256, range=(0, 256))
        total = gray.size
        sum_total = float(np.dot(np.arange(256), histogram))

        sum_background = 0.0
        weight_background = 0
        max_variance = 0.0
        threshold = 0
        for value, count in enumerate(histogram):
            weight_background += count
            if weight_background == 0:
                continue

            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break

            sum_background += value * count
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground

            variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            if variance > max_variance:
                max_variance = variance
                threshold = value

        return threshold

    @staticmethod
    def _find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for index, value in enumerate(mask):
            if bool(value) and start is None:
                start = index
            elif not bool(value) and start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, mask.size))
        return runs

    # ------------------------------------------------------------------
    # Canvas interaction
    # ------------------------------------------------------------------
    def _on_canvas_button_press(self, event: tk.Event) -> None:
        self._pressed_overlay = None
        self._modifier_drag = bool(event.state & (CONTROL_MASK | SHIFT_MASK))
        canvas_x = self.canvas.canvasx(event.x) if hasattr(self.canvas, "canvasx") else event.x
        canvas_y = self.canvas.canvasy(event.y) if hasattr(self.canvas, "canvasy") else event.y
        self._drag_start = (canvas_x, canvas_y)

        for overlay in self.overlay_items:
            left, top, right, bottom = self.canvas.coords(overlay.rect_id)
            if left <= canvas_x <= right and top <= canvas_y <= bottom:
                self._pressed_overlay = overlay
                if not self._modifier_drag:
                    self._clear_selection()
                self._select_overlay(overlay, additive=self._modifier_drag)
                return

        if self.mode_var.get() == "draw":
            self._start_manual_overlay(canvas_x, canvas_y)
        else:
            self._clear_selection()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._active_temp_rect is not None:
            canvas_x = self.canvas.canvasx(event.x) if hasattr(self.canvas, "canvasx") else event.x
            canvas_y = self.canvas.canvasy(event.y) if hasattr(self.canvas, "canvasy") else event.y
            x0, y0 = self._drag_start if self._drag_start is not None else (canvas_x, canvas_y)
            self.canvas.coords(self._active_temp_rect, x0, y0, canvas_x, canvas_y)

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self._active_temp_rect is not None:
            canvas_x = self.canvas.canvasx(event.x) if hasattr(self.canvas, "canvasx") else event.x
            canvas_y = self.canvas.canvasy(event.y) if hasattr(self.canvas, "canvasy") else event.y
            self._finish_manual_overlay(canvas_x, canvas_y)
            return

        if self._pressed_overlay is not None and self._modifier_drag:
            self._select_overlay(self._pressed_overlay, additive=True)
        elif self._pressed_overlay is not None:
            self._select_overlay(self._pressed_overlay, additive=False)

    def _start_manual_overlay(self, x: float, y: float) -> None:
        self._active_temp_rect = self.canvas.create_rectangle(x, y, x, y, outline="#10b981", dash=(4, 2))

    def _finish_manual_overlay(self, x: float, y: float) -> None:
        if self._active_temp_rect is None or self._drag_start is None:
            return
        coords = self.canvas.coords(self._active_temp_rect)
        self.canvas.delete(self._active_temp_rect)
        self._active_temp_rect = None
        self._drag_start = None

        left, top, right, bottom = coords
        if abs(right - left) < 5 or abs(bottom - top) < 5:
            return

        base_bbox = self._to_base_bbox((left, top, right, bottom))
        self.manual_token_counter += 1
        order_key = (9, 9, 9, self.manual_token_counter, 1)
        overlay = self._create_overlay("", base_bbox, order_key, None, is_manual=True, select=True)
        overlay.entry.focus_set()
        if hasattr(self, "status_var"):
            self.status_var.set("Added manual line")
        self._update_transcription_from_overlays()
        self._push_undo(lambda key=order_key: self._undo_manual_overlay(key))

    def _select_overlay(self, overlay: OverlayItem, *, additive: bool) -> None:
        if not additive:
            self._clear_selection()
        overlay.selected = True
        self.selected_rects.add(overlay.rect_id)
        self.canvas.itemconfigure(overlay.rect_id, outline="#2563eb")
        if hasattr(self, "delete_button"):
            self.delete_button.config(state=tk.NORMAL)

    def _clear_selection(self) -> None:
        for overlay in self.overlay_items:
            if overlay.selected:
                overlay.selected = False
                self.canvas.itemconfigure(overlay.rect_id, outline="#f97316")
        self.selected_rects.clear()
        if hasattr(self, "delete_button"):
            self.delete_button.config(state=tk.DISABLED)

    def _on_overlay_modified(self, _event: Optional[tk.Event]) -> None:
        self._user_modified_transcription = True
        self._update_transcription_from_overlays()

    def _update_transcription_from_overlays(self) -> None:
        if self._setting_transcription:
            return
        tokens: List[OcrToken] = []
        for overlay in sorted(self.overlay_items, key=lambda item: item.order_key):
            tokens.append(
                OcrToken(
                    text=overlay.text,
                    bbox=overlay.bbox,
                    order_key=overlay.order_key,
                    baseline=(0, 0, 0),
                    origin=(0, 0, 0),
                )
            )
        self.current_tokens = tokens
        text = self._compose_text_from_tokens(tokens)
        self._set_transcription(text)

    def _delete_selected(self) -> None:
        if not self.selected_rects:
            return
        snapshots: List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int, int], Optional[OcrToken], bool]] = []
        overlays_to_delete: List[OverlayItem] = []
        for rect_id in list(self.selected_rects):
            overlay = self.rect_to_overlay.pop(rect_id, None)
            if overlay is None:
                continue
            snapshots.append(
                (
                    overlay.text,
                    overlay.bbox,
                    overlay.order_key,
                    self._clone_token(overlay.token),
                    overlay.is_manual,
                )
            )
            overlays_to_delete.append(overlay)
        for overlay in overlays_to_delete:
            try:
                overlay.entry.destroy()
            except Exception:  # pragma: no cover
                pass
            try:
                self.canvas.delete(overlay.rect_id)
            except Exception:  # pragma: no cover - stub canvas may not implement delete
                pass
            if overlay in self.overlay_items:
                self.overlay_items.remove(overlay)
            if overlay.entry in self.overlay_entries:
                self.overlay_entries.remove(overlay.entry)
            self.selected_rects.discard(overlay.rect_id)
        self.selected_rects.clear()
        if snapshots:
            self._push_undo(lambda data=snapshots: self._restore_overlays(data))
        if hasattr(self, "delete_button"):
            self.delete_button.config(state=tk.DISABLED)
        self._update_transcription_from_overlays()

    def _on_delete_selected(self, _event: Optional[tk.Event]) -> None:
        try:
            widget = self.master.focus_get()
        except tk.TclError:
            widget = None
        if widget is not None:
            widget_class = widget.winfo_class()
            if widget_class in {"Entry", "Text", "TEntry"}:
                return
        self._delete_selected()

    # ------------------------------------------------------------------
    # Undo and zoom helpers
    # ------------------------------------------------------------------
    def _push_undo(self, action: Callable[[], None]) -> None:
        stack = getattr(self, "_undo_stack", None)
        if stack is None:
            stack = []
            self._undo_stack = stack
        stack.append(action)
        if len(stack) > 50:
            stack.pop(0)

    def _undo_manual_overlay(self, order_key: Tuple[int, int, int, int, int]) -> None:
        overlay = self._find_overlay_by_order_key(order_key)
        if overlay is None:
            return
        self._destroy_overlay(overlay)
        self._update_transcription_from_overlays()
        if hasattr(self, "status_var"):
            self.status_var.set("Removed manual line")

    def _restore_overlays(
        self,
        data: List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int, int], Optional[OcrToken], bool]],
    ) -> None:
        overlays: List[OverlayItem] = []
        for text, bbox, order_key, token, is_manual in data:
            overlay = self._create_overlay(text, bbox, order_key, token, is_manual=is_manual, select=False)
            overlays.append(overlay)
        self._clear_selection()
        for overlay in overlays:
            self._select_overlay(overlay, additive=True)
        self._update_transcription_from_overlays()
        if hasattr(self, "status_var"):
            self.status_var.set("Restored deleted lines")

    def _find_overlay_by_order_key(
        self, order_key: Tuple[int, int, int, int, int]
    ) -> Optional[OverlayItem]:
        for overlay in self.overlay_items:
            if overlay.order_key == order_key:
                return overlay
        return None

    def _destroy_overlay(self, overlay: OverlayItem) -> None:
        try:
            self.canvas.delete(overlay.rect_id)
        except Exception:  # pragma: no cover
            pass
        try:
            overlay.entry.destroy()
        except Exception:  # pragma: no cover
            pass
        if overlay in self.overlay_items:
            self.overlay_items.remove(overlay)
        if overlay.entry in self.overlay_entries:
            self.overlay_entries.remove(overlay.entry)
        self.rect_to_overlay.pop(overlay.rect_id, None)
        self.selected_rects.discard(overlay.rect_id)

    def _clone_token(self, token: Optional[OcrToken]) -> Optional[OcrToken]:
        if token is None:
            return None
        return OcrToken(
            text=token.text,
            bbox=token.bbox,
            order_key=token.order_key,
            baseline=token.baseline,
            origin=token.origin,
        )

    def _on_undo(self, _event: Optional[tk.Event]) -> None:
        stack = getattr(self, "_undo_stack", None)
        if not stack:
            return
        action = stack.pop()
        action()

    def _on_zoom_in(self, _event: Optional[tk.Event]) -> None:
        self._adjust_zoom(self.ZOOM_STEP)

    def _on_zoom_out(self, _event: Optional[tk.Event]) -> None:
        self._adjust_zoom(1 / self.ZOOM_STEP)

    def _on_zoom_reset(self, _event: Optional[tk.Event]) -> None:
        self._reset_zoom()

    def _on_mouse_zoom(self, event: tk.Event) -> str:
        delta = getattr(event, "delta", 0)
        if delta > 0:
            self._adjust_zoom(self.ZOOM_STEP)
        elif delta < 0:
            self._adjust_zoom(1 / self.ZOOM_STEP)
        return "break"

    def _adjust_zoom(self, factor: float) -> None:
        if self._base_image is None:
            return
        current_zoom = getattr(self, "_zoom_level", 1.0)
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, current_zoom * factor))
        if abs(new_zoom - current_zoom) < 1e-3:
            return
        self._zoom_level = new_zoom
        self._render_zoomed_image()

    def _reset_zoom(self) -> None:
        if self._base_image is None:
            return
        self._zoom_level = 1.0
        self._render_zoomed_image()

    def _render_zoomed_image(self) -> None:
        if self._base_image is None:
            return
        fit_scale = getattr(self, "_fit_scale", 1.0)
        zoom_level = getattr(self, "_zoom_level", 1.0)
        self._zoom_level = zoom_level
        scale = fit_scale * zoom_level
        width = max(1, int(self._base_image.width * scale))
        height = max(1, int(self._base_image.height * scale))
        display = self._base_image.resize((width, height), Image.LANCZOS)
        view_x = self.canvas.xview() if hasattr(self.canvas, "xview") else (0.0, 1.0)
        view_y = self.canvas.yview() if hasattr(self.canvas, "yview") else (0.0, 1.0)
        tokens = [
            OcrToken(
                text=token.text,
                bbox=token.bbox,
                order_key=token.order_key,
                baseline=token.baseline,
                origin=token.origin,
            )
            for token in self.current_tokens
        ]
        self.display_scale = (
            display.width / self._base_image.width if self._base_image.width else 1.0,
            display.height / self._base_image.height if self._base_image.height else 1.0,
        )
        self._display_image(display, tokens)
        if hasattr(self.canvas, "config"):
            self.canvas.config(scrollregion=(0, 0, display.width, display.height))
        if hasattr(self.canvas, "xview_moveto"):
            self.canvas.xview_moveto(view_x[0])
        if hasattr(self.canvas, "yview_moveto"):
            self.canvas.yview_moveto(view_y[0])

    # ------------------------------------------------------------------
    # Transcription export helpers
    # ------------------------------------------------------------------
    def _collect_lines(self) -> List[Line]:
        lines: List[Line] = []
        for idx, overlay in enumerate(sorted(self.overlay_items, key=lambda item: item.order_key), start=1):
            left, top, right, bottom = overlay.bbox
            baseline = [(left, bottom - 1), (right, bottom - 1)]
            line = Line(
                id=idx,
                baseline=baseline,
                bbox=overlay.bbox,
                text=overlay.text,
                order_key=overlay.order_key,
                selected=overlay.selected,
                is_manual=overlay.is_manual,
            )
            lines.append(line)
        return lines

    def _save_annotation(self, item: AnnotationItem, label: str) -> Path:
        lines = self._collect_lines()
        if self.options.export_format == "pagexml":
            out_dir = self.train_dir / "pagexml"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{self._slugify(item.path.stem)}.xml"
            save_pagexml(item.path, lines, out_path)
            return out_path

        out_dir = self.train_dir / "lines"
        save_line_crops(item.path, lines, out_dir)
        return out_dir

    def _append_log(
        self, path: Path, label: str, status: str, saved_path: Optional[Path]
    ) -> None:
        if self.log_path is None:
            return

        now = datetime.now(timezone.utc).isoformat()
        row = {
            "page": path.name,
            "transcription": label,
            "timestamp": now,
            "status": status,
            "saved_path": saved_path.name if saved_path else "",
        }

        fieldnames = ["page", "transcription", "timestamp", "status", "saved_path"]
        write_header = False

        if self.log_path.exists():
            with self.log_path.open("r", newline="", encoding="utf8") as handle:
                reader = csv.DictReader(handle)
                existing_fields = reader.fieldnames or []
                missing_required = [
                    name
                    for name in ("page", "transcription", "timestamp")
                    if name not in existing_fields
                ]
                if missing_required:
                    migrated_rows = []
                    for existing in reader:
                        migrated_rows.append(
                            {
                                "page": existing.get("page")
                                or existing.get("filename")
                                or "",
                                "transcription": existing.get("transcription")
                                or existing.get("label")
                                or "",
                                "timestamp": existing.get("timestamp")
                                or now,
                                "status": existing.get("status") or "",
                                "saved_path": existing.get("saved_path") or "",
                            }
                        )

                    with self.log_path.open("w", newline="", encoding="utf8") as out:
                        writer = csv.DictWriter(out, fieldnames=fieldnames)
                        writer.writeheader()
                        for migrated in migrated_rows:
                            writer.writerow(migrated)
                else:
                    fieldnames = list(existing_fields)
                    for extra in ("status", "saved_path"):
                        if extra and extra not in fieldnames:
                            fieldnames.append(extra)
        else:
            write_header = True

        with self.log_path.open("a", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _update_transcript(self, item: AnnotationItem, text: str) -> None:
        transcripts_dir = getattr(self, "transcripts_dir", None)
        if transcripts_dir is None:
            return

        transcript_path = Path(transcripts_dir) / f"{item.path.stem}.txt"
        transcript_path.write_text(text, encoding="utf8")


@dataclass
class AnnotationAutoTrainConfig:
    auto_train: int
    output_model: str
    model_dir: Path
    base_lang: str
    max_iterations: int
    tessdata_dir: Optional[Path] = None


@dataclass
class AnnotationTrainer:
    master: tk.Misc
    train_dir: Path
    config: AnnotationAutoTrainConfig
    _pending: Sequence[Path] = field(default_factory=list, init=False)
    seen_samples: List[Path] = field(default_factory=list, init=False)

    def __call__(self, sample_path: Path) -> None:
        path = Path(sample_path)
        self._pending = list(self._pending) + [path]
        self.seen_samples.append(path)
        if len(self._pending) >= self.config.auto_train:
            self.master.after(0, self._maybe_train)

    def _maybe_train(self) -> None:
        if len(self._pending) < self.config.auto_train:
            return
        batch = list(self._pending)[: self.config.auto_train]
        self._pending = self._pending[self.config.auto_train :]

        def worker() -> None:
            try:
                logging.info("Auto-training triggered by %d new annotations", len(batch))
                model_path = _train_model(
                    self.train_dir,
                    self.config.output_model,
                    model_dir=self.config.model_dir,
                    tessdata_dir=self.config.tessdata_dir,
                    base_lang=self.config.base_lang,
                    max_iterations=self.config.max_iterations,
                )
                logging.info("Updated model saved to %s", model_path)
            except Exception:
                logging.exception("Auto-training failed")

        threading.Thread(target=worker, daemon=True).start()


def _train_model(*args, **kwargs):  # pragma: no cover
    if __package__:
        from .training import train_model as _impl
    else:
        from training import train_model as _impl  # type: ignore
    return _impl(*args, **kwargs)


def annotate_images(
    sources: Iterable[Path],
    train_dir: Path,
    *,
    options: Optional[AnnotationOptions] = None,
    log_path: Optional[Path] = None,
    auto_train_config: Optional[AnnotationAutoTrainConfig] = None,
    transcripts_dir: Optional[Path] = None,
) -> None:
    items = [AnnotationItem(Path(path)) for path in sources]
    if not items:
        raise ValueError("No images found to annotate.")

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(
            "Tkinter could not be initialised. Ensure a display is available or install python3-tk."
        ) from exc

    callback = None
    if auto_train_config is not None:
        callback = AnnotationTrainer(root, train_dir=Path(train_dir), config=auto_train_config)

    app = AnnotationApp(
        root,
        items,
        Path(train_dir),
        options=options,
        log_path=log_path,
        on_sample_saved=callback,
        transcripts_dir=transcripts_dir,
    )

    logging.info(
        "Annotation window ready for %d image(s). Interact with the GUI to continue (e.g. Save/Skip).",
        len(items),
    )
    root.mainloop()
