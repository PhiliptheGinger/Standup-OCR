"""Tkinter application for creating OCR training annotations."""

from __future__ import annotations

import csv
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk

try:  # pragma: no cover - allow running as a package or script
    from .exporters import save_line_crops, save_pagexml
    from .kraken_adapter import is_available as kraken_available, segment_lines
    from .line_store import Line
except ImportError:  # pragma: no cover
    from exporters import save_line_crops, save_pagexml
    from kraken_adapter import is_available as kraken_available, segment_lines
    from line_store import Line

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


@dataclass
class AnnotationOptions:
    engine: str = "kraken"
    segmentation: str = "auto"
    export_format: str = "lines"


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

    def __init__(
        self,
        master: tk.Tk,
        items: Iterable[AnnotationItem],
        train_dir: Path,
        *,
        options: Optional[AnnotationOptions] = None,
        log_path: Optional[Path] = None,
        on_sample_saved: Optional[callable[[Path], None]] = None,
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

        canvas_frame = tk.Frame(container)
        canvas_frame.pack(fill="both", expand=True, pady=12)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bd=1, relief="sunken", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        toolbar = tk.Frame(container)
        toolbar.pack(anchor="w", pady=(0, 8))
        tk.Radiobutton(toolbar, text="Select", variable=self.mode_var, value="select").pack(side="left")
        tk.Radiobutton(toolbar, text="Draw", variable=self.mode_var, value="draw").pack(side="left", padx=(8, 0))

        delete_btn = tk.Button(toolbar, text="Delete Selected", command=self._delete_selected, state=tk.DISABLED)
        delete_btn.pack(side="left", padx=(16, 0))
        self.delete_button = delete_btn

        lines_container = tk.LabelFrame(container, text="Lines")
        lines_container.pack(fill="x", pady=(0, 8))
        self.lines_frame = tk.Frame(lines_container)
        self.lines_frame.pack(fill="x")

        entry_frame = tk.Frame(container)
        entry_frame.pack(fill="x", pady=(0, 8))
        tk.Label(entry_frame, text="Transcription:").pack(side="left")
        text_widget = tk.Text(entry_frame, height=4, wrap="word")
        text_widget.pack(side="left", fill="both", expand=True, padx=(8, 0))
        text_widget.bind("<Key>", self._on_transcription_modified)
        text_widget.bind("<Control-Return>", self._on_confirm)
        text_widget.bind("<Command-Return>", self._on_confirm)
        self.entry_widget = text_widget

        buttons = tk.Frame(container)
        buttons.pack(pady=(0, 8))
        back_btn = tk.Button(buttons, text="Back", command=self.back)
        back_btn.pack(side="left", padx=4)
        self.back_button = back_btn
        tk.Button(buttons, text="Skip", command=self.skip).pack(side="left", padx=4)
        confirm_btn = tk.Button(buttons, text="Confirm", command=self.confirm, default=tk.ACTIVE)
        confirm_btn.pack(side="left", padx=4)
        tk.Button(buttons, text="Unsure", command=self.unsure).pack(side="left", padx=4)

        status_label = tk.Label(container, textvariable=self.status_var, fg="gray")
        status_label.pack(anchor="w")

        self.canvas.bind("<ButtonPress-1>", self._on_canvas_button_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self.master.bind("<Escape>", self._on_escape)
        self.master.bind("<Delete>", self._on_delete_selected)
        self.master.bind("<BackSpace>", self._on_delete_selected)

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
        return self.entry_widget.get("1.0", tk.END).strip()

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
        text = raw_text.replace("\r\n", "\n").rstrip("\n")

        overlays = sorted(self.overlay_items, key=lambda item: item.order_key)
        remaining = text
        previous_paragraph: Optional[Tuple[int, int, int]] = None
        previous_line: Optional[Tuple[int, int, int, int]] = None
        updated_tokens: List[OcrToken] = []

        for overlay in overlays:
            paragraph_key = overlay.order_key[:3]
            line_key = overlay.order_key[:4]

            if previous_paragraph is not None:
                if paragraph_key != previous_paragraph:
                    if remaining.startswith("\n\n"):
                        remaining = remaining[2:]
                    else:
                        remaining = remaining.lstrip("\n")
                elif line_key != previous_line:
                    if remaining.startswith("\n"):
                        remaining = remaining[1:]

            if "\n" in remaining:
                line_text, remaining = remaining.split("\n", 1)
            else:
                line_text, remaining = remaining, ""

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

            previous_paragraph = paragraph_key
            previous_line = line_key

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

        display = base.copy()
        display.thumbnail(self.MAX_SIZE, Image.LANCZOS)
        sx = display.width / base.width if base.width else 1.0
        sy = display.height / base.height if base.height else 1.0
        self.display_scale = (sx, sy)
        self._base_image = base

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
            tokens = []

        self._display_image(display, tokens)

        if item.label:
            self._set_transcription(item.label)
            self.status_var.set("Loaded existing transcription.")
        else:
            suggestion = self._suggest_label(item.path)
            if suggestion:
                self._set_transcription(suggestion)
                self.status_var.set("Pre-filled transcription using OCR result.")
            else:
                self._set_transcription(self._compose_text_from_tokens(tokens))
                if tokens:
                    self.status_var.set("Kraken returned baseline segmentation.")
                else:
                    self.status_var.set("Draw baselines manually using the canvas.")

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

    def _suggest_label(self, _path: Path) -> str:
        return ""

    def _extract_tokens(self, _image: Image.Image) -> List[OcrToken]:
        return []

    # ------------------------------------------------------------------
    # Canvas interaction
    # ------------------------------------------------------------------
    def _on_canvas_button_press(self, event: tk.Event) -> None:
        self._pressed_overlay = None
        self._modifier_drag = bool(event.state & (CONTROL_MASK | SHIFT_MASK))
        self._drag_start = (event.x, event.y)

        for overlay in self.overlay_items:
            left, top, right, bottom = self.canvas.coords(overlay.rect_id)
            if left <= event.x <= right and top <= event.y <= bottom:
                self._pressed_overlay = overlay
                if not self._modifier_drag:
                    self._clear_selection()
                self._select_overlay(overlay, additive=self._modifier_drag)
                return

        if self.mode_var.get() == "draw":
            self._start_manual_overlay(event.x, event.y)
        else:
            self._clear_selection()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._active_temp_rect is not None:
            x0, y0 = self._drag_start if self._drag_start is not None else (event.x, event.y)
            self.canvas.coords(self._active_temp_rect, x0, y0, event.x, event.y)

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self._active_temp_rect is not None:
            self._finish_manual_overlay(event.x, event.y)
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
        order_key = (9, 9, 9, 9, self.manual_token_counter)
        overlay = self._create_overlay("", base_bbox, order_key, None, is_manual=True, select=True)
        overlay.entry.focus_set()
        if hasattr(self, "status_var"):
            self.status_var.set("Added manual line")
        self._update_transcription_from_overlays()

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
        for rect_id in list(self.selected_rects):
            overlay = self.rect_to_overlay.pop(rect_id, None)
            if overlay is None:
                continue
            try:
                overlay.entry.destroy()
            except Exception:  # pragma: no cover
                pass
            try:
                self.canvas.delete(rect_id)
            except Exception:  # pragma: no cover - stub canvas may not implement delete
                pass
            if overlay in self.overlay_items:
                self.overlay_items.remove(overlay)
            if overlay.entry in self.overlay_entries:
                self.overlay_entries.remove(overlay.entry)
        self.selected_rects.clear()
        if hasattr(self, "delete_button"):
            self.delete_button.config(state=tk.DISABLED)
        self._update_transcription_from_overlays()

    def _on_delete_selected(self, _event: Optional[tk.Event]) -> None:
        self._delete_selected()

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

    def _append_log(self, path: Path, label: str, status: str, saved_path: Optional[Path]) -> None:
        if self.log_path is None:
            return
        is_new = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(["filename", "label", "status", "saved_path"])
            writer.writerow([path.name, label, status, saved_path.name if saved_path else ""])


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

    app = AnnotationApp(root, items, Path(train_dir), options=options, log_path=log_path, on_sample_saved=callback)
    root.mainloop()
