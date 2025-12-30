from __future__ import annotations

"""Tkinter application for creating OCR training annotations."""

import csv
import logging
import threading
import difflib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import tkinter as tk
from tkinter import messagebox

try:  # pragma: no cover - optional dependency for auto-segmentation
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

from PIL import Image, ImageOps, ImageTk

try:  # pragma: no cover - allow running as a package or script
    from .exporters import save_line_crops, save_pagexml
    from .kraken_adapter import (
        is_available as kraken_available,
        ocr_to_string,
    )
    from .line_store import Line
    from .transcripts import load_transcript, save_transcript
    from .xschema import (
        AlignmentPlanner,
        PlannerConfig,
        XApplier,
        LayoutSpan,
        build_blocks_for_page,
        extract_layout_spans_from_overlays,
        extract_transcript_spans,
    )
except ImportError:  # pragma: no cover
    from exporters import save_line_crops, save_pagexml
    from kraken_adapter import (
        is_available as kraken_available,
        ocr_to_string,
    )
    from line_store import Line
    from transcripts import load_transcript, save_transcript  # type: ignore
    from xschema import (  # type: ignore
        AlignmentPlanner,
        PlannerConfig,
        XApplier,
        LayoutSpan,
        build_blocks_for_page,
        extract_layout_spans_from_overlays,
        extract_transcript_spans,
    )


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
    pagexml_dir: Optional[Path] = None


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
    entry_frame: Optional[tk.Frame] = None
    window_id: Optional[int] = None

    @property
    def text(self) -> str:
        return self.entry.get()


try:  # pragma: no cover
    from .annotation_io import load_tokens
except ImportError:  # pragma: no cover
    from annotation_io import load_tokens  # type: ignore


class _FallbackStringVar:
    def __init__(self) -> None:
        self._value = ""

    def set(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _FallbackEntry:
    def __init__(self, value: str = "") -> None:
        self.value = value
        self._bindings: Dict[str, object] = {}

    def insert(self, _index: object, text: str) -> None:
        self.value = text

    def delete(self, *_args: object, **_kwargs: object) -> None:
        self.value = ""

    def get(self, *_args: object, **_kwargs: object) -> str:
        return self.value

    def bind(self, sequence: str, handler) -> None:
        self._bindings[sequence] = handler

    def focus_set(self) -> None:  # pragma: no cover - no focus in tests
        pass

    def destroy(self) -> None:  # pragma: no cover - nothing to clean
        pass


class _FallbackCanvas:
    def __init__(self) -> None:
        self._next_id = 1
        self._objects: Dict[int, Dict[str, object]] = {}

    def _allocate_id(self) -> int:
        obj_id = self._next_id
        self._next_id += 1
        return obj_id

    def create_rectangle(self, x1, y1, x2, y2, **_kwargs):
        obj_id = self._allocate_id()
        self._objects[obj_id] = {"coords": [x1, y1, x2, y2]}
        return obj_id

    def create_window(self, x, y, **_kwargs):
        obj_id = self._allocate_id()
        self._objects[obj_id] = {"coords": [x, y]}
        return obj_id

    def delete(self, target):
        if target == "all":
            self._objects.clear()
            return
        self._objects.pop(target, None)

    def coords(self, obj_id, *values):
        obj = self._objects.setdefault(obj_id, {"coords": list(values) if values else []})
        if values:
            obj["coords"] = list(values)
        return list(obj.get("coords", []))

    def itemconfig(self, *_args, **_kwargs):
        return None

    def config(self, **_kwargs):
        return None

    def yview_moveto(self, *_args):  # pragma: no cover - not used in tests
        return None

    def xview_moveto(self, *_args):  # pragma: no cover - not used in tests
        return None


def _load_annotation_items(paths: Iterable[Path]) -> List[AnnotationItem]:
    return [AnnotationItem(Path(p)) for p in paths]


def _train_model(*args, **kwargs):  # pragma: no cover - runtime helper
    try:
        from .training import train_model
    except ImportError:
        from training import train_model  # type: ignore
    return train_model(*args, **kwargs)


class AnnotationTrainer:
    """Schedule background training after every N samples."""

    def __init__(self, master, train_dir: Path, config: "AnnotationAutoTrainConfig") -> None:
        self.master = master
        self.train_dir = train_dir
        self.config = config
        self.seen_samples: List[Path] = []

    def __call__(self, sample_path: Path) -> None:
        self.seen_samples.append(sample_path)
        if len(self.seen_samples) % max(1, self.config.auto_train) != 0:
            return

        def runner() -> None:
            _train_model(
                train_dir=self.train_dir,
                output_model=self.config.output_model,
                model_dir=self.config.model_dir,
                base_lang=self.config.base_lang,
                max_iterations=self.config.max_iterations,
                tessdata_dir=self.config.tessdata_dir,
                use_gpt_ocr=self.config.use_gpt_ocr,
                gpt_model=self.config.gpt_model,
                gpt_prompt=self.config.gpt_prompt,
                gpt_cache_dir=self.config.gpt_cache_dir,
                gpt_max_output_tokens=self.config.gpt_max_output_tokens,
                gpt_max_images=self.config.gpt_max_images,
                resume=self.config.resume,
                deserialize_check_limit=self.config.deserialize_check_limit,
                unicharset_size_override=self.config.unicharset_size_override,
            )

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        try:
            self.master.after(0, lambda: None)
        except Exception:  # pragma: no cover - defensive guard
            pass


class AnnotationApp:
    MAX_SIZE = (1800, 1200)

    def __init__(
        self,
        master,
        items: Sequence[AnnotationItem],
        train_dir: Path,
        *,
        options: Optional[AnnotationOptions] = None,
        log_path: Optional[Path] = None,
        on_sample_saved: Optional[Callable[[Path], None]] = None,
        transcripts_dir: Optional[Path] = None,
    ) -> None:
        self.master = master
        self.items = list(items)
        self.index = 0
        self.train_dir = Path(train_dir)
        self.options = options or AnnotationOptions()
        self.log_path = log_path
        self._on_sample_saved = on_sample_saved
        if transcripts_dir is not None:
            self.transcripts_dir = Path(transcripts_dir)
        else:
            self.transcripts_dir = None
            default_dir = Path(__file__).resolve().parent.parent / "data" / "train" / "images"
            if default_dir.exists():
                self.transcripts_dir = default_dir

        self.overlay_items: List[OverlayItem] = []
        self.overlay_entries: List[tk.Entry] = []
        self.rect_to_overlay: Dict[int, OverlayItem] = {}
        self.selected_rects: set[int] = set()
        self.current_tokens: List[OcrToken] = []

        self.display_scale = (1.0, 1.0)
        self.manual_token_counter = 0
        self._active_temp_rect: Optional[int] = None
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_overlay_start_bbox: Optional[Tuple[float, float, float, float]] = None
        self._marquee_rect: Optional[int] = None
        self._modifier_drag = False
        self._pressed_overlay: Optional[OverlayItem] = None
        self._user_modified_transcription = False
        self._setting_transcription = False
        self._syncing_overlays = False
        self._undo_stack: List[List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int, int], bool]]] = []

        master.minsize(1400, 800)

        # UI widgets (minimal layout)
        right = tk.Frame(master, width=440, bg="#f5f5f5")
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        left = tk.Frame(master)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(left, bg="white", width=self.MAX_SIZE[0], height=self.MAX_SIZE[1])
        y_scroll = tk.Scrollbar(left, orient=tk.VERTICAL, command=self.canvas.yview)
        x_scroll = tk.Scrollbar(left, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.filename_var = tk.StringVar()
        self.status_var = tk.StringVar()
        tk.Label(right, textvariable=self.filename_var, bg="#f5f5f5", anchor="w").pack(fill=tk.X)
        tk.Label(right, textvariable=self.status_var, bg="#f5f5f5", wraplength=400, anchor="w", justify=tk.LEFT).pack(fill=tk.X)
        tk.Label(right, text="Transcript (gt.txt)", bg="#f5f5f5", anchor="w").pack(fill=tk.X)
        self.entry_widget = tk.Text(right, width=64, height=12)
        self.entry_widget.pack(fill=tk.BOTH, expand=False)
        self.entry_widget.bind("<<Modified>>", self._on_transcription_modified)

        tk.Label(right, text="Lines", bg="#f5f5f5", anchor="w").pack(fill=tk.X, pady=(6, 0))
        self.lines_list = tk.Text(right, width=64, height=14, wrap=tk.NONE)
        self.lines_list.configure(cursor="arrow")
        self.lines_list.pack(fill=tk.BOTH, expand=True)
        self.lines_list.bind("<Key>", lambda _e: "break")
        self.lines_list.bind("<Button-1>", self._on_lines_click)
        self._line_tag_names: Dict[int, str] = {}
        self._line_tag_to_overlay: Dict[str, OverlayItem] = {}

        controls = tk.Frame(right)
        controls.pack(fill=tk.X)
        self.undo_button = tk.Button(controls, text="Undo", command=self._on_undo, state=tk.DISABLED)
        self.undo_button.pack(side=tk.LEFT)
        self.delete_button = tk.Button(controls, text="Delete", command=self._delete_selected, state=tk.DISABLED)
        self.delete_button.pack(side=tk.LEFT)
        self.back_button = tk.Button(controls, text="Back", command=self.back, state=tk.DISABLED)
        self.back_button.pack(side=tk.LEFT)
        tk.Button(controls, text="Skip", command=self.skip).pack(side=tk.LEFT)
        tk.Button(controls, text="Unsure", command=self.unsure).pack(side=tk.LEFT)
        tk.Button(controls, text="Confirm", command=self.confirm).pack(side=tk.LEFT)

        self.mode_var = tk.StringVar(value="select")
        mode_frame = tk.Frame(right)
        mode_frame.pack(fill=tk.X)
        tk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Select", variable=self.mode_var, value="select", command=self._on_mode_change).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Draw", variable=self.mode_var, value="draw", command=self._on_mode_change).pack(side=tk.LEFT)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_button_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel_scroll)
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)
        master.bind("<Return>", self._on_confirm)
        master.bind("<Escape>", self._on_escape)
        master.bind("<Delete>", self._on_delete_selected)
        master.bind("<Control-z>", self._on_undo)

        self._show_current()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _slugify(self, value: str) -> str:
        slug = "-".join(value.strip().lower().split())
        slug = slug.replace("_", "-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug[:60].rstrip("-")

    def _safe_messagebox(self, name: str, *args, **kwargs):
        func = getattr(messagebox, name)
        try:
            return func(*args, **kwargs)
        except tk.TclError:
            return None

    def _ensure_runtime_defaults(self) -> None:
        if not hasattr(self, "_undo_stack"):
            self._undo_stack: List[List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int, int], bool]]] = []
        if not hasattr(self, "_on_sample_saved"):
            self._on_sample_saved = None
        if not hasattr(self, "status_var"):
            self.status_var = _FallbackStringVar()
        if not hasattr(self, "filename_var"):
            self.filename_var = _FallbackStringVar()
        if not hasattr(self, "options"):
            self.options = AnnotationOptions()
        if not hasattr(self, "transcripts_dir"):
            self.transcripts_dir = None
        if not hasattr(self, "_line_tag_names"):
            self._line_tag_names = {}
        if not hasattr(self, "_line_tag_to_overlay"):
            self._line_tag_to_overlay = {}

    def _render_lines_list(self) -> None:
        try:
            self.lines_list.delete("1.0", tk.END)
            self._line_tag_names = {}
            self._line_tag_to_overlay = {}
            for idx, overlay in enumerate(sorted(self.overlay_items, key=lambda o: o.order_key), start=1):
                text = overlay.entry.get().replace("\r", " ")
                display = text.replace("\n", " ⏎ ")
                start_index = self.lines_list.index(tk.END)
                self.lines_list.insert(tk.END, f"{idx:02d}: {display}\n")
                end_index = self.lines_list.index(tk.END)
                tag_name = f"line-{overlay.rect_id}"
                self.lines_list.tag_add(tag_name, start_index, end_index)
                self._line_tag_names[overlay.rect_id] = tag_name
                self._line_tag_to_overlay[tag_name] = overlay
            self._update_lines_highlight()
        except Exception:
            pass

    def _update_lines_highlight(self) -> None:
        try:
            for overlay in self.overlay_items:
                tag_name = self._line_tag_names.get(overlay.rect_id)
                if not tag_name:
                    continue
                bg = "#fdeaca" if overlay.selected else ""
                self.lines_list.tag_configure(tag_name, background=bg)
        except Exception:
            pass

    def _on_lines_click(self, event) -> str:
        try:
            index = self.lines_list.index(f"@{event.x},{event.y}")
        except Exception:
            return "break"
        overlay: Optional[OverlayItem] = None
        for tag in self.lines_list.tag_names(index):
            overlay = self._line_tag_to_overlay.get(tag)
            if overlay is not None:
                break
        if overlay is None:
            return "break"
        if event.state & CONTROL_MASK:
            self._toggle_selection(overlay)
        else:
            self._select_single_line(overlay)
        return "break"

    def _snapshot_overlays(self) -> List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int, int], bool]]:
        snapshots: List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int, int], bool]] = []
        for ov in sorted(self.overlay_items, key=lambda o: o.order_key):
            text_getter = getattr(ov.entry, "get", None)
            if callable(text_getter):
                text = text_getter()
            else:
                text = getattr(ov.entry, "value", "")
            snapshots.append((text, ov.bbox, ov.order_key, ov.is_manual))
        return snapshots

    def _push_undo(
        self,
        snapshot: Optional[List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int, int], bool]]] = None,
    ) -> None:
        self._ensure_runtime_defaults()
        if snapshot is None:
            snapshot = self._snapshot_overlays()
        self._undo_stack.append(snapshot)
        if hasattr(self, "undo_button"):
            self.undo_button.config(state=tk.NORMAL if self._undo_stack else tk.DISABLED)

    def _restore_snapshot(self, snapshot: List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int, int], bool]]) -> None:
        self.canvas.delete("all")
        self._clear_overlay_entries()
        self.overlay_items.clear()
        self.rect_to_overlay.clear()
        self.selected_rects.clear()
        self._render_base_image(keep_overlays=False)
        for text, bbox, order_key, is_manual in snapshot:
            self._create_overlay(text, bbox, order_key, None, is_manual=is_manual, select=False)
        self._update_transcription_from_overlays()

    def _on_undo(self, _event: Optional[tk.Event] = None) -> None:
        self._ensure_runtime_defaults()
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        if hasattr(self, "undo_button"):
            self.undo_button.config(state=tk.NORMAL if self._undo_stack else tk.DISABLED)
        self._restore_snapshot(snapshot)

    def _set_zoom(self, zoom: float) -> None:
        self._zoom_level = max(0.2, min(3.5, zoom))
        self._render_base_image(keep_overlays=True)

    def _on_mode_change(self) -> None:
        self._ensure_runtime_defaults()
        mode = self.mode_var.get()
        cursor = "crosshair" if mode == "draw" else "arrow"
        try:
            self.canvas.config(cursor=cursor)
        except Exception:
            pass
        if mode == "draw":
            self.status_var.set("Draw: click-drag to add a line box. Press Esc to cancel.")
        else:
            self.status_var.set("Select: click to select a box; drag to move; Delete to remove.")

    def _on_mousewheel_scroll(self, event) -> None:
        # Windows uses event.delta in multiples of 120
        delta = int(-1 * (event.delta / 120))
        try:
            if event.state & CONTROL_MASK:
                # Ctrl+wheel handled separately for zoom
                return
            self.canvas.yview_scroll(delta, "units")
        except Exception:
            pass

    def _on_zoom(self, event) -> None:
        step = 1.1 if event.delta > 0 else 0.9
        self._set_zoom(self._zoom_level * step)

    def _get_transcription_text(self) -> str:
        raw = self.entry_widget.get("1.0", tk.END)
        return raw.rstrip("\n")

    def _set_transcription(self, text: str) -> None:
        self._setting_transcription = True
        try:
            self.entry_widget.delete("1.0", tk.END)
            self.entry_widget.insert("1.0", text)
            marker = getattr(self.entry_widget, "edit_modified", None)
            if callable(marker):
                marker(False)
        finally:
            self._setting_transcription = False

    # ------------------------------------------------------------------
    # Prefill helpers
    # ------------------------------------------------------------------
    def _suggest_label(self, image_path: Path) -> str:
        if self.options.prefill_model and kraken_available():
            return ocr_to_string(image_path, self.options.prefill_model)

        if self.options.prefill_tessdata and cv2 is not None:
            return ""

        impl = _get_prefill_ocr()
        if impl is None:
            return ""
        try:
            result = impl(image_path, psm=self.options.prefill_psm)  # type: ignore[arg-type]
        except Exception:
            return ""
        return result or ""

    def _get_prefill(self, item: AnnotationItem) -> str:
        if not self.options.prefill_enabled:
            return ""
        if item.prefill is not None:
            return item.prefill
        item.prefill = self._suggest_label(item.path)
        return item.prefill

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _on_confirm(self, _event: Optional[tk.Event]) -> None:
        self.confirm()

    def _on_escape(self, _event: Optional[tk.Event]) -> None:
        if self._active_temp_rect is not None:
            self.canvas.delete(self._active_temp_rect)
            self._active_temp_rect = None
            self._drag_start = None
        else:
            self._clear_selection()

    def _on_delete_selected(self, _event: Optional[tk.Event]) -> None:
        self._delete_selected()

    def confirm(self) -> None:
        self._ensure_runtime_defaults()
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
        if self._on_sample_saved is not None and saved_path is not None:
            self._on_sample_saved(saved_path)
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
        self._ensure_runtime_defaults()
        item = self.items[self.index]
        self.filename_var.set(f"{item.path.name} ({self.index + 1}/{len(self.items)})")
        self.back_button.config(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self._display_item(item)
        self.entry_widget.focus_set()
        if revisit:
            status = self.status_var.get()
            extra: List[str] = []
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
        self._base_image = base
        self._fit_scale = fit_scale
        self._zoom_level = 1.0
        self._undo_stack: List[Callable[[], None]] = []
        self._render_base_image(keep_overlays=False)

        tokens, pagexml_used, status_msg = load_tokens(
            item,
            self.options,
            self._baseline_to_bbox,
            lambda img: self._extract_tokens(img),
            base,
            token_factory=OcrToken,
        )
        if status_msg:
            self.status_var.set(status_msg)

        self._display_image(tokens)

        transcript_text = self._load_page_transcript(item)
        if transcript_text:
            self._set_transcription(transcript_text)
            self._apply_transcription_to_overlays()
            self.status_var.set("Loaded transcript from gt.txt.")
            return

        if item.label:
            self._set_transcription(item.label)
            self._apply_transcription_to_overlays()
            self.status_var.set("Loaded existing transcription.")
            return

        if pagexml_used and tokens:
            fallback = self._compose_text_from_tokens(tokens)
            self._set_transcription(fallback)
            self._apply_transcription_to_overlays()
            self.status_var.set("Loaded text from PAGE-XML boxes.")
            return

        should_prefill = not (self.options.segmentation == "load" and self.options.pagexml_dir)
        suggestion = self._get_prefill(item) if should_prefill else ""
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
            self.mode_var.set("draw")
            self._on_mode_change()
            self.status_var.set("No boxes found. Switch to Draw mode and drag to add line boxes.")

    def _render_base_image(self, *, keep_overlays: bool) -> None:
        if not hasattr(self, "_base_image"):
            return
        base = self._base_image
        scale = max(0.1, min(4.0, self._fit_scale * self._zoom_level))
        display_size = (
            max(1, int(base.width * scale)),
            max(1, int(base.height * scale)),
        )
        display = base.resize(display_size, Image.LANCZOS)
        sx = display.width / base.width if base.width else 1.0
        sy = display.height / base.height if base.height else 1.0
        self.display_scale = (sx, sy)

        try:
            self.current_photo = ImageTk.PhotoImage(display)
        except (tk.TclError, RuntimeError):
            self.current_photo = None
            return

        if not hasattr(self, "canvas"):
            return

        if keep_overlays and hasattr(self, "canvas_image_id"):
            try:
                self.canvas.itemconfig(self.canvas_image_id, image=self.current_photo)
            except Exception:
                keep_overlays = False

        if not keep_overlays:
            self.canvas.delete("all")
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.current_photo)

        self.canvas.config(scrollregion=(0, 0, display.width, display.height))
        # Keep the visible area stable around the last mouse position
        try:
            self.canvas.yview_moveto(0)
            self.canvas.xview_moveto(0)
        except Exception:
            pass
        if keep_overlays:
            self._refresh_all_overlay_positions()

    def _display_image(self, arg1, arg2: Optional[Sequence[OcrToken]] = None) -> None:
        if arg2 is None:
            tokens: Sequence[OcrToken] = arg1
        else:
            tokens = arg2
        self._ensure_runtime_defaults()
        self.canvas.delete("all")
        self._clear_overlay_entries()
        self.overlay_items.clear()
        self.rect_to_overlay.clear()
        self.selected_rects.clear()
        self.current_tokens = list(tokens)
        self._undo_stack.clear()
        if hasattr(self, "undo_button"):
            self.undo_button.config(state=tk.DISABLED)

        self._render_base_image(keep_overlays=False)

        # Keep sidebar visible and ensure canvas uses a sensible cursor for the active mode
        self._on_mode_change()

        if not tokens:
            self.status_var.set("No boxes loaded. Drag on the image to draw a box, then type the line in the sidebar text area.")

        for token in tokens:
            self._create_overlay(token.text, token.bbox, token.order_key, token)
        self._update_transcription_from_overlays()

    # ------------------------------------------------------------------
    # Overlay creation and editing
    # ------------------------------------------------------------------
    def _scale_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        sx, sy = self.display_scale if hasattr(self, "display_scale") else (1.0, 1.0)
        x1, y1, x2, y2 = bbox
        return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

    def _to_base_bbox(self, coords: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        sx, sy = self.display_scale if hasattr(self, "display_scale") else (1.0, 1.0)
        x1, y1, x2, y2 = coords
        if sx == 0 or sy == 0:
            return (0, 0, 0, 0)
        left, top, right, bottom = x1 / sx, y1 / sy, x2 / sx, y2 / sy
        return (int(left), int(top), int(right), int(bottom))

    def _clone_token(self, token: OcrToken) -> OcrToken:
        return OcrToken(token.text, token.bbox, token.order_key, token.baseline, token.origin)

    def _calculate_order_key_for_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, int]:
        return (1, 1, 1, len(self.overlay_items) + 1, 1)

    def _estimate_bbox_for_new_line(self, index: int) -> Tuple[int, int, int, int]:
        if self.overlay_items:
            last = self.overlay_items[min(index, len(self.overlay_items) - 1)]
            x1, y1, x2, y2 = last.bbox
            height = max(1, y2 - y1)
            offset = height + 5
            return (x1, y2 + offset, x2, y2 + offset + height)
        if hasattr(self, "_base_image"):
            return (0, 0, self._base_image.width, max(10, self._base_image.height // 20))
        return (0, 0, 10, 10)

    def _create_overlay(
        self,
        text: str,
        bbox: Tuple[int, int, int, int],
        order_key: Tuple[int, int, int, int, int],
        token: Optional[OcrToken],
        *,
        is_manual: bool = False,
        select: bool = False,
        has_box: bool = True,
    ) -> OverlayItem:
        if not hasattr(self, "canvas") or self.canvas is None:
            self.canvas = _FallbackCanvas()
        canvas = self.canvas
        scaled = self._scale_bbox(bbox)
        if has_box:
            rect_id = canvas.create_rectangle(*scaled, outline="red", width=2)
        else:
            rect_id = canvas.create_rectangle(*scaled, outline="red", width=0)

        frame = None
        window_id = None
        try:
            frame = tk.Frame(canvas, bg="white", highlightthickness=0, bd=0)
        except Exception:
            frame = None

        try:
            parent = frame if frame is not None else self.canvas
            entry = tk.Entry(parent, width=40, relief=tk.FLAT)
        except Exception:
            entry = _FallbackEntry()
        entry.insert(0, text)
        if frame is not None:
            try:
                entry.pack(fill=tk.X, padx=2, pady=1)
            except Exception:
                pass
            try:
                window_id = canvas.create_window(
                    scaled[0],
                    max(0, scaled[1] - 24),
                    anchor="nw",
                    window=frame,
                )
            except Exception:
                window_id = None

        overlay = OverlayItem(
            rect_id,
            entry,
            bbox,
            order_key,
            token,
            is_manual=is_manual,
            selected=False,
            entry_frame=frame,
            window_id=window_id,
        )
        entry.bind("<KeyRelease>", self._on_overlay_modified)
        entry.bind("<FocusIn>", lambda _e, ov=overlay: self._select_single_line(ov))
        self.overlay_items.append(overlay)
        self.rect_to_overlay[rect_id] = overlay
        self.overlay_entries.append(entry)
        self._refresh_overlay_visuals(overlay)
        if select:
            self._select_single_line(overlay)
        return overlay

    def _destroy_overlay(self, overlay: OverlayItem) -> None:
        try:
            self.canvas.delete(overlay.rect_id)
        except Exception:
            pass
        if overlay.window_id is not None:
            try:
                self.canvas.delete(overlay.window_id)
            except Exception:
                pass
        if overlay.entry_frame is not None:
            try:
                overlay.entry_frame.destroy()
            except Exception:
                pass
        destroy_entry = getattr(overlay.entry, "destroy", None)
        if callable(destroy_entry):
            try:
                destroy_entry()
            except Exception:
                pass
        if overlay.entry in self.overlay_entries:
            self.overlay_entries.remove(overlay.entry)
        if overlay in self.overlay_items:
            self.overlay_items.remove(overlay)
        self.rect_to_overlay.pop(overlay.rect_id, None)
        self.selected_rects.discard(overlay.rect_id)

    def _clear_overlay_entries(self) -> None:
        for overlay in list(self.overlay_items):
            if overlay.window_id is not None:
                try:
                    self.canvas.delete(overlay.window_id)
                except Exception:
                    pass
            if overlay.entry_frame is not None:
                try:
                    overlay.entry_frame.destroy()
                except Exception:
                    pass
        for entry in list(self.overlay_entries):
            try:
                entry.destroy()
            except Exception:
                pass
        self.overlay_entries.clear()

    def _reset_entry_value(self, entry) -> None:
        deleter = getattr(entry, "delete", None)
        if callable(deleter):
            try:
                deleter(0, tk.END)
                return
            except TypeError:
                try:
                    deleter(0, "end")
                    return
                except Exception:
                    pass
            except Exception:
                pass
        if hasattr(entry, "value"):
            entry.value = ""

    def _reorder_overlays(self) -> None:
        self.overlay_items.sort(key=lambda o: o.order_key)
        self.overlay_entries = [overlay.entry for overlay in self.overlay_items]

    def _prune_empty_overlays(self) -> None:
        for overlay in list(self.overlay_items):
            if overlay.entry.get().strip():
                continue
            if overlay.is_manual:
                self._destroy_overlay(overlay)

    def _update_entry_window_for_coords(self, overlay: OverlayItem, coords: Tuple[float, float, float, float]) -> None:
        if overlay.window_id is None:
            return
        x1, y1, _x2, _y2 = coords
        entry_y = max(0, y1 - 24)
        self.canvas.coords(overlay.window_id, x1, entry_y)

    def _update_overlay_position(self, overlay: OverlayItem) -> None:
        coords = self._scale_bbox(overlay.bbox)
        self.canvas.coords(overlay.rect_id, *coords)
        if overlay.entry_frame is not None:
            try:
                overlay.entry_frame.config(width=max(80, int(coords[2] - coords[0])))
            except Exception:
                pass
        self._update_entry_window_for_coords(overlay, coords)

    def _update_overlay_style(self, overlay: OverlayItem) -> None:
        outline = "blue" if overlay.selected else "red"
        width = 2 if overlay.selected else 1
        try:
            self.canvas.itemconfig(overlay.rect_id, outline=outline, width=width)
        except Exception:
            pass
        bg = "#e7f0ff" if overlay.selected else "white"
        if overlay.entry_frame is not None:
            try:
                overlay.entry_frame.config(bg=bg)
            except Exception:
                pass
        try:
            overlay.entry.config(bg=bg)
        except Exception:
            pass

    def _refresh_overlay_visuals(self, overlay: OverlayItem) -> None:
        self._update_overlay_position(overlay)
        self._update_overlay_style(overlay)

    def _refresh_overlay_styles(self) -> None:
        for overlay in self.overlay_items:
            self._update_overlay_style(overlay)

    def _update_transcription_from_overlays(self) -> None:
        tokens: List[OcrToken] = []
        overlays = sorted(self.overlay_items, key=lambda item: item.order_key)
        for overlay in overlays:
            text = overlay.entry.get()
            if overlay.token:
                token = self._clone_token(overlay.token)
                token.text = text
            else:
                token = OcrToken(
                    text=text,
                    bbox=overlay.bbox,
                    order_key=overlay.order_key,
                    baseline=(0, 0, 0),
                    origin=(0, 0, 0),
                )
            tokens.append(token)
        self.current_tokens = tokens
        self._set_transcription(self._compose_text_from_tokens(tokens))
        self._render_lines_list()

    def _line_similarity(self, a: str, b: str) -> float:
        """Return a normalized similarity score between two lines of text.

        Uses difflib.SequenceMatcher on lowercased, whitespace-stripped text.
        Falls back to 0.0 if either side is empty or an error occurs.
        """

        a_norm = "".join(a.lower().split())
        b_norm = "".join(b.lower().split())
        if not a_norm or not b_norm:
            return 0.0
        try:
            matcher = difflib.SequenceMatcher(None, a_norm, b_norm)
            return float(matcher.ratio())
        except Exception:
            return 0.0

    def _apply_transcription_to_overlays(self, *_args, **_kwargs) -> None:
        if not hasattr(self, "_syncing_overlays"):
            self._syncing_overlays = False
        if self._syncing_overlays:
            return
        self._syncing_overlays = True
        try:
            self._ensure_runtime_defaults()
            raw_text = self.entry_widget.get("1.0", tk.END)
            normalized = raw_text.replace("\r\n", "\n").rstrip("\n")
            segments = [s for s in normalized.split("\n") if s.strip()]

            # If there are no overlays yet, build them from the transcript lines
            if not self.overlay_items:
                for entry in list(getattr(self, "overlay_entries", [])):
                    self._reset_entry_value(entry)
                self.overlay_entries = []
                updated_tokens: List[OcrToken] = []
                for idx, text in enumerate(segments):
                    bbox = self._estimate_bbox_for_new_line(idx)
                    order_key = self._calculate_order_key_for_bbox(bbox)
                    overlay = self._create_overlay(text, bbox, order_key, None, is_manual=True, select=False)
                    if text:
                        token = OcrToken(
                            text=text,
                            bbox=bbox,
                            order_key=order_key,
                            baseline=(0, 0, 0),
                            origin=(0, 0, 0),
                        )
                        updated_tokens.append(token)
                self.current_tokens = updated_tokens
                self._reorder_overlays()
                self._render_lines_list()
                return

            overlays = sorted(
                self.overlay_items,
                key=lambda item: (((item.bbox[1] + item.bbox[3]) // 2), item.order_key),
            )

            if not segments:
                for overlay in overlays:
                    overlay.entry.delete(0, tk.END)
                self.current_tokens = []
                self._render_lines_list()
                return

            options = getattr(self, "options", None)
            segmentation_mode = getattr(options, "segmentation", None)
            pagexml_dir = getattr(options, "pagexml_dir", None)
            use_text_scoring = bool(segmentation_mode == "load" and pagexml_dir)
            for overlay in overlays:
                entry = getattr(overlay, "entry", None)
                if entry is None or hasattr(entry, "get"):
                    continue
                setattr(entry, "get", lambda entry=entry: getattr(entry, "value", ""))

            page_id = "page"
            try:
                if getattr(self, "items", None):
                    item = self.items[self.index]
                    page_id = Path(getattr(item, "path", "page")).stem
            except Exception:
                pass

            transcript_spans = extract_transcript_spans(page_id, normalized)
            layout_spans = extract_layout_spans_from_overlays(page_id, overlays)
            block_source: Literal["pagexml", "kraken", "manual", "mixed", "legacy"] = "legacy"
            if segmentation_mode == "manual":
                block_source = "manual"
            elif segmentation_mode == "load":
                block_source = "pagexml"
            elif segmentation_mode == "auto":
                block_source = "kraken"
            blocks = build_blocks_for_page(page_id, transcript_spans, layout_spans, source=block_source)

            planner_mode: Literal["pagexml_trusted", "auto", "manual"] = "auto"
            if segmentation_mode == "manual":
                planner_mode = "manual"
            elif segmentation_mode == "load":
                planner_mode = "pagexml_trusted"

            planner = AlignmentPlanner(
                PlannerConfig(
                    use_text_scoring=use_text_scoring,
                    segmentation_mode=planner_mode,
                ),
                self._line_similarity,
            )
            plan = planner.plan(page_id, blocks)

            layout_map = {span.id: span.overlay_item for span in layout_spans if span.overlay_item is not None}
            layout_metadata: Dict[str, LayoutSpan] = {}
            for block in plan.blocks:
                for span in block.layout_spans:
                    layout_metadata[span.id] = span
            for link in plan.links:
                for span in link.layouts:
                    layout_metadata.setdefault(span.id, span)

            context = _AnnotationXContext(self, layout_map, layout_metadata)
            applier = XApplier(context)
            applier.apply(plan.operations)

            updated_tokens: List[OcrToken] = []
            for overlay in sorted(self.overlay_items, key=lambda item: item.order_key):
                text = overlay.entry.get()
                if not text:
                    continue
                token = self._clone_token(overlay.token) if overlay.token else OcrToken(
                    text=text,
                    bbox=overlay.bbox,
                    order_key=overlay.order_key,
                    baseline=(0, 0, 0),
                    origin=(0, 0, 0),
                )
                token.text = text
                token.bbox = overlay.bbox
                token.order_key = overlay.order_key
                updated_tokens.append(token)

            self.current_tokens = updated_tokens
            self._reorder_overlays()
            self._prune_empty_overlays()
            self.overlay_entries = [overlay.entry for overlay in self.overlay_items]
            self._render_lines_list()
        finally:
            self._syncing_overlays = False

    def _on_transcription_modified(self, _event: Optional[tk.Event]) -> None:
        if self._setting_transcription:
            self.entry_widget.edit_modified(False)
            return
        self.entry_widget.edit_modified(False)
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

    # ------------------------------------------------------------------
    # Canvas interactions
    # ------------------------------------------------------------------
    def _clear_selection(self) -> None:
        for overlay in self.overlay_items:
            overlay.selected = False
        self.selected_rects.clear()
        self.delete_button.config(state=tk.DISABLED)
        self._refresh_overlay_styles()
        self._update_lines_highlight()

    def _refresh_all_overlay_positions(self) -> None:
        for overlay in self.overlay_items:
            self._refresh_overlay_visuals(overlay)

    def _select_single_line(self, overlay: OverlayItem) -> None:
        self._clear_selection()
        overlay.selected = True
        self.selected_rects.add(overlay.rect_id)
        self.delete_button.config(state=tk.NORMAL)
        self._refresh_overlay_styles()
        self._update_lines_highlight()

    def _toggle_selection(self, overlay: OverlayItem) -> None:
        if overlay.selected:
            overlay.selected = False
            self.selected_rects.discard(overlay.rect_id)
        else:
            overlay.selected = True
            self.selected_rects.add(overlay.rect_id)
        self.delete_button.config(state=tk.NORMAL if self.selected_rects else tk.DISABLED)
        self._refresh_overlay_styles()
        self._update_lines_highlight()

    def _overlay_at_point(self, x: float, y: float) -> Optional[OverlayItem]:
        for overlay in self.overlay_items:
            x1, y1, x2, y2 = self.canvas.coords(overlay.rect_id)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return overlay
        return None

    def _on_canvas_button_press(self, event) -> None:
        self.canvas.focus_set()
        self._drag_overlay_start_bbox = None
        if self.mode_var.get() == "draw":
            self._drag_start = (event.x, event.y)
            self._active_temp_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="blue")
            return

        overlay = self._overlay_at_point(event.x, event.y)
        self._pressed_overlay = overlay
        if overlay is not None:
            self._drag_start = (event.x, event.y)
            self._drag_overlay_start_bbox = tuple(self.canvas.coords(overlay.rect_id))  # type: ignore[assignment]
            if event.state & CONTROL_MASK:
                self._toggle_selection(overlay)
            else:
                self._select_single_line(overlay)
            return
        self._clear_selection()

    def _on_canvas_drag(self, event) -> None:
        if self.mode_var.get() == "draw" and self._active_temp_rect is not None and self._drag_start:
            x0, y0 = self._drag_start
            self.canvas.coords(self._active_temp_rect, x0, y0, event.x, event.y)
            return

        if (
            self.mode_var.get() == "select"
            and self._pressed_overlay is not None
            and self._drag_start is not None
            and getattr(self, "_drag_overlay_start_bbox", None) is not None
        ):
            start_bbox = self._drag_overlay_start_bbox  # type: ignore[attr-defined]
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            new_bbox = (
                start_bbox[0] + dx,
                start_bbox[1] + dy,
                start_bbox[2] + dx,
                start_bbox[3] + dy,
            )
            self.canvas.coords(self._pressed_overlay.rect_id, *new_bbox)
            self._update_entry_window_for_coords(self._pressed_overlay, new_bbox)

    def _finish_manual_overlay(self, end_x: float, end_y: float) -> None:
        if self._active_temp_rect is None or self._drag_start is None:
            return
        self._push_undo(None)
        x0, y0 = self._drag_start
        coords = self.canvas.coords(self._active_temp_rect)
        if not coords:
            coords = (x0, y0, end_x, end_y)
        x1, y1, x2, y2 = coords
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])
        bbox = self._to_base_bbox((left, top, right, bottom))
        self.canvas.delete(self._active_temp_rect)
        self._active_temp_rect = None
        self._drag_start = None
        self.manual_token_counter += 1
        overlay = self._create_overlay(
            "",
            bbox,
            self._calculate_order_key_for_bbox(bbox),
            None,
            is_manual=True,
            select=True,
        )
        overlay.entry.focus_set()
        self._update_transcription_from_overlays()
        self.status_var.set("Added manual line")

    def _on_canvas_release(self, event) -> None:
        if self.mode_var.get() == "draw" and self._active_temp_rect is not None:
            self._finish_manual_overlay(event.x, event.y)
            return
        if self._pressed_overlay:
            if not (event.state & CONTROL_MASK):
                self._select_single_line(self._pressed_overlay)
            coords = self.canvas.coords(self._pressed_overlay.rect_id)
            if coords:
                if self._drag_overlay_start_bbox and tuple(coords) != tuple(self._drag_overlay_start_bbox):
                    self._push_undo(None)
                self._pressed_overlay.bbox = self._to_base_bbox(tuple(coords))
                self._refresh_overlay_visuals(self._pressed_overlay)
                self._update_transcription_from_overlays()
            self._pressed_overlay = None
        self._drag_start = None
        self._drag_overlay_start_bbox = None

    def _delete_selected(self) -> None:
        if self.selected_rects:
            self._push_undo(None)
        for rect_id in list(self.selected_rects):
            overlay = self.rect_to_overlay.get(rect_id)
            if overlay is None:
                continue
            self._destroy_overlay(overlay)
        self._clear_selection()
        self._update_transcription_from_overlays()

    def _on_overlay_modified(self, _event) -> None:
        self._user_modified_transcription = True
        self._update_transcription_from_overlays()

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------
    def _baseline_to_bbox(self, baseline: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in baseline]
        ys = [p[1] for p in baseline]
        return (min(xs), min(ys), max(xs), max(ys) + 4)

    def _extract_tokens(self, image: Image.Image) -> List[OcrToken]:
        # Default: no segmentation available
        return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _append_log(self, path: Path, text: str, status: str, saved: Optional[Path]) -> None:
        if not self.log_path:
            return
        line = [datetime.now(timezone.utc).isoformat(), path.name, status, saved.name if saved else "", text]
        with self.log_path.open("a", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(line)

    def _update_transcript(self, item: AnnotationItem, text: str) -> None:
        transcripts_dir = getattr(self, "transcripts_dir", None)
        save_transcript(transcripts_dir, item.path, text)

    def _load_page_transcript(self, item: AnnotationItem) -> str:
        try:
            default_dir = Path(__file__).resolve().parent.parent / "data" / "train" / "images"
            search_roots: List[Path] = [default_dir] if default_dir.exists() else []
        except Exception:
            search_roots = []
        return load_transcript(item.path, self.transcripts_dir, search_roots=search_roots)

    def _overlays_to_lines(self) -> List[Line]:
        lines: List[Line] = []
        for index, overlay in enumerate(sorted(self.overlay_items, key=lambda o: o.order_key), start=1):
            left, top, right, bottom = overlay.bbox
            mid_y = (top + bottom) // 2
            baseline = [(left, mid_y), (right, mid_y)]
            lines.append(
                Line(
                    id=index,
                    baseline=baseline,
                    bbox=overlay.bbox,
                    text=overlay.entry.get(),
                    order_key=overlay.order_key,
                    selected=overlay.selected,
                    is_manual=overlay.is_manual,
                )
            )
        return lines

    def _save_annotation(self, item: AnnotationItem, text: str) -> Path:
        self.train_dir.mkdir(parents=True, exist_ok=True)
        lines = self._overlays_to_lines()
        if self.options.export_format == "pagexml":
            saved = self.train_dir / f"{item.path.stem}.xml"
            save_pagexml(item.path, lines, saved)
        else:
            save_line_crops(item.path, lines, self.train_dir)
            saved = self.train_dir / f"{item.path.stem}.png"
        return saved


class _AnnotationXContext:
    """Bridge ``XApplier`` operations to the annotation canvas state."""

    def __init__(
        self,
        app: "AnnotationApp",
        layout_map: Dict[str, OverlayItem],
        layout_metadata: Dict[str, "LayoutSpan"],
    ) -> None:
        self.app = app
        self.layout_map = dict(layout_map)
        self.layout_metadata = layout_metadata
        self._snapshot_taken = False

    # XContext API -----------------------------------------------------
    def ensure_snapshot(self) -> None:
        if not self._snapshot_taken:
            self.app._push_undo(None)
            self._snapshot_taken = True

    def has_layout(self, layout_id: str) -> bool:
        return layout_id in self.layout_map

    def create_layout(
        self,
        layout_id: str,
        *,
        text: str,
        bbox: Optional[Tuple[int, int, int, int]],
        after_layout_id: Optional[str],
        before_layout_id: Optional[str],
    ) -> None:
        metadata = self.layout_metadata.get(layout_id)
        target_bbox = bbox or (metadata.bbox if metadata else None)
        if target_bbox is None:
            target_bbox = self.app._estimate_bbox_for_new_line(len(self.app.overlay_items))
        order_key = metadata.order_key if metadata else self.app._calculate_order_key_for_bbox(target_bbox)
        is_manual = metadata.is_manual if metadata else True
        overlay = self.app._create_overlay(
            text,
            target_bbox,
            order_key,
            None,
            is_manual=is_manual,
            select=False,
        )
        self.layout_map[layout_id] = overlay
        self._reposition_overlay(overlay, after_layout_id, before_layout_id)

    def set_layout_text(self, layout_id: str, text: str) -> None:
        overlay = self._require_overlay(layout_id)
        entry = overlay.entry
        deleter = getattr(entry, "delete", None)
        inserter = getattr(entry, "insert", None)
        if callable(deleter):
            try:
                deleter(0, tk.END)
            except Exception:
                pass
        if callable(inserter):
            try:
                inserter(0, text)
                return
            except Exception:
                pass
        setattr(entry, "value", text)

    def set_layout_bbox(self, layout_id: str, bbox: Tuple[int, int, int, int]) -> None:
        overlay = self._require_overlay(layout_id)
        overlay.bbox = bbox
        self.app._refresh_overlay_visuals(overlay)

    def get_layout_bbox(self, layout_id: str) -> Tuple[int, int, int, int]:
        overlay = self._require_overlay(layout_id)
        return overlay.bbox

    def remove_layout(self, layout_id: str) -> None:
        overlay = self.layout_map.pop(layout_id, None)
        if overlay is None:
            return
        self.app._destroy_overlay(overlay)

    # Internal helpers -------------------------------------------------
    def _require_overlay(self, layout_id: str) -> OverlayItem:
        overlay = self.layout_map.get(layout_id)
        if overlay is None:
            raise KeyError(f"Missing layout {layout_id}")
        return overlay

    def _reposition_overlay(
        self,
        overlay: OverlayItem,
        after_layout_id: Optional[str],
        before_layout_id: Optional[str],
    ) -> None:
        overlays = self.app.overlay_items
        try:
            overlays.remove(overlay)
        except ValueError:
            pass
        index = len(overlays)
        if after_layout_id and after_layout_id in self.layout_map:
            anchor = self.layout_map.get(after_layout_id)
            if anchor in overlays:
                index = overlays.index(anchor) + 1
        elif before_layout_id and before_layout_id in self.layout_map:
            anchor = self.layout_map.get(before_layout_id)
            if anchor in overlays:
                index = overlays.index(anchor)
        overlays.insert(index, overlay)
        self.app.overlay_entries = [ov.entry for ov in overlays]


@dataclass
class AnnotationAutoTrainConfig:
    auto_train: int
    output_model: str
    model_dir: Path
    base_lang: str
    max_iterations: int
    tessdata_dir: Optional[Path] = None
    use_gpt_ocr: bool = True
    gpt_model: str = "gpt-4o-mini"
    gpt_prompt: Optional[str] = None
    gpt_cache_dir: Optional[Path] = None
    gpt_max_output_tokens: int = 256
    gpt_max_images: Optional[int] = None
    resume: bool = True
    deserialize_check_limit: Optional[int] = None
    unicharset_size_override: Optional[int] = None


def annotate_images(
    items: Iterable[Path],
    train_dir: Path,
    *,
    options: Optional[AnnotationOptions] = None,
    log_path: Optional[Path] = None,
    auto_train_config: Optional[AnnotationAutoTrainConfig] = None,
    transcripts_dir: Optional[Path] = None,
) -> None:
    """Launch the annotation UI for the given image paths."""
    opts = options or AnnotationOptions()
    annotation_items = [AnnotationItem(Path(p), None) for p in items]
    root = tk.Tk()
    app = AnnotationApp(
        root,
        annotation_items,
        Path(train_dir),
        options=opts,
        log_path=log_path,
        on_sample_saved=None,
        transcripts_dir=transcripts_dir,
    )
    if auto_train_config is not None:
        setattr(app, "auto_train_config", auto_train_config)
        app._on_sample_saved = AnnotationTrainer(root, Path(train_dir), auto_train_config)
    root.mainloop()


__all__ = [
    "AnnotationAutoTrainConfig",
    "AnnotationOptions",
    "annotate_images",
    "AnnotationItem",
    "AnnotationApp",
    "AnnotationTrainer",
    "_prepare_image",
    "_load_annotation_items",
]
