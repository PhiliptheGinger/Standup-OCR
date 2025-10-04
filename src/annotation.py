"""GUI for manually annotating handwriting samples."""
from __future__ import annotations

import csv
import logging
import re
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk
import pytesseract
from pytesseract import Output

try:  # pragma: no cover - package/runtime compatibility
    from .overlay_store import (
        OcrToken,
        Overlay,
        OverlayStore,
    )
except ImportError:  # pragma: no cover - fallback when running as script
    from overlay_store import (  # type: ignore
        OcrToken,
        Overlay,
        OverlayStore,
    )


TokenOrder = Tuple[int, int, int, int, int]
LineKey = Tuple[int, int, int]

CONTROL_MASK = 0x0004
SHIFT_MASK = 0x0001
MIN_DRAW_SIZE = 8


@dataclass(slots=True)
class AnnotationItem:
    """Represent a single image queued for annotation."""

    path: Path
    label: Optional[str] = None
    status: Optional[str] = None
    saved_path: Optional[Path] = None


@dataclass(slots=True)
class OverlayView:
    overlay_id: int
    rect_id: int
    window_id: int
    entry: tk.Entry


@dataclass(slots=True)
class LegacyOverlayItem:
    overlay_id: int
    token: OcrToken
    rect_id: int
    window_id: int
    entry: tk.Entry
    is_manual: bool
    selected: bool


def _prepare_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and return a new image instance."""

    return ImageOps.exif_transpose(image)


def prepare_image(path: Path) -> Image.Image:
    """Open ``path`` and apply EXIF-based orientation for consistent display."""

    with Image.open(path) as src:
        prepared = _prepare_image(src)
        return prepared.copy()


class AnnotationApp:
    """Tkinter-based interface for stepping through a set of images."""

    MAX_SIZE = (900, 700)
    FADE_DELAY_MS = 150
    MIN_ZOOM = 0.5
    MAX_ZOOM = 3.0
    ZOOM_STEP = 1.1

    def __init__(
        self,
        master: tk.Tk,
        items: Iterable[AnnotationItem],
        train_dir: Path,
        log_path: Optional[Path] = None,
        on_sample_saved: Optional[Callable[[Path], None]] = None,
    ) -> None:
        self.master = master
        self.items: List[AnnotationItem] = list(items)
        if not self.items:
            raise ValueError("No images were provided for annotation.")
        self.index = 0
        self.train_dir = Path(train_dir)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._on_sample_saved = on_sample_saved

        self.store = OverlayStore()
        self._store_unsubscribes = [
            self.store.on_overlays(self._on_store_overlays),
            self.store.on_selection(self._on_store_selection),
            self.store.on_focus(self._on_store_focus),
            self.store.on_status(self._on_store_status),
        ]

        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.canvas_image_id: Optional[int] = None
        self.display_scale: Tuple[float, float] = (1.0, 1.0)
        self._base_display_image: Optional[Image.Image] = None
        self._base_scale: Tuple[float, float] = (1.0, 1.0)
        self.zoom_factor: float = 1.0

        self.overlay_views: Dict[int, OverlayView] = {}
        self._overlay_positions: Dict[int, Tuple[float, float, float, float]] = {}
        self.overlay_entries: List[tk.Entry] = []
        self.overlay_items: List[LegacyOverlayItem] = []
        self.rect_to_overlay: Dict[int, LegacyOverlayItem] = {}
        self.selected_rects: Set[int] = set()
        self.current_tokens: List[OcrToken] = []
        self._entry_guard = False

        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_mode: Optional[str] = None
        self._pending_click_id: Optional[int] = None
        self._active_draw_rect: Optional[int] = None
        self._marquee_rect: Optional[int] = None
        self._marquee_additive = False
        self._resize_overlay_id: Optional[int] = None
        self._resize_original_bbox: Optional[Tuple[float, float, float, float]] = None
        self._resize_original_base_bbox: Optional[Tuple[int, int, int, int]] = None
        self._resize_anchor: Optional[Tuple[str, str]] = None
        self._resize_press_point: Optional[Tuple[float, float]] = None
        self._resize_changed = False

        self.filename_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="select")
        self._user_modified_transcription = False
        self._setting_transcription = False

        self._build_ui()
        self._show_current()

    # ------------------------------------------------------------------
    # UI building
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
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_button_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Button-4>", self._on_canvas_mousewheel)  # X11 scroll up
        self.canvas.bind("<Button-5>", self._on_canvas_mousewheel)  # X11 scroll down

        toolbar = tk.Frame(container)
        toolbar.pack(anchor="w", pady=(0, 8))

        select_btn = tk.Radiobutton(toolbar, text="Select", variable=self.mode_var, value="select")
        select_btn.pack(side="left")
        draw_btn = tk.Radiobutton(toolbar, text="Draw", variable=self.mode_var, value="draw")
        draw_btn.pack(side="left", padx=(8, 0))
        zoom_btn = tk.Radiobutton(toolbar, text="Zoom", variable=self.mode_var, value="zoom")
        zoom_btn.pack(side="left", padx=(8, 0))

        delete_btn = tk.Button(toolbar, text="Delete Selected", command=self._delete_selected, state=tk.DISABLED)
        delete_btn.pack(side="left", padx=(16, 0))
        self.delete_button = delete_btn

        entry_frame = tk.Frame(container)
        entry_frame.pack(fill="x", pady=(0, 8))

        tk.Label(entry_frame, text="Transcription:").pack(side="left")
        text_widget = tk.Text(entry_frame, height=4, wrap="word")
        text_widget.pack(side="left", fill="both", expand=True, padx=(8, 0))
        text_widget.bind("<Control-Return>", self._on_confirm)
        text_widget.bind("<Command-Return>", self._on_confirm)
        text_widget.bind("<Key>", self._on_transcription_modified)
        self.entry_widget = text_widget

        buttons = tk.Frame(container)
        buttons.pack(pady=(0, 8))

        back_btn = tk.Button(buttons, text="Back", command=self.back)
        back_btn.pack(side="left", padx=4)
        self.back_button = back_btn
        self.back_button.config(state=tk.DISABLED)

        confirm_btn = tk.Button(buttons, text="Confirm", command=self.confirm, default=tk.ACTIVE)
        confirm_btn.pack(side="left", padx=4)

        skip_btn = tk.Button(buttons, text="Skip", command=self.skip)
        skip_btn.pack(side="left", padx=4)

        unsure_btn = tk.Button(buttons, text="Unsure", command=self.unsure)
        unsure_btn.pack(side="left", padx=4)

        self.status_label = tk.Label(container, textvariable=self.status_var, fg="gray")
        self.status_label.pack(anchor="w")

        self.master.bind("<Escape>", self._on_exit)
        self.master.bind("<Delete>", self._on_delete_selected)
        self.master.bind("<BackSpace>", self._on_delete_selected)
        self.master.bind("<Control-z>", self._on_undo)
        self.master.bind("<Control-Z>", self._on_undo)
        self.master.bind("<Control-y>", self._on_redo)
        self.master.bind("<Control-Y>", self._on_redo)
        self.master.bind("<Control-Shift-Z>", self._on_redo)
        self.master.protocol("WM_DELETE_WINDOW", self._on_exit)

    # ------------------------------------------------------------------
    # Navigation & item display
    # ------------------------------------------------------------------
    def _on_confirm(self, event: Optional[tk.Event]) -> None:
        self.confirm()

    def _on_exit(self, event: Optional[tk.Event] = None) -> None:
        if messagebox.askokcancel("Quit", "Abort annotation and close the window?"):
            self.master.destroy()

    def confirm(self) -> None:
        label = self._get_transcription_text()
        if not label:
            messagebox.showinfo("Missing text", "Enter a transcription or choose Skip/Unsure.")
            return

        item = self.items[self.index]
        try:
            saved_path = self._save_annotation(item.path, label)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save annotation: {exc}")
            return
        item.label = label
        item.status = "confirmed"
        item.saved_path = saved_path
        self._append_log(item.path, label, "confirmed", saved_path)
        self.status_var.set(f"Saved to {saved_path.name}")
        callback = getattr(self, "_on_sample_saved", None)
        if callback is not None and saved_path is not None:
            callback(saved_path)
        self._advance()

    def skip(self) -> None:
        item = self.items[self.index]
        item.label = ""
        item.status = "skipped"
        item.saved_path = None
        self._append_log(item.path, item.label, "skipped", None)
        self.status_var.set("Skipped")
        self._advance()

    def unsure(self) -> None:
        item = self.items[self.index]
        label = self._get_transcription_text()
        item.label = label
        item.status = "unsure"
        item.saved_path = None
        self._append_log(item.path, label, "unsure", None)
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
            messagebox.showinfo("Complete", "All images have been processed.")
            self.master.destroy()
            return
        self._show_current()

    def _show_current(self, *, revisit: bool = False) -> None:
        item = self.items[self.index]
        self.filename_var.set(f"{item.path.name} ({self.index + 1}/{len(self.items)})")
        self._user_modified_transcription = False
        self.back_button.config(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self._display_item(item)
        self.entry_widget.focus_set()
        if revisit:
            previous_status = self.status_var.get() or ""
            reminder = "Returned to previous item; previous response has not been re-recorded."
            if reminder not in previous_status:
                message = f"{previous_status} {reminder}".strip()
                self.status_var.set(message)

    def _display_item(self, item: AnnotationItem) -> None:
        path = item.path
        try:
            with Image.open(path) as image:
                image = _prepare_image(image)
                image = image.convert("RGBA")
                image.thumbnail(self.MAX_SIZE, Image.LANCZOS)
                prepared_image = image.copy()
        except Exception as exc:  # pragma: no cover - GUI feedback only
            messagebox.showerror("Error", f"Could not open {path.name}: {exc}")
            self.skip()
            return

        tokens: List[OcrToken] = []
        prefilled = False
        if item.status == "confirmed" and item.label:
            self._set_transcription(item.label)
            if item.saved_path:
                saved_name = Path(item.saved_path).name or str(item.saved_path)
                self.status_var.set(f"Previously saved to {saved_name}")
            else:
                self.status_var.set("Previously confirmed.")
            prefilled = True
        elif item.status == "unsure":
            self._set_transcription(item.label or "")
            self.status_var.set("Previously marked as unsure.")
            prefilled = True
        elif item.status == "skipped":
            self._set_transcription("")
            self.status_var.set("Previously skipped.")
            prefilled = True
        elif item.label:
            self._set_transcription(item.label)
            self.status_var.set("Previously annotated.")
            prefilled = True

        if not prefilled:
            tokens = self._extract_tokens(prepared_image)
            suggestion = self._compose_text_from_tokens(tokens)
            if suggestion:
                self._set_transcription(suggestion)
                self.status_var.set("Pre-filled transcription using OCR result.")
            else:
                self._set_transcription("")
                filename_hint = self._suggest_label(path)
                if filename_hint:
                    self.status_var.set(
                        "OCR produced no suggestion; using filename hint: "
                        f"{filename_hint}"
                    )
                else:
                    self.status_var.set("OCR produced no suggestion; please transcribe manually.")

        self._display_image(prepared_image, tokens)
        prepared_image.close()

    # ------------------------------------------------------------------
    # Canvas rendering & interactions
    # ------------------------------------------------------------------
    def _display_image(self, image: Image.Image, tokens: Sequence[OcrToken]) -> None:
        if not hasattr(self, "overlay_views"):
            self.overlay_views = {}
        if not hasattr(self, "_overlay_positions"):
            self._overlay_positions = {}
        if not hasattr(self, "overlay_entries"):
            self.overlay_entries = []
        if not hasattr(self, "store"):
            self.store = OverlayStore()
            self._store_unsubscribes = [
                self.store.on_overlays(self._on_store_overlays),
                self.store.on_selection(self._on_store_selection),
                self.store.on_focus(self._on_store_focus),
                self.store.on_status(self._on_store_status),
            ]

        base_width, base_height = image.size
        display_image = image.copy().convert("RGBA")
        display_image.thumbnail(self.MAX_SIZE, Image.LANCZOS)
        self._base_display_image = display_image.copy()
        self._base_scale = (
            display_image.width / base_width if base_width else 1.0,
            display_image.height / base_height if base_height else 1.0,
        )
        self.display_scale = self._base_scale
        self.zoom_factor = 1.0

        photo = ImageTk.PhotoImage(display_image)
        self.current_photo = photo

        self.canvas.delete("all")
        self.overlay_views.clear()
        self._overlay_positions.clear()
        self.overlay_entries.clear()

        self.canvas_image_id = self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, display_image.width, display_image.height))
        if hasattr(self.canvas, "xview_moveto"):
            self.canvas.xview_moveto(0)
        if hasattr(self.canvas, "yview_moveto"):
            self.canvas.yview_moveto(0)

        self.store.set_tokens(tokens)
        self._render_overlays()
        self._update_combined_transcription()

    def _render_overlays(self) -> None:
        overlays = self.store.list_overlays()
        existing_ids = set(self.overlay_views.keys())
        seen_ids: Set[int] = set()
        self.overlay_entries = []

        for overlay in overlays:
            seen_ids.add(overlay.id)
            bbox_display = self._to_display(overlay.bbox_base)
            self._overlay_positions[overlay.id] = bbox_display
            view = self.overlay_views.get(overlay.id)
            if view is None:
                view = self._create_overlay_view(overlay, bbox_display)
                self.overlay_views[overlay.id] = view
            else:
                self._update_overlay_view(view, overlay, bbox_display)
            self._style_overlay_view(view, overlay)
            self.overlay_entries.append(view.entry)

        for overlay_id in existing_ids - seen_ids:
            self._remove_overlay_view(overlay_id)

        self._sync_legacy_structures(overlays)
        self._refresh_delete_button()

    def _sync_legacy_structures(self, overlays: Sequence[Overlay]) -> None:
        existing = {item.overlay_id: item for item in getattr(self, "overlay_items", [])}
        legacy_items: List[LegacyOverlayItem] = []
        rect_map: Dict[int, LegacyOverlayItem] = {}
        selected_rects: Set[int] = set()
        tokens: List[OcrToken] = []
        for overlay in overlays:
            view = self.overlay_views.get(overlay.id)
            if view is None:
                continue
            token = OcrToken(
                text=overlay.text,
                bbox=overlay.bbox_base,
                order_key=overlay.order_key,
                line_key=overlay.line_key,
            )
            legacy = existing.get(overlay.id)
            if legacy is None:
                legacy = LegacyOverlayItem(
                    overlay_id=overlay.id,
                    token=token,
                    rect_id=view.rect_id,
                    window_id=view.window_id,
                    entry=view.entry,
                    is_manual=overlay.is_manual,
                    selected=overlay.selected,
                )
            else:
                legacy.token = token
                legacy.rect_id = view.rect_id
                legacy.window_id = view.window_id
                legacy.entry = view.entry
                legacy.is_manual = overlay.is_manual
                legacy.selected = overlay.selected
            legacy_items.append(legacy)
            rect_map[view.rect_id] = legacy
            if overlay.selected:
                selected_rects.add(view.rect_id)
            tokens.append(token)
        self.overlay_items = legacy_items
        self.rect_to_overlay = rect_map
        self.selected_rects = selected_rects
        self.current_tokens = tokens

    def _create_overlay_view(
        self, overlay: Overlay, bbox_display: Tuple[float, float, float, float]
    ) -> OverlayView:
        left, top, right, bottom = bbox_display
        rect_id = self.canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            outline="#2F80ED",
            width=1,
            tags="overlay",
        )
        entry_width = max(4, int(max(1, abs(right - left)) / 8))
        entry = tk.Entry(self.canvas, width=entry_width)
        entry.insert(0, overlay.text)
        entry._overlay_id = overlay.id  # type: ignore[attr-defined]
        entry.bind("<KeyRelease>", self._on_overlay_entry_changed)
        entry.bind("<FocusIn>", lambda event, oid=overlay.id: self._on_entry_focus(oid))

        desired_top = top - 24
        if desired_top < 0:
            desired_top = top
        window_id = self.canvas.create_window(
            left,
            desired_top,
            anchor="nw",
            window=entry,
            tags="overlay",
        )
        if hasattr(self.canvas, "tag_bind"):
            self.canvas.tag_bind(
                rect_id,
                "<ButtonPress-1>",
                lambda event, oid=overlay.id: self._on_overlay_press(event, oid),
            )
            self.canvas.tag_bind(
                rect_id,
                "<B1-Motion>",
                lambda event, oid=overlay.id: self._on_overlay_drag(event, oid),
            )
            self.canvas.tag_bind(
                rect_id,
                "<ButtonRelease-1>",
                lambda event, oid=overlay.id: self._on_overlay_release(event, oid),
            )
        return OverlayView(overlay.id, rect_id, window_id, entry)

    def _update_overlay_view(
        self,
        view: OverlayView,
        overlay: Overlay,
        bbox_display: Tuple[float, float, float, float],
    ) -> None:
        left, top, right, bottom = bbox_display
        try:
            self.canvas.coords(view.rect_id, left, top, right, bottom)
            desired_top = top - 24 if top - 24 >= 0 else top
            self.canvas.coords(view.window_id, left, desired_top)
            width = max(4, int(max(1, abs(right - left)) / 8))
            if hasattr(view.entry, "configure"):
                view.entry.configure(width=width)
        except tk.TclError:
            pass

        current_text = view.entry.get()
        if current_text != overlay.text:
            self._entry_guard = True
            try:
                view.entry.delete(0, tk.END)
                if overlay.text:
                    view.entry.insert(0, overlay.text)
            finally:
                self._entry_guard = False

    def _style_overlay_view(self, view: OverlayView, overlay: Overlay) -> None:
        outline = "#F2994A" if overlay.selected else "#2F80ED"
        width = 2 if overlay.selected else 1
        try:
            self.canvas.itemconfigure(view.rect_id, outline=outline, width=width)
            self.canvas.itemconfigure(view.window_id, state="normal")
            if hasattr(view.entry, "configure"):
                bg = "#FFFFFF" if overlay.selected else "#F8F9FA"
                fg = "#000000"
                view.entry.configure(state=tk.NORMAL)
                view.entry.configure(bg=bg, fg=fg, insertbackground=fg)
        except tk.TclError:
            pass

    def _remove_overlay_view(self, overlay_id: int) -> None:
        view = self.overlay_views.pop(overlay_id, None)
        if view is None:
            return
        try:
            self.canvas.delete(view.rect_id)
            self.canvas.delete(view.window_id)
        except tk.TclError:
            pass
        try:
            view.entry.destroy()
        except tk.TclError:
            pass
        self._overlay_positions.pop(overlay_id, None)

    def _refresh_delete_button(self) -> None:
        if not hasattr(self, "store") or not hasattr(self, "delete_button"):
            return
        state = tk.NORMAL if self.store.selection else tk.DISABLED
        try:
            self.delete_button.config(state=state)
        except tk.TclError:
            pass

    def _to_display(self, bbox_base: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        sx, sy = self.display_scale
        left, top, right, bottom = bbox_base
        return (left * sx, top * sy, right * sx, bottom * sy)

    def _to_base(self, bbox_display: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        sx, sy = self.display_scale
        if sx == 0 or sy == 0:
            return tuple(int(value) for value in bbox_display)
        left, top, right, bottom = bbox_display
        return (
            int(round(left / sx)),
            int(round(top / sy)),
            int(round(right / sx)),
            int(round(bottom / sy)),
        )

    def _event_has_ctrl(self, event: tk.Event) -> bool:
        return bool(getattr(event, "state", 0) & CONTROL_MASK)

    def _event_has_shift(self, event: tk.Event) -> bool:
        return bool(getattr(event, "state", 0) & SHIFT_MASK)

    def _canvas_coords(self, x: float, y: float) -> Tuple[float, float]:
        if hasattr(self.canvas, "canvasx"):
            x = self.canvas.canvasx(x)
        if hasattr(self.canvas, "canvasy"):
            y = self.canvas.canvasy(y)
        return float(x), float(y)

    def _find_overlay_at_point(self, x: float, y: float) -> Optional[int]:
        for overlay in reversed(self.store.list_overlays()):
            bbox = self._overlay_positions.get(overlay.id)
            if bbox is None:
                continue
            left, top, right, bottom = bbox
            if left <= x <= right and top <= y <= bottom:
                return overlay.id
        return None

    def _on_canvas_button_press(self, event: tk.Event) -> None:
        if self.mode_var.get() == "zoom":
            self._drag_mode = "zoom"
            self._drag_start = (event.x, event.y)
            return

        x, y = self._canvas_coords(event.x, event.y)
        self._drag_start = (x, y)
        self._pending_click_id = None
        self._drag_mode = None

        if self.mode_var.get() == "draw":
            self._drag_mode = "draw"
            self._active_draw_rect = self.canvas.create_rectangle(x, y, x, y, outline="#2F80ED", dash=(2, 2))
            self._active_temp_rect = self._active_draw_rect
            return

        overlay_id = self._find_overlay_at_point(x, y)
        additive = self._event_has_ctrl(event) or self._event_has_shift(event)
        if overlay_id is not None:
            self._pending_click_id = overlay_id
            if not additive:
                self.store.select_click(overlay_id, additive=False)
            else:
                self.store.select_click(overlay_id, additive=True)
            return

        self._drag_mode = "marquee"
        self._marquee_additive = additive
        self._marquee_rect = self.canvas.create_rectangle(
            x,
            y,
            x,
            y,
            outline="#F2994A" if additive else "#2F80ED",
            dash=(4, 2),
        )
        if not additive:
            self.store.clear_selection()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        if self._drag_mode == "zoom":
            return

        x, y = self._canvas_coords(event.x, event.y)

        if self._drag_mode == "draw" and self._active_draw_rect is not None:
            self.canvas.coords(self._active_draw_rect, self._drag_start[0], self._drag_start[1], x, y)
            return

        if self._drag_mode == "marquee" and self._marquee_rect is not None:
            self.canvas.coords(self._marquee_rect, self._drag_start[0], self._drag_start[1], x, y)
            return

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self._drag_mode == "zoom":
            self._drag_start = None
            self._drag_mode = None
            return

        if self._drag_mode == "draw":
            self._finalize_draw()
        elif self._drag_mode == "marquee":
            self._finalize_marquee()
        elif self._pending_click_id is None:
            if not (self._event_has_ctrl(event) or self._event_has_shift(event)):
                self.store.clear_selection()
        self._cleanup_temporary_items()

    def _cleanup_temporary_items(self) -> None:
        active_rect = getattr(self, "_active_draw_rect", None)
        if active_rect is None:
            active_rect = getattr(self, "_active_temp_rect", None)
        if active_rect is not None:
            try:
                self.canvas.delete(active_rect)
            except tk.TclError:
                pass
            self._active_draw_rect = None
            self._active_temp_rect = None
        if self._marquee_rect is not None:
            try:
                self.canvas.delete(self._marquee_rect)
            except tk.TclError:
                pass
            self._marquee_rect = None
        self._drag_start = None
        self._drag_mode = None
        self._pending_click_id = None
        self._marquee_additive = False
        self._reset_resize_state()

    def _finalize_draw(self) -> None:
        if self._active_draw_rect is None:
            return
        coords = self.canvas.coords(self._active_draw_rect)
        if len(coords) != 4:
            return
        left, top, right, bottom = coords
        bbox = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
        if abs(bbox[2] - bbox[0]) < MIN_DRAW_SIZE or abs(bbox[3] - bbox[1]) < MIN_DRAW_SIZE:
            return
        base_bbox = self._to_base(bbox)
        overlay_id = self.store.add_manual(base_bbox)
        self.store.set_status("Added manual overlay")
        self.store.request_focus(overlay_id)

    def _finalize_marquee(self) -> None:
        if self._marquee_rect is None or self._drag_start is None:
            return
        coords = self.canvas.coords(self._marquee_rect)
        if len(coords) != 4:
            return
        left, top, right, bottom = coords
        bbox = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
        base_bbox = self._to_base(bbox)
        ids = self.store.ids_intersecting(base_bbox)
        if self._marquee_additive:
            ids |= set(self.store.selection)
        if ids:
            self.store.select_set(ids)
            self.store.set_status(f"Selected {len(ids)} overlays")

    def _on_canvas_mousewheel(self, event: tk.Event) -> str:
        if self.mode_var.get() != "zoom":
            return ""
        delta = getattr(event, "delta", 0)
        direction = 0
        if delta > 0:
            direction = 1
        elif delta < 0:
            direction = -1
        else:
            num = getattr(event, "num", 0)
            if num == 4:
                direction = 1
            elif num == 5:
                direction = -1
        if direction == 0:
            return ""
        step = self.ZOOM_STEP if direction > 0 else 1 / self.ZOOM_STEP
        focus = self._canvas_coords(event.x, event.y)
        self._apply_zoom(self.zoom_factor * step, focus=focus)
        return "break"

    def _apply_zoom(
        self,
        target: float,
        *,
        focus: Optional[Tuple[float, float]] = None,
    ) -> None:
        if self._base_display_image is None:
            return
        target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, target))
        if abs(target - self.zoom_factor) < 1e-3:
            return
        old_factor = self.zoom_factor
        base_width, base_height = self._base_display_image.size
        new_width = max(1, int(base_width * target))
        new_height = max(1, int(base_height * target))
        resized = self._base_display_image.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        self.current_photo = photo
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(0, 0, image=photo, anchor="nw")
        else:
            try:
                self.canvas.itemconfigure(self.canvas_image_id, image=photo)
            except tk.TclError:
                self.canvas_image_id = self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        self.zoom_factor = target
        base_scale_x, base_scale_y = self._base_scale
        self.display_scale = (base_scale_x * target, base_scale_y * target)
        self._render_overlays()
        if focus is not None:
            self._adjust_canvas_view(focus, old_factor, target, new_width, new_height)

    def _adjust_canvas_view(
        self,
        focus: Tuple[float, float],
        old_factor: float,
        new_factor: float,
        new_width: int,
        new_height: int,
    ) -> None:
        if old_factor <= 0:
            scale_ratio = new_factor
        else:
            scale_ratio = new_factor / old_factor
        new_focus_x = focus[0] * scale_ratio
        new_focus_y = focus[1] * scale_ratio
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        if new_width > canvas_width:
            left = max(0.0, min(new_focus_x - canvas_width / 2, new_width - canvas_width))
            self.canvas.xview_moveto(left / new_width)
        else:
            self.canvas.xview_moveto(0)
        if new_height > canvas_height:
            top = max(0.0, min(new_focus_y - canvas_height / 2, new_height - canvas_height))
            self.canvas.yview_moveto(top / new_height)
        else:
            self.canvas.yview_moveto(0)

    # ------------------------------------------------------------------
    # Overlay entry handling
    # ------------------------------------------------------------------
    def _reset_resize_state(self) -> None:
        self._resize_overlay_id = None
        self._resize_original_bbox = None
        self._resize_original_base_bbox = None
        self._resize_anchor = None
        self._resize_press_point = None
        self._resize_changed = False
        self._drag_mode = None
        self._drag_start = None
        self._pending_click_id = None

    def _determine_resize_anchor(
        self,
        bbox: Tuple[float, float, float, float],
        x: float,
        y: float,
    ) -> Tuple[str, str]:
        left, top, right, bottom = bbox
        threshold = 12.0
        if abs(x - left) <= threshold:
            anchor_x = "left"
        elif abs(x - right) <= threshold:
            anchor_x = "right"
        else:
            anchor_x = "center"

        if abs(y - top) <= threshold:
            anchor_y = "top"
        elif abs(y - bottom) <= threshold:
            anchor_y = "bottom"
        else:
            anchor_y = "center"
        return anchor_x, anchor_y

    def _on_overlay_press(self, event: tk.Event, overlay_id: int) -> str:
        if self.mode_var.get() != "select":
            return ""
        overlay = self.store.get_overlay(overlay_id)
        bbox = self._overlay_positions.get(overlay_id)
        if overlay is None or bbox is None:
            return ""
        x, y = self._canvas_coords(event.x, event.y)
        self._resize_overlay_id = overlay_id
        self._resize_original_bbox = bbox
        self._resize_original_base_bbox = overlay.bbox_base
        self._resize_press_point = (x, y)
        self._resize_anchor = self._determine_resize_anchor(bbox, x, y)
        self._resize_changed = False
        self._drag_mode = "resize"
        self._drag_start = (x, y)
        additive = self._event_has_ctrl(event) or self._event_has_shift(event)
        self.store.select_click(overlay_id, additive=additive)
        self._pending_click_id = overlay_id
        return "break"

    def _on_overlay_drag(self, event: tk.Event, overlay_id: int) -> str:
        if self.mode_var.get() != "select":
            return ""
        if self._resize_overlay_id != overlay_id:
            return ""
        if self._resize_original_bbox is None or self._resize_anchor is None:
            return ""
        x, y = self._canvas_coords(event.x, event.y)
        left, top, right, bottom = self._resize_original_bbox
        anchor_x, anchor_y = self._resize_anchor
        min_size = 4.0

        if anchor_x == "left":
            new_left = min(x, right - min_size)
            new_right = right
        elif anchor_x == "right":
            new_left = left
            new_right = max(x, left + min_size)
        else:
            press_x = self._resize_press_point[0] if self._resize_press_point else left
            dx = x - press_x
            new_left = left + dx
            new_right = right + dx

        if anchor_y == "top":
            new_top = min(y, bottom - min_size)
            new_bottom = bottom
        elif anchor_y == "bottom":
            new_top = top
            new_bottom = max(y, top + min_size)
        else:
            press_y = self._resize_press_point[1] if self._resize_press_point else top
            dy = y - press_y
            new_top = top + dy
            new_bottom = bottom + dy

        if anchor_x == "center":
            width = right - left
            new_right = new_left + width
        if anchor_y == "center":
            height = bottom - top
            new_bottom = new_top + height

        if new_left > new_right:
            new_left, new_right = new_right, new_left
        if new_top > new_bottom:
            new_top, new_bottom = new_bottom, new_top

        base_bbox = self._to_base((new_left, new_top, new_right, new_bottom))
        if (
            self._resize_original_base_bbox is not None
            and base_bbox == self._resize_original_base_bbox
        ):
            return "break"
        overlay = self.store.get_overlay(overlay_id)
        if overlay is not None and overlay.bbox_base == base_bbox:
            return "break"
        self.store.update_bbox(overlay_id, base_bbox)
        self._resize_changed = True
        return "break"

    def _on_overlay_release(self, event: tk.Event, overlay_id: int) -> str:
        if self.mode_var.get() != "select":
            self._reset_resize_state()
            return ""
        if self._resize_overlay_id != overlay_id:
            self._reset_resize_state()
            return ""
        if self._resize_changed and self._resize_original_base_bbox is not None:
            overlay = self.store.get_overlay(overlay_id)
            if overlay is not None:
                self.store.update_bbox(
                    overlay_id,
                    overlay.bbox_base,
                    commit=True,
                    previous_bbox=self._resize_original_base_bbox,
                )
        self._reset_resize_state()
        return "break"

    def _on_rect_click(self, event: tk.Event, overlay_id: int) -> None:
        additive = self._event_has_ctrl(event) or self._event_has_shift(event)
        self.store.select_click(overlay_id, additive=additive)

    def _on_entry_focus(self, overlay_id: int) -> None:
        if overlay_id not in self.store.selection:
            self.store.select_click(overlay_id, additive=False)
        self.store.request_focus(overlay_id)

    def _on_overlay_entry_changed(self, event: tk.Event) -> None:
        if self._entry_guard:
            return
        entry = event.widget
        overlay_id = getattr(entry, "_overlay_id", None)
        if overlay_id is None:
            return
        text = entry.get().strip()
        self.store.update_text(int(overlay_id), text)
        self._user_modified_transcription = False
        self._update_combined_transcription()

    def _focus_overlay(self, overlay_id: Optional[int]) -> None:
        if overlay_id is None:
            return
        view = self.overlay_views.get(overlay_id)
        if view is None:
            return
        try:
            view.entry.focus_set()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Store event handlers
    # ------------------------------------------------------------------
    def _on_store_overlays(self, _value: object) -> None:
        self._render_overlays()
        self._update_combined_transcription()

    def _on_store_selection(self, selection: Tuple[int, ...]) -> None:
        if len(selection) == 1:
            self.store.request_focus(selection[0])
            self.store.set_status("Focused overlay selected")
        elif not selection:
            self.store.set_status(None)
        else:
            self.store.set_status(f"{len(selection)} overlays selected")
        self._refresh_delete_button()

    def _on_store_focus(self, overlay_id: Optional[int]) -> None:
        self._focus_overlay(overlay_id)

    def _on_store_status(self, message: Optional[str]) -> None:
        if hasattr(self, "status_var"):
            self.status_var.set(message or "")

    # ------------------------------------------------------------------
    # Undo / redo / deletion
    # ------------------------------------------------------------------
    def _delete_selected(self) -> None:
        selection = list(self.store.selection)
        if not selection:
            return
        self.store.remove_by_ids(selection)
        self.store.set_status("Deleted selected overlays")

    def _on_delete_selected(self, event: Optional[tk.Event]) -> str:
        self._delete_selected()
        return "break"

    def _on_undo(self, event: Optional[tk.Event]) -> str:
        if self.store.undo():
            return "break"
        return ""

    def _on_overlay_modified(self, _event: Optional[tk.Event]) -> None:
        if hasattr(self, "store"):
            for legacy in getattr(self, "overlay_items", []):
                text = legacy.entry.get().strip()
                overlay = self.store.get_overlay(legacy.overlay_id)
                if overlay is None or overlay.text.strip() == text:
                    continue
                self.store.update_text(legacy.overlay_id, text)
        self._user_modified_transcription = False
        self._update_combined_transcription()

    def _on_redo(self, event: Optional[tk.Event]) -> str:
        if self.store.redo():
            return "break"
        return ""

    # ------------------------------------------------------------------
    # Transcription synchronisation
    # ------------------------------------------------------------------
    def _on_transcription_modified(self, event: tk.Event | None) -> None:
        if not self._setting_transcription:
            self._user_modified_transcription = True
            self._apply_transcription_to_overlays()

    def _apply_transcription_to_overlays(self) -> None:
        if self._setting_transcription:
            return
        text = self.entry_widget.get("1.0", tk.END)
        tokens = re.findall(r"\S+", text)
        if not hasattr(self, "store"):
            for index, entry in enumerate(getattr(self, "overlay_entries", [])):
                value = tokens[index] if index < len(tokens) else ""
                try:
                    entry.delete(0, tk.END)
                except Exception:  # pragma: no cover - test doubles
                    entry.delete(0)
                if value:
                    entry.insert(0, value)
            self._user_modified_transcription = False
            return

        overlays = self.store.list_overlays()
        for index, overlay in enumerate(overlays):
            value = tokens[index] if index < len(tokens) else ""
            value = value.strip()
            if overlay.text.strip() == value:
                continue
            self.store.update_text(overlay.id, value)
        self._user_modified_transcription = False
        self._update_combined_transcription()

    def _update_combined_transcription(self) -> None:
        if self._user_modified_transcription:
            return
        text = self.store.compose_text()
        self._set_transcription(text)

    def _set_transcription(self, value: str) -> None:
        self._setting_transcription = True
        self.entry_widget.delete("1.0", tk.END)
        if value:
            self.entry_widget.insert("1.0", value)
        self._setting_transcription = False
        self._user_modified_transcription = False

    def _get_transcription_text(self) -> str:
        return self.entry_widget.get("1.0", tk.END).strip()

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------
    def _extract_tokens(self, image: Image.Image) -> List[OcrToken]:
        ocr_image: Optional[Image.Image] = None
        try:
            ocr_image = image.copy()
            if ocr_image.mode not in {"RGB", "L"}:
                converted = ocr_image.convert("RGB")
                ocr_image.close()
                ocr_image = converted
            data = pytesseract.image_to_data(
                ocr_image,
                config="--psm 6",
                output_type=Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as exc:
            logging.warning("Tesseract not found: %s", exc)
            return []
        except pytesseract.TesseractError as exc:
            logging.warning("Tesseract error: %s", exc)
            return []
        finally:
            if ocr_image is not None:
                ocr_image.close()

        if not data or "text" not in data:
            return []

        tokens: List[OcrToken] = []
        length = len(data.get("text", []))
        for index in range(length):
            text = (data["text"][index] or "").strip()
            if not text:
                continue
            try:
                left = int(data.get("left", [0])[index])
                top = int(data.get("top", [0])[index])
                width = int(data.get("width", [0])[index])
                height = int(data.get("height", [0])[index])
                page = int(data.get("page_num", [1])[index])
                block = int(data.get("block_num", [0])[index])
                paragraph = int(data.get("par_num", [0])[index])
                line = int(data.get("line_num", [0])[index])
                word = int(data.get("word_num", [index + 1])[index]) or (index + 1)
            except (TypeError, ValueError):
                continue

            bbox = (left, top, left + width, top + height)
            order_key: TokenOrder = (page, block, paragraph, line, word)
            line_key: LineKey = (page, block, line)
            tokens.append(OcrToken(text=text, bbox=bbox, order_key=order_key, line_key=line_key))

        tokens.sort(key=lambda token: token.order_key)
        return tokens

    def _compose_text_from_tokens(self, tokens: Sequence[OcrToken]) -> str:
        if not tokens:
            return ""

        lines: Dict[LineKey, List[Tuple[int, str]]] = {}
        for token in tokens:
            lines.setdefault(token.line_key, []).append((token.order_key[-1], token.text))

        composed: List[str] = []
        for line_key in sorted(lines.keys()):
            words = [word for _, word in sorted(lines[line_key], key=lambda item: item[0])]
            composed.append(" ".join(words))
        return "\n".join(composed)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _save_annotation(self, image_path: Path, label: str) -> Path:
        slug = self._slugify(label)
        output_path = self.train_dir / f"{image_path.stem}-{slug}.png"
        with Image.open(image_path) as image:
            image = _prepare_image(image)
            image.save(output_path)
        return output_path

    def _append_log(
        self,
        path: Path,
        label: str,
        status: str,
        saved_path: Optional[Path],
    ) -> None:
        if self.log_path is None:
            return
        is_new = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(["filename", "label", "status", "saved_path"])
            writer.writerow([path.name, label, status, saved_path.name if saved_path else ""])

    def _suggest_label(self, path: Path) -> str:
        cleaned = [c if c.isalnum() else " " for c in path.stem]
        text = re.sub(r"\s+", " ", "".join(cleaned)).strip()
        return text

    def _slugify(self, value: str) -> str:
        cleaned = [c if c.isalnum() else "-" for c in value.strip().lower()]
        slug = "".join(cleaned)
        slug = re.sub("-+", "-", slug).strip("-")
        if len(slug) > 60:
            slug = slug[:60].rstrip("-")
        return slug or "sample"


def _train_model(*args, **kwargs):  # pragma: no cover - deferred import
    if __package__:
        from .training import train_model as _impl
    else:  # pragma: no cover - fallback for test imports
        from training import train_model as _impl
    return _impl(*args, **kwargs)


@dataclass(slots=True)
class AnnotationAutoTrainConfig:
    auto_train: int
    output_model: str
    model_dir: Path
    base_lang: str
    max_iterations: int
    tessdata_dir: Optional[Path] = None


class AnnotationTrainer:
    """Schedule background training runs as annotations are confirmed."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        train_dir: Path,
        config: AnnotationAutoTrainConfig,
    ) -> None:
        if config.auto_train <= 0:
            raise ValueError("auto_train must be a positive integer")
        self.master = master
        self.train_dir = Path(train_dir)
        self.config = config
        self._pending: deque[Path] = deque()
        self._seen_samples: List[Path] = []
        self._running = False

    @property
    def seen_samples(self) -> List[Path]:
        return list(self._seen_samples)

    def __call__(self, sample_path: Path) -> None:
        path = Path(sample_path)
        self._pending.append(path)
        self._seen_samples.append(path)
        self.master.after(0, self._maybe_train)

    def _maybe_train(self) -> None:
        if self._running:
            return
        if len(self._pending) < self.config.auto_train:
            return

        batch = [self._pending.popleft() for _ in range(self.config.auto_train)]
        self._running = True

        def worker() -> None:
            try:
                logging.info(
                    "Auto-training triggered by %d new annotations.",
                    len(batch),
                )
                model_path = _train_model(
                    self.train_dir,
                    self.config.output_model,
                    model_dir=self.config.model_dir,
                    tessdata_dir=self.config.tessdata_dir,
                    base_lang=self.config.base_lang,
                    max_iterations=self.config.max_iterations,
                )
                logging.info("Updated model saved to %s", model_path)
            except Exception:  # pragma: no cover - logging side effects
                logging.exception("Auto-training failed.")
            finally:
                self._running = False
                self.master.after(0, self._maybe_train)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


def annotate_images(
    sources: Iterable[Path],
    train_dir: Path,
    *,
    log_path: Optional[Path] = None,
    auto_train_config: Optional[AnnotationAutoTrainConfig] = None,
) -> None:
    """Launch the annotation UI for the provided image paths."""

    items = [AnnotationItem(Path(path)) for path in sources]
    if not items:
        raise ValueError("No images found to annotate.")

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Tkinter could not be initialised. Ensure a display is available or "
            "use a system package such as python3-tk."
        ) from exc

    callback = None
    if auto_train_config is not None:
        callback = AnnotationTrainer(
            root,
            train_dir=train_dir,
            config=auto_train_config,
        )

    app = AnnotationApp(root, items, train_dir, log_path, on_sample_saved=callback)
    root.mainloop()
