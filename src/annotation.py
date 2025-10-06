"""GUI for baseline-based handwriting annotation."""
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

from .exporters import save_line_crops, save_pagexml
from .kraken_adapter import is_available as kraken_available, segment_lines
from .line_store import AddLine, LineStore, Point, RemoveLines, SetSelection, UpdateText

CONTROL_MASK = 0x0004
SHIFT_MASK = 0x0001


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
class LineView:
    line_id: int
    canvas_item: Optional[int]
    text_var: tk.StringVar
    entry: tk.Entry


def _prepare_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image)


def prepare_image(path: Path) -> Image.Image:
    with Image.open(path) as src:
        prepared = _prepare_image(src)
        return prepared.copy()


def to_display_point(point: Point, scale: Tuple[float, float]) -> Tuple[float, float]:
    sx, sy = scale
    return point[0] * sx, point[1] * sy


def to_base_point(point: Tuple[float, float], scale: Tuple[float, float]) -> Point:
    sx, sy = scale
    if sx == 0 or sy == 0:
        return point[0], point[1]
    return point[0] / sx, point[1] / sy


def to_base_bbox(bbox: Tuple[float, float, float, float], scale: Tuple[float, float]) -> Tuple[int, int, int, int]:
    x1, y1 = to_base_point((bbox[0], bbox[1]), scale)
    x2, y2 = to_base_point((bbox[2], bbox[3]), scale)
    return int(x1), int(y1), int(x2), int(y2)


class AnnotationApp:
    """Tkinter-based interface for baseline annotation."""

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

        self.store = LineStore()
        self.display_scale: Tuple[float, float] = (1.0, 1.0)
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.canvas_image_id: Optional[int] = None
        self._base_image: Optional[Image.Image] = None

        self.mode_var = tk.StringVar(value="select")
        self.status_var = tk.StringVar()
        self.filename_var = tk.StringVar()
        self._user_modified_transcription = False
        self._setting_transcription = False

        self.canvas: tk.Canvas
        self.entry_widget: tk.Text
        self.delete_button: tk.Button
        self.back_button: tk.Button
        self.lines_frame: tk.Frame

        self.line_views: Dict[int, LineView] = {}
        self._drawing_points: List[Point] = []
        self._drawing_canvas_item: Optional[int] = None

        self._build_ui()
        self._show_current()

    # ------------------------------------------------------------------
    # UI construction
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
        confirm_btn = tk.Button(buttons, text="Confirm", command=self.confirm, default=tk.ACTIVE)
        confirm_btn.pack(side="left", padx=4)
        tk.Button(buttons, text="Skip", command=self.skip).pack(side="left", padx=4)
        tk.Button(buttons, text="Unsure", command=self.unsure).pack(side="left", padx=4)

        status_label = tk.Label(container, textvariable=self.status_var, fg="gray")
        status_label.pack(anchor="w")

        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self.master.bind("<Escape>", self._on_escape)
        self.master.bind("<Delete>", self._on_delete_selected)
        self.master.bind("<BackSpace>", self._on_delete_selected)
        self.master.bind("<Control-z>", self._on_undo)
        self.master.bind("<Control-Z>", self._on_undo)
        self.master.bind("<Control-y>", self._on_redo)
        self.master.bind("<Control-Y>", self._on_redo)
        self.master.bind("<Control-Shift-Z>", self._on_redo)
        self.master.bind("<Alt-Left>", lambda event: self.back())
        self.master.bind("<Return>", self._on_return)

        self.master.protocol("WM_DELETE_WINDOW", self._on_exit)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _on_confirm(self, event: Optional[tk.Event]) -> None:
        self.confirm()

    def _on_escape(self, event: Optional[tk.Event]) -> None:
        if self._drawing_points:
            self._cancel_drawing()
        else:
            self._on_exit()

    def _on_exit(self, event: Optional[tk.Event] = None) -> None:
        if messagebox.askokcancel("Quit", "Abort annotation and close the window?"):
            self.master.destroy()

    def confirm(self) -> None:
        if not self.store.list():
            messagebox.showinfo("No lines", "Create at least one line before confirming.")
            return
        text = self.store.compose_text().strip()
        if not text:
            if not messagebox.askyesno(
                "Empty transcription",
                "No text was entered for the selected lines. Export anyway?",
            ):
                return

        item = self.items[self.index]
        try:
            saved_path = self._export_lines(item)
        except Exception as exc:
            logging.exception("Export failed")
            messagebox.showerror("Export failed", f"Could not export training data: {exc}")
            return

        item.label = text
        item.status = "confirmed"
        item.saved_path = saved_path
        self._append_log(item.path, text, "confirmed", saved_path)
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
        item.label = self.store.compose_text()
        item.saved_path = None
        self._append_log(item.path, item.label or "", "unsure", None)
        self.status_var.set("Marked as unsure")
        self._advance()

    def back(self) -> None:
        if self.index == 0:
            self.status_var.set("Already at the first item")
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
        self.back_button.config(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self._user_modified_transcription = False
        self._load_item(item)
        self.entry_widget.focus_set()
        if revisit:
            self.status_var.set("Returned to previous item.")

    # ------------------------------------------------------------------
    # Item display and export
    # ------------------------------------------------------------------
    def _load_item(self, item: AnnotationItem) -> None:
        self._clear_canvas()
        self.store = LineStore()
        self.line_views.clear()
        try:
            with Image.open(item.path) as image:
                image = _prepare_image(image)
                base_image = image.convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open {item.path.name}: {exc}")
            self.skip()
            return

        self._base_image = base_image
        display = base_image.copy()
        display.thumbnail(self.MAX_SIZE, Image.LANCZOS)
        sx = display.width / base_image.width if base_image.width else 1.0
        sy = display.height / base_image.height if base_image.height else 1.0
        self.display_scale = (sx, sy)

        self.current_photo = ImageTk.PhotoImage(display)
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.current_photo)
        self.canvas.configure(scrollregion=(0, 0, display.width, display.height))

        if item.label:
            self.entry_widget.delete("1.0", tk.END)
            self.entry_widget.insert("1.0", item.label)
        else:
            self.entry_widget.delete("1.0", tk.END)

        if self.options.segmentation == "auto" and self.options.engine == "kraken":
            self._auto_segment(item.path)
        else:
            self.status_var.set("Manual annotation mode")

        self._refresh_lines()
        self._update_transcription()

    def _auto_segment(self, path: Path) -> None:
        if not kraken_available():
            self.status_var.set("Kraken not installed; manual segmentation required (pip install kraken[serve]).")
            return
        try:
            baselines = segment_lines(path)
        except RuntimeError as exc:
            self.status_var.set(str(exc))
            return

        for baseline in baselines:
            if len(baseline) < 2:
                continue
            self.store.add_line(baseline, is_manual=False)
        if baselines:
            self.status_var.set(f"Loaded {len(baselines)} baselines from Kraken")
        else:
            self.status_var.set("Kraken returned no lines; draw them manually")

    def _export_lines(self, item: AnnotationItem) -> Optional[Path]:
        lines = self.store.list()
        if self.options.export_format == "pagexml":
            out_dir = self.train_dir / "pagexml"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{item.path.stem}.xml"
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

    def _clear_canvas(self) -> None:
        self.canvas.delete("all")
        for view in self.line_views.values():
            view.entry.destroy()
        self.line_views.clear()

    # ------------------------------------------------------------------
    # Canvas interaction
    # ------------------------------------------------------------------
    def _on_canvas_press(self, event: tk.Event) -> None:
        if self.mode_var.get() == "draw":
            self._start_drawing(event)
            return
        base_point = to_base_point((event.x, event.y), self.display_scale)
        line_id = self.store.hit_test(base_point[0], base_point[1], tol=5.0)
        if line_id is not None:
            additive = bool(event.state & (CONTROL_MASK | SHIFT_MASK))
            self.store.do(SetSelection({line_id}, additive=additive))
            self._refresh_lines()
            self._focus_selected()
        elif event.state & (CONTROL_MASK | SHIFT_MASK):
            self._start_marquee(event)
        else:
            self.store.select_only(set())
            self._refresh_lines()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._drawing_points:
            self._update_drawing(event)
        elif hasattr(self, "_marquee_start"):
            self._update_marquee(event)

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self._drawing_points:
            return
        if hasattr(self, "_marquee_start"):
            self._finish_marquee(event)

    def _start_drawing(self, event: tk.Event) -> None:
        point = to_base_point((event.x, event.y), self.display_scale)
        self._drawing_points = [point]
        display_point = to_display_point(point, self.display_scale)
        self._drawing_canvas_item = self.canvas.create_line(*display_point, *display_point, fill="orange", width=2)
        self.status_var.set("Drawing baseline: press Enter to finish, Esc to cancel")

    def _update_drawing(self, event: tk.Event) -> None:
        point = to_base_point((event.x, event.y), self.display_scale)
        if self._drawing_points and self._drawing_points[-1] == point:
            return
        self._drawing_points.append(point)
        display_points = []
        for pt in self._drawing_points:
            display_points.extend(to_display_point(pt, self.display_scale))
        if self._drawing_canvas_item is not None:
            self.canvas.coords(self._drawing_canvas_item, *display_points)

    def _finish_drawing(self) -> None:
        if len(self._drawing_points) < 2:
            self._cancel_drawing()
            return
        baseline = list(self._drawing_points)
        self.store.do(AddLine(baseline))
        self._cancel_drawing()
        self._refresh_lines()
        self.status_var.set("Added manual line")

    def _cancel_drawing(self) -> None:
        if self._drawing_canvas_item is not None:
            self.canvas.delete(self._drawing_canvas_item)
        self._drawing_canvas_item = None
        self._drawing_points = []
        self.status_var.set("Drawing cancelled")

    def _on_return(self, event: tk.Event) -> None:
        if self._drawing_points:
            self._finish_drawing()

    def _start_marquee(self, event: tk.Event) -> None:
        self._marquee_start = (event.x, event.y)
        self._marquee_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="blue", dash=(2, 2))

    def _update_marquee(self, event: tk.Event) -> None:
        if not hasattr(self, "_marquee_start"):
            return
        x0, y0 = self._marquee_start
        if self._marquee_rect is not None:
            self.canvas.coords(self._marquee_rect, x0, y0, event.x, event.y)

    def _finish_marquee(self, event: tk.Event) -> None:
        x0, y0 = self._marquee_start
        x1, y1 = event.x, event.y
        if self._marquee_rect is not None:
            self.canvas.delete(self._marquee_rect)
        del self._marquee_start
        self._marquee_rect = None
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return
        base_bbox = to_base_bbox((x0, y0, x1, y1), self.display_scale)
        hits = self.store.bbox_intersect(base_bbox)
        self.store.do(SetSelection(hits, additive=True))
        self._refresh_lines()
        self.status_var.set(f"Selected {len(hits)} lines")

    # ------------------------------------------------------------------
    # Line rendering
    # ------------------------------------------------------------------
    def _refresh_lines(self) -> None:
        # Remove previous line drawings but keep background image
        for view in list(self.line_views.values()):
            if view.canvas_item is not None:
                self.canvas.delete(view.canvas_item)
            view.entry.destroy()
        self.line_views.clear()

        for line in self.store.list():
            display_points: List[float] = []
            for point in line.baseline:
                display_points.extend(to_display_point(point, self.display_scale))
            colour = "#2563eb" if line.selected else "#f97316"
            canvas_item = self.canvas.create_line(*display_points, fill=colour, width=2)
            text_var = tk.StringVar(value=line.text)
            entry = tk.Entry(self.lines_frame, textvariable=text_var, width=80)
            entry.pack(fill="x", pady=2)
            entry.bind("<FocusOut>", lambda event, line_id=line.id, var=text_var: self._commit_entry(line_id, var))
            entry.bind("<Return>", lambda event, line_id=line.id, var=text_var: self._commit_entry(line_id, var))
            entry.bind("<Key>", lambda event: self._on_entry_key())
            if line.selected:
                entry.configure(background="#dbeafe")
            self.line_views[line.id] = LineView(line.id, canvas_item, text_var, entry)

        self.delete_button.config(state=tk.NORMAL if self.store.selection() else tk.DISABLED)
        self._update_transcription()

    def _commit_entry(self, line_id: int, var: tk.StringVar) -> None:
        try:
            text = var.get()
        except tk.TclError:
            return
        line = next((line for line in self.store.list() if line.id == line_id), None)
        if line is None or line.text == text:
            return
        self.store.do(UpdateText(line_id, text))
        self._refresh_lines()
        self.status_var.set("Updated text")

    def _on_entry_key(self) -> None:
        self._user_modified_transcription = True

    def _focus_selected(self) -> None:
        selected = self.store.selection()
        if len(selected) == 1:
            line_id = next(iter(selected))
            view = self.line_views.get(line_id)
            if view is not None:
                view.entry.focus_set()

    # ------------------------------------------------------------------
    # Editing helpers
    # ------------------------------------------------------------------
    def _delete_selected(self) -> None:
        selection = self.store.selection()
        if not selection:
            return
        self.store.do(RemoveLines(selection))
        self._refresh_lines()
        self.status_var.set("Removed selected lines")

    def _on_delete_selected(self, event: Optional[tk.Event]) -> None:
        self._delete_selected()

    def _on_undo(self, event: Optional[tk.Event]) -> None:
        if self.store.undo():
            self._refresh_lines()
            self.status_var.set("Undo")

    def _on_redo(self, event: Optional[tk.Event]) -> None:
        if self.store.redo():
            self._refresh_lines()
            self.status_var.set("Redo")

    def _on_transcription_modified(self, event: tk.Event) -> None:
        if event.keysym in {"Shift_L", "Shift_R", "Control_L", "Control_R"}:
            return
        self._user_modified_transcription = True

    def _update_transcription(self) -> None:
        if self._user_modified_transcription:
            return
        text = self.store.compose_text()
        self._setting_transcription = True
        self.entry_widget.delete("1.0", tk.END)
        self.entry_widget.insert("1.0", text)
        self._setting_transcription = False

def _train_model(*args, **kwargs):  # pragma: no cover
    if __package__:
        from .training import train_model as _impl
    else:
        from training import train_model as _impl  # type: ignore
    return _impl(*args, **kwargs)


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

    def __call__(self, sample_path: Path) -> None:
        path = Path(sample_path)
        self._pending = list(self._pending) + [path]
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
