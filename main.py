"""Command-line interface for the handwriting OCR toolkit."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from src.annotation import AnnotationAutoTrainConfig, AnnotationOptions, annotate_images
from src.ocr import ocr_image
from src.gpt_ocr import GPTTranscriber, GPTTranscriptionError
from src.review import ReviewAborted, ReviewConfig, ReviewSession
from src.training import SUPPORTED_EXTENSIONS, train_model
from src.refine import DEFAULT_REFINE_PROMPT, run_refinement
from src.kraken_adapter import (
    DeskewConfig,
    is_available as kraken_available,
    ocr as kraken_ocr,
    segment_pages_with_kraken,
    train as kraken_train,
)
from src.kraken_dataset import sanitize_line_dataset
from src.foreground_filter import (
    DEFAULT_GPT_FILTER_MODEL,
    DEFAULT_GPT_FILTER_PROMPT,
    ForegroundFilterConfig,
)
from src.xnet import (
    XNetController,
    XNetConfig,
    KrakenSegmenter,
    KrakenRecognizer,
    TesseractRecognizer,
)


DEFAULT_TRAIN_DIR = Path("train")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_RESULTS_FILE = Path("results.csv")
DEFAULT_TRANSCRIPTS_DIR = Path("transcripts") / "raw"
DEFAULT_REFINED_DIR = Path("transcripts") / "refined"


def notify_end(title: str, message: str) -> None:
    """Best-effort end-of-run notification.

    On Windows, attempts a toast notification if the optional `win10toast`
    package is available; otherwise falls back to an audible beep.
    Always emits a terminal bell as a fallback.
    """

    if sys.platform.startswith("win"):
        try:
            from win10toast import ToastNotifier  # type: ignore

            ToastNotifier().show_toast(title, message, duration=6, threaded=True)
            return
        except Exception:
            pass

        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

    try:
        # Terminal bell fallback.
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass

    logging.info("%s - %s", title, message)


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    # PIL can emit extremely verbose EXIF/TIFF debug logs when the root logger
    # is configured for DEBUG elsewhere. Keep it quiet by default.
    if not verbose:
        logging.getLogger("PIL").setLevel(logging.WARNING)


def iter_images(folder: Path) -> Iterable[Path]:
    """Yield image files from ``folder`` that match supported extensions."""
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def add_gpt_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach common ChatGPT OCR arguments to a subparser."""

    parser.add_argument(
        "--no-gpt-ocr",
        action="store_true",
        help=(
            "Disable ChatGPT-based transcription when preparing training data "
            "and fall back to file-name derived labels."
        ),
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-4o-mini",
        help="ChatGPT model identifier to use for OCR (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--gpt-prompt",
        help="Custom prompt sent alongside each image when requesting ChatGPT OCR.",
    )
    parser.add_argument(
        "--gpt-cache-dir",
        type=Path,
        help="Optional directory used to cache ChatGPT OCR responses.",
    )
    parser.add_argument(
        "--gpt-max-output-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens ChatGPT may return per transcription (default: 256).",
    )
    parser.add_argument(
        "--gpt-max-images",
        type=int,
        help=(
            "Upper bound on how many images should be sent to ChatGPT for transcription. "
            "Remaining samples fall back to filename labels."
        ),
    )


def handle_train(args: argparse.Namespace) -> None:
    if args.engine == "kraken":
        if not kraken_available():
            raise RuntimeError(
                "Kraken is not installed. Install it with 'pip install kraken[serve]' to train Kraken models."
            )
        model_out = args.model if args.model else args.model_dir / "kraken.mlmodel"
        base_model = args.base_model
        progress = None
        if args.kraken_progress == "plain":
            progress = "plain"
        elif args.kraken_progress == "rich":
            progress = "rich"
        elif args.kraken_progress == "none":
            progress = "none"
        model_path = kraken_train(
            args.train_dir,
            Path(model_out),
            epochs=args.epochs,
            val_split=args.val_split,
            base_model=base_model,
            progress=progress,
        )
        logging.info("Kraken model saved to %s", model_path)
        return

    model_path = train_model(
        args.train_dir,
        args.output_model,
        model_dir=args.model_dir,
        tessdata_dir=args.tessdata_dir,
        base_lang=args.base_lang,
        max_iterations=args.max_iterations,
        unicharset_size_override=args.unicharset_size,
        deserialize_check_limit=args.deserialize_check_limit,
        use_gpt_ocr=not args.no_gpt_ocr,
        gpt_model=args.gpt_model,
        gpt_prompt=args.gpt_prompt,
        gpt_cache_dir=args.gpt_cache_dir,
        gpt_max_output_tokens=args.gpt_max_output_tokens,
        gpt_max_images=args.gpt_max_images,
        resume=not args.no_resume,
    )
    logging.info("Model saved to %s", model_path)


def handle_kraken_lines(args: argparse.Namespace) -> None:
    resize_width = args.resize_width if args.resize_width and args.resize_width > 0 else None
    stats = sanitize_line_dataset(
        args.source,
        args.output,
        adaptive=not args.no_adaptive,
        force_landscape=not args.no_force_landscape,
        resize_width=resize_width,
    )
    logging.info(
        "Cleaned %d line crops (skipped %d missing GT, %d empty GT)",
        stats.processed,
        stats.missing_gt,
        stats.empty_gt,
    )


def _build_kraken_segment_cli_args(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    global_args: list[str] = []
    segment_args: list[str] = []

    if args.segment_device:
        global_args.extend(["--device", args.segment_device])
    if args.segment_threads:
        global_args.extend(["--threads", str(args.segment_threads)])
    if args.segment_autocast:
        global_args.append("--autocast")
    if args.segment_no_legacy_polygons:
        global_args.append("--no-legacy-polygons")

    if args.segment_subline:
        global_args.append("--subline-segmentation")
    elif args.segment_no_subline:
        global_args.append("--no-subline-segmentation")

    if args.segment_global_arg:
        global_args.extend(args.segment_global_arg)

    if args.segment_strategy == "baseline":
        segment_args.append("--baseline")
    elif args.segment_strategy == "boxes":
        segment_args.append("--boxes")

    if args.segment_direction:
        segment_args.extend(["--text-direction", args.segment_direction])
    if args.segment_scale is not None:
        segment_args.extend(["--scale", str(args.segment_scale)])
    if args.segment_maxcolseps is not None:
        segment_args.extend(["--maxcolseps", str(args.segment_maxcolseps)])
    if args.segment_black_colseps:
        segment_args.append("--black-colseps")
    if args.segment_keep_hlines:
        segment_args.append("--hlines")
    if args.segment_pad:
        left, right = args.segment_pad
        segment_args.extend(["--pad", str(left), str(right)])
    if args.segment_mask:
        segment_args.extend(["--mask", str(args.segment_mask)])

    if args.segment_cli_arg:
        segment_args.extend(args.segment_cli_arg)

    return global_args, segment_args


def handle_kraken_segment(args: argparse.Namespace) -> None:
    if not kraken_available():
        raise RuntimeError(
            "Kraken is not installed. Install it with 'pip install kraken[serve]' to run kraken-segment."
        )

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.is_dir():
        images = list(iter_images(source))
    else:
        images = [source]

    if not images:
        logging.warning("No images found in %s", source)
        return

    output_dir = Path(args.out_lines) if getattr(args, "out_lines", None) else Path(args.output)
    pagexml_dir = Path(args.pagexml) if args.pagexml is not None else None

    if output_dir.exists():
        try:
            has_content = any(output_dir.iterdir())
        except OSError:
            has_content = True

        if has_content and not args.overwrite:
            raise RuntimeError(
                f"Output directory {output_dir} already exists. Re-run with --overwrite to replace its contents."
            )
        if args.overwrite:
            shutil.rmtree(output_dir, ignore_errors=True)

    global_cli_args, segment_cli_args = _build_kraken_segment_cli_args(args)

    filter_config: ForegroundFilterConfig | None = None
    if args.filter_non_writing:
        filter_config = ForegroundFilterConfig(
            ink_ratio_threshold=args.filter_ink_threshold,
            edge_density_threshold=args.filter_edge_threshold,
            contrast_threshold=args.filter_contrast_threshold,
        )

    deskew_config = None
    if not args.no_deskew:
        deskew_config = DeskewConfig(
            enabled=True,
            max_skew=args.deskew_max_angle,
            force_landscape=not args.deskew_allow_portrait,
            force_upright=not args.deskew_allow_upside_down,
        )

    stats = segment_pages_with_kraken(
        images,
        output_dir,
        model=str(args.model) if args.model else None,
        pagexml_dir=pagexml_dir,
        padding=args.padding,
        min_width=args.min_width,
        min_height=args.min_height,
        global_cli_args=global_cli_args or None,
        segment_cli_args=segment_cli_args or None,
        filter_config=filter_config,
        filter_use_gpt=args.filter_use_gpt,
        filter_gpt_model=args.filter_gpt_model,
        filter_gpt_prompt=args.filter_gpt_prompt,
        filter_gpt_cache_dir=args.filter_gpt_cache_dir,
        deskew_config=deskew_config,
    )

    logging.info(
        "Segmented %d page(s) into %d line crops (%d skipped, %d errors). Output: %s",
        stats.pages,
        stats.lines,
        stats.skipped,
        stats.errors,
        output_dir,
    )


def handle_test(args: argparse.Namespace) -> None:
    text = ocr_image(
        args.image,
        model_path=args.model,
        tessdata_dir=args.tessdata_dir,
        psm=args.psm,
    )
    print(text)


def handle_xnet(args: argparse.Namespace) -> None:
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    started = time.time()
    processed = 0
    accepted = 0
    success = False

    transcripts_dir = Path(args.output_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    segmenter = KrakenSegmenter(
        model=str(args.kraken_model) if args.kraken_model else None,
        pagexml_dir=Path(args.pagexml) if args.pagexml else None,
        padding=args.padding,
        min_width=args.min_width,
        min_height=args.min_height,
        deskew=not args.no_deskew,
        deskew_max_angle=args.deskew_max_angle,
        force_landscape=not args.deskew_allow_portrait,
        force_upright=not args.deskew_allow_upside_down,
    )

    if getattr(args, "recognizer", "tesseract") == "kraken":
        if args.kraken_ocr_model is None:
            raise ValueError("--kraken-ocr-model is required when --recognizer kraken")
        recognizer = KrakenRecognizer(model_path=Path(args.kraken_ocr_model))
    else:
        recognizer = TesseractRecognizer(
            model_path=args.tesseract_model,
            tessdata_dir=args.tessdata_dir,
            psm=args.psm,
            use_gpt_fallback=args.gpt_fallback,
            gpt_model=args.gpt_model,
            gpt_cache_dir=args.gpt_cache_dir,
        )

    config = XNetConfig(
        min_confidence=args.min_confidence,
        min_coherence=args.min_coherence,
        min_similarity=args.min_similarity,
        max_retries=args.max_retries,
        auto_tune=not args.no_auto_tune,
        segmenter=segmenter,
        recognizer=recognizer,
        ground_truth_page_dir=args.gt_page_dir,
        write_report=args.write_report,
    )
    controller = XNetController(config)

    images: List[Path]
    if source.is_dir():
        images = [p for p in sorted(source.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    else:
        images = [source]

    if args.max_images is not None and args.max_images > 0:
        images = images[: args.max_images]

    try:
        for image_path in images:
            processed += 1
            logging.info("XNet processing %s", image_path)
            result = controller.run(image_path, output_dir=artifacts_dir / image_path.stem)
            # Compose final page text from ordered lines
            ordered = sorted(result.lines, key=lambda r: r.line.order_key)
            page_text = "\n".join(rl.line.text for rl in ordered).strip()
            if page_text and result.accepted:
                accepted += 1
                out_txt = transcripts_dir / f"{image_path.stem}.txt"
                out_txt.write_text(page_text, encoding="utf8")
                logging.info(
                    "Wrote transcript %s (coherence=%.1f, attempts=%d, sim=%s)",
                    out_txt,
                    result.coherence,
                    result.attempts,
                    f"{result.similarity:.3f}" if result.similarity is not None else "n/a",
                )
            elif page_text and not result.accepted:
                logging.warning(
                    "Transcript rejected for %s (conf=%.1f coh=%.1f sim=%s, attempts=%d); skipping write.",
                    image_path,
                    result.avg_confidence,
                    result.coherence,
                    f"{result.similarity:.3f}" if result.similarity is not None else "n/a",
                    result.attempts,
                )
            else:
                logging.warning(
                    "No transcript content for %s (coherence=%.1f, attempts=%d); skipping write.",
                    image_path,
                    result.coherence,
                    result.attempts,
                )
        success = True
    finally:
        if getattr(args, "notify", False):
            elapsed = time.time() - started
            title = "Standup-OCR XNet finished" if success else "Standup-OCR XNet failed"
            notify_end(
                title,
                f"Processed {processed} page(s); accepted {accepted}. Elapsed {elapsed:.1f}s.",
            )


def handle_batch(args: argparse.Namespace) -> None:
    folder = Path(args.folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows: List[dict[str, str]] = []
    for image_path in iter_images(folder):
        try:
            text = ocr_image(
                image_path,
                model_path=args.model,
                tessdata_dir=args.tessdata_dir,
                psm=args.psm,
            )
        except Exception as exc:  # pragma: no cover - runtime logging only
            logging.exception("Failed to OCR %s", image_path)
            text = f"ERROR: {exc}"
        rows.append({"image": image_path.name, "text": text})

    df = pd.DataFrame(rows)
    output = Path(args.output)
    df.to_csv(output, index=False)
    logging.info("Batch OCR complete. Results saved to %s", output)


def handle_ocr(args: argparse.Namespace) -> None:
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.engine == "kraken":
        if not kraken_available():
            raise RuntimeError(
                "Kraken is not installed. Install it with 'pip install kraken[serve]' to run Kraken OCR."
            )
        if args.model is None:
            raise ValueError("--model is required when using --engine kraken")
        model_path = Path(args.model)
        if input_path.is_dir():
            images = list(iter_images(input_path))
        else:
            images = [input_path]
        if not images:
            logging.warning("No images found in %s", input_path)
        for image_path in images:
            out_txt = output_dir / f"{image_path.stem}.txt"
            kraken_ocr(image_path, model_path, out_txt)
        return

    if input_path.is_dir():
        images = list(iter_images(input_path))
    else:
        images = [input_path]
    if not images:
        logging.warning("No images found in %s", input_path)
    for image_path in images:
        text = ocr_image(
            image_path,
            model_path=args.model,
            tessdata_dir=args.tessdata_dir,
            psm=args.psm,
        )
        out_txt = output_dir / f"{image_path.stem}.txt"
        out_txt.write_text(text, encoding="utf8")


def handle_refine(args: argparse.Namespace) -> None:
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.is_dir():
        images = list(iter_images(source))
        if not images:
            logging.warning("No images found in %s", source)
    else:
        images = [source]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.gpt_prompt or DEFAULT_REFINE_PROMPT
    transcriber = GPTTranscriber(
        model=args.gpt_model,
        prompt=prompt,
        max_output_tokens=args.gpt_max_output_tokens,
        cache_dir=args.gpt_cache_dir,
    )

    results = run_refinement(
        images,
        transcriber=transcriber,
        engine=args.engine,
        tesseract_model=args.tesseract_model,
        tessdata_dir=args.tessdata_dir,
        psm=args.psm,
        kraken_model=args.kraken_model,
        temperature=args.gpt_temperature,
        max_output_tokens=args.gpt_max_output_tokens,
    )

    for result in results:
        payload = {
            "image": result.image.name,
            "image_path": str(result.image),
            "engine": result.engine,
            "rough_text": result.rough_text,
            "corrected_text": result.corrected_text,
            "confidence": result.confidence,
            "notes": result.notes,
            "tokens": result.tokens,
        }
        json_path = output_dir / f"{result.image.stem}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        txt_path = output_dir / f"{result.image.stem}.txt"
        txt_path.write_text(result.corrected_text.strip() + "\n", encoding="utf-8")
        if result.notes:
            logging.info(
                "Refined %s with confidence %.2f (notes: %s)",
                result.image.name,
                result.confidence,
                result.notes,
            )
        else:
            logging.info(
                "Refined %s with confidence %.2f",
                result.image.name,
                result.confidence,
            )


def handle_review(args: argparse.Namespace) -> None:
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if args.auto_train is not None and args.auto_train <= 0:
        raise ValueError("--auto-train must be a positive integer")

    config = ReviewConfig(
        threshold=args.threshold,
        model_path=args.model,
        tessdata_dir=args.tessdata_dir,
        psm=args.psm,
        train_dir=args.train_dir,
        preview=not args.no_preview,
        full_image_gpt=(
            args.full_image_gpt if args.full_image_gpt is not None else True
        ),
    )
    session: ReviewSession
    last_trained_count = 0
    gpt_max_images = args.gpt_max_images
    transcriber: Optional[GPTTranscriber] = None
    prompt_handler = None
    tk_prompt = None

    if gpt_max_images is not None and gpt_max_images < 0:
        raise ValueError("--gpt-max-images must be zero or a positive integer")

    if not args.no_gpt_ocr and (gpt_max_images is None or gpt_max_images != 0):
        transcriber_kwargs: dict[str, object] = {
            "model": args.gpt_model,
            "max_output_tokens": args.gpt_max_output_tokens,
        }
        if args.gpt_prompt is not None:
            transcriber_kwargs["prompt"] = args.gpt_prompt
        if args.gpt_cache_dir is not None:
            transcriber_kwargs["cache_dir"] = Path(args.gpt_cache_dir)
        try:
            transcriber = GPTTranscriber(**transcriber_kwargs)
        except GPTTranscriptionError as exc:
            raise RuntimeError(f"Unable to initialise ChatGPT OCR: {exc}") from exc

    def maybe_train() -> None:
        nonlocal last_trained_count
        if not args.auto_train:
            return
        while session.saved_samples - last_trained_count >= args.auto_train:
            logging.info(
                "Auto-training triggered after %d new samples.",
                session.saved_samples,
            )
            model_path = train_model(
                args.train_dir,
                args.output_model,
                model_dir=args.model_dir,
                tessdata_dir=args.tessdata_dir,
                base_lang=args.base_lang,
                max_iterations=args.max_iterations,
                unicharset_size_override=args.unicharset_size,
                deserialize_check_limit=args.deserialize_check_limit,
                use_gpt_ocr=not args.no_gpt_ocr,
                gpt_model=args.gpt_model,
                gpt_prompt=args.gpt_prompt,
                gpt_cache_dir=args.gpt_cache_dir,
                gpt_max_output_tokens=args.gpt_max_output_tokens,
                gpt_max_images=args.gpt_max_images,
                resume=not args.no_resume,
            )
            logging.info("Updated model saved to %s", model_path)
            last_trained_count += args.auto_train

    if getattr(args, "gui", False):
        from src.review_tk import TkReviewPrompt

        tk_prompt = TkReviewPrompt()
        prompt_handler = tk_prompt
        config.preview = False

    session = ReviewSession(
        config,
        on_sample_saved=lambda *_args: maybe_train(),
        transcriber=transcriber,
        gpt_max_images=gpt_max_images,
        prompt_handler=prompt_handler,
    )

    try:
        paths: List[Path]
        if source.is_dir():
            paths = list(iter_images(source))
            if not paths:
                logging.warning("No images found in %s", source)
        else:
            paths = [source]

        for path in paths:
            session.review_image(path)
            maybe_train()
    except ReviewAborted:
        logging.info("Review aborted by operator.")
        maybe_train()
    finally:
        if tk_prompt is not None:
            tk_prompt.destroy()


def handle_annotate(args: argparse.Namespace) -> None:
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if args.auto_train is not None and args.auto_train <= 0:
        raise ValueError("--auto-train must be a positive integer")

    if source.is_dir():
        paths = list(iter_images(source))
        if not paths:
            raise FileNotFoundError(f"No supported images found in {source}")
    else:
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image type: {source.suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        paths = [source]

    auto_train_config = None
    if args.auto_train:
        auto_train_config = AnnotationAutoTrainConfig(
            auto_train=args.auto_train,
            output_model=args.output_model,
            model_dir=args.model_dir,
            base_lang=args.base_lang,
            max_iterations=args.max_iterations,
            tessdata_dir=args.tessdata_dir,
            use_gpt_ocr=not args.no_gpt_ocr,
            gpt_model=args.gpt_model,
            gpt_prompt=args.gpt_prompt,
            gpt_cache_dir=args.gpt_cache_dir,
            gpt_max_output_tokens=args.gpt_max_output_tokens,
            gpt_max_images=args.gpt_max_images,
            resume=not args.no_resume,
            deserialize_check_limit=args.deserialize_check_limit,
            unicharset_size_override=args.unicharset_size,
        )

    # Validate pagexml_dir if load mode is specified
    pagexml_dir = getattr(args, 'pagexml_dir', None)
    if args.seg == "load":
        if pagexml_dir is None:
            raise ValueError("--seg load requires --pagexml-dir to be specified")
        if not pagexml_dir.exists():
            raise FileNotFoundError(f"PAGE-XML directory not found: {pagexml_dir}")

    if args.seg == "auto" and args.engine == "kraken" and not kraken_available():
        logging.warning(
            "Kraken auto-segmentation requested but Kraken is not installed; falling back to manual mode."
        )

    prefill_psm = args.prefill_psm if args.prefill_psm is not None else 6

    options = AnnotationOptions(
        engine=args.engine,
        segmentation=args.seg,
        export_format=args.export,
        prefill_enabled=not args.no_prefill,
        prefill_model=args.prefill_model,
        prefill_tessdata=args.prefill_tessdata,
        prefill_psm=prefill_psm,
        pagexml_dir=pagexml_dir,
    )

    transcripts_dir = None if args.skip_transcripts else args.transcripts_dir

    annotate_images(
        paths,
        args.train_dir,
        options=options,
        log_path=args.output_log,
        auto_train_config=auto_train_config,
        transcripts_dir=transcripts_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Fine-tune a Tesseract model using samples placed in the train/ folder.",
    )
    train_parser.add_argument(
        "--engine",
        choices=["tesseract", "kraken"],
        default="tesseract",
        help="Training engine to use (default: tesseract).",
    )
    train_parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Directory with training images (default: train/).",
    )
    train_parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Where to store trained models (default: models/).",
    )
    train_parser.add_argument(
        "--output-model",
        default="handwriting",
        help="Base name of the output model (default: handwriting).",
    )
    train_parser.add_argument(
        "--model",
        type=Path,
        help="Output model path when training with Kraken.",
    )
    train_parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Path to tessdata directory containing base traineddata files.",
    )
    train_parser.add_argument(
        "--base-lang",
        default="eng",
        help="Base language code to fine-tune (default: eng).",
    )
    train_parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Training iterations to run (default: 1000).",
    )
    train_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate all .lstmf samples instead of reusing cached ones.",
    )
    train_parser.add_argument(
        "--deserialize-check-limit",
        type=int,
        help=(
            "Maximum number of .lstmf samples to sanity-check with Tesseract before training. "
            "Omit to check all samples."
        ),
    )
    train_parser.add_argument(
        "--unicharset-size",
        type=int,
        help=(
            "Override LSTM unicharset size used for network specification. "
            "If not provided, the size is inferred from the base traineddata."
        ),
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Epochs to train when using Kraken (default: 50).",
    )
    train_parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split for Kraken training (default: 0.1).",
    )
    train_parser.add_argument(
        "--base-model",
        type=Path,
        help="Optional base Kraken model to fine-tune.",
    )
    train_parser.add_argument(
        "--kraken-progress",
        choices=["auto", "plain", "rich", "none"],
        default="auto",
        help=(
            "Progress renderer passed to Kraken training. Use 'plain' to avoid Rich crashes on Windows, "
            "'rich' for the default TTY experience, 'none' to leave the environment untouched, or keep 'auto' "
            "(default) to fall back to a safe choice per platform."
        ),
    )
    add_gpt_arguments(train_parser)
    train_parser.set_defaults(func=handle_train)

    kraken_lines_parser = subparsers.add_parser(
        "kraken-lines",
        help="Regenerate Kraken-ready line crops using the preprocessing pipeline.",
    )
    kraken_lines_parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_TRAIN_DIR / "lines",
        help="Directory containing existing line crops (default: train/lines).",
    )
    kraken_lines_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TRAIN_DIR / "kraken_lines",
        help="Directory where cleaned line crops will be written (default: train/kraken_lines).",
    )
    kraken_lines_parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Optional width passed to the preprocessing pipeline (default: keep original width).",
    )
    kraken_lines_parser.add_argument(
        "--no-force-landscape",
        action="store_true",
        help="Disable automatic rotation of portrait-oriented line crops.",
    )
    kraken_lines_parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive thresholding when cleaning line crops.",
    )
    kraken_lines_parser.set_defaults(func=handle_kraken_lines)

    kraken_segment_parser = subparsers.add_parser(
        "kraken-segment",
        help="Use Kraken to auto-segment full pages into individual line crops.",
    )
    kraken_segment_parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_TRAIN_DIR / "images",
        help="Image file or directory to segment (default: train/images).",
    )
    kraken_segment_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TRAIN_DIR / "kraken_auto_lines",
        help="Directory where cropped line images will be saved (default: train/kraken_auto_lines).",
    )
    kraken_segment_parser.add_argument(
        "--out-lines",
        dest="out_lines",
        type=Path,
        help="Deprecated alias for --output (will be removed in a future release).",
    )
    kraken_segment_parser.add_argument(
        "--pagexml",
        type=Path,
        help="Optional directory for PAGE-XML exports produced by Kraken.",
    )
    kraken_segment_parser.add_argument(
        "--model",
        type=Path,
        help="Optional Kraken segmentation model (.mlmodel) to guide line detection.",
    )
    kraken_segment_parser.add_argument(
        "--padding",
        type=int,
        default=12,
        help="Pixel padding applied around each detected line before cropping (default: 12).",
    )
    kraken_segment_parser.add_argument(
        "--min-width",
        type=int,
        default=24,
        help="Skip cropped lines narrower than this many pixels (default: 24).",
    )
    kraken_segment_parser.add_argument(
        "--min-height",
        type=int,
        default=16,
        help="Skip cropped lines shorter than this many pixels (default: 16).",
    )
    kraken_segment_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing new crops.",
    )
    kraken_segment_parser.add_argument(
        "--segment-device",
        help="Device string passed to kraken (for example cpu or cuda:0).",
    )
    kraken_segment_parser.add_argument(
        "--segment-threads",
        type=int,
        help="Thread count passed to kraken's --threads flag.",
    )
    kraken_segment_parser.add_argument(
        "--segment-autocast",
        action="store_true",
        help="Enable Kraken's --autocast flag to reduce GPU memory usage.",
    )
    kraken_segment_parser.add_argument(
        "--segment-no-legacy-polygons",
        action="store_true",
        help="Pass --no-legacy-polygons to Kraken to disable the legacy polygon extractor.",
    )
    subline_group = kraken_segment_parser.add_mutually_exclusive_group()
    subline_group.add_argument(
        "--segment-subline",
        action="store_true",
        help="Explicitly enable --subline-segmentation on the Kraken command.",
    )
    subline_group.add_argument(
        "--segment-no-subline",
        action="store_true",
        help="Disable subline segmentation via --no-subline-segmentation.",
    )
    kraken_segment_parser.add_argument(
        "--segment-strategy",
        choices=["baseline", "boxes"],
        help="Switch between Kraken's baseline (--baseline) or legacy box (--boxes) segmenters.",
    )
    kraken_segment_parser.add_argument(
        "--segment-direction",
        choices=["horizontal-lr", "horizontal-rl", "vertical-lr", "vertical-rl"],
        help="Override Kraken's --text-direction option.",
    )
    kraken_segment_parser.add_argument(
        "--segment-scale",
        type=float,
        help="Override Kraken's --scale value.",
    )
    kraken_segment_parser.add_argument(
        "--segment-maxcolseps",
        type=int,
        help="Set Kraken's --maxcolseps value.",
    )
    kraken_segment_parser.add_argument(
        "--segment-black-colseps",
        action="store_true",
        help="Use Kraken's --black-colseps flag instead of the default white separators.",
    )
    kraken_segment_parser.add_argument(
        "--segment-keep-hlines",
        action="store_true",
        help="Preserve horizontal lines by passing --hlines.",
    )
    kraken_segment_parser.add_argument(
        "--segment-pad",
        type=int,
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        help="Override Kraken's --pad <left> <right> values.",
    )
    kraken_segment_parser.add_argument(
        "--segment-mask",
        type=Path,
        help="Path to a segmentation mask passed as --mask.",
    )
    kraken_segment_parser.add_argument(
        "--segment-global-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Additional raw argument inserted before '-i' in the Kraken command (repeatable).",
    )
    kraken_segment_parser.add_argument(
        "--segment-cli-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Additional raw argument appended after 'kraken segment' (repeatable).",
    )
    kraken_segment_parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Disable automatic deskew/orientation normalization before Kraken segmentation.",
    )
    kraken_segment_parser.add_argument(
        "--deskew-max-angle",
        type=float,
        default=25.0,
        help="Maximum absolute skew angle (degrees) corrected automatically (default: 25).",
    )
    kraken_segment_parser.add_argument(
        "--deskew-allow-portrait",
        action="store_true",
        help="Keep portrait-oriented pages after deskew instead of rotating them to landscape.",
    )
    kraken_segment_parser.add_argument(
        "--deskew-allow-upside-down",
        action="store_true",
        help="Skip the upright heuristic that flips pages appearing upside-down after deskew.",
    )
    kraken_segment_parser.add_argument(
        "--filter-non-writing",
        action="store_true",
        help="Drop crops that look like blank background instead of handwriting.",
    )
    kraken_segment_parser.add_argument(
        "--filter-ink-threshold",
        type=float,
        default=0.01,
        metavar="RATIO",
        help="Minimum ink ratio (0-1) required to keep a crop (default: 0.01).",
    )
    kraken_segment_parser.add_argument(
        "--filter-edge-threshold",
        type=float,
        default=0.005,
        metavar="RATIO",
        help="Minimum edge density (0-1) required to keep a crop (default: 0.005).",
    )
    kraken_segment_parser.add_argument(
        "--filter-contrast-threshold",
        type=float,
        default=5.0,
        metavar="VALUE",
        help="Minimum grayscale contrast needed to keep a crop (default: 5.0).",
    )
    kraken_segment_parser.add_argument(
        "--filter-use-gpt",
        action="store_true",
        help="Ask ChatGPT to double-check borderline crops before discarding them.",
    )
    kraken_segment_parser.add_argument(
        "--filter-gpt-model",
        default=DEFAULT_GPT_FILTER_MODEL,
        help=(
            "ChatGPT model to use when --filter-use-gpt is enabled "
            f"(default: {DEFAULT_GPT_FILTER_MODEL})."
        ),
    )
    kraken_segment_parser.add_argument(
        "--filter-gpt-prompt",
        default=DEFAULT_GPT_FILTER_PROMPT,
        help="Custom prompt passed to ChatGPT when vetting borderline crops.",
    )
    kraken_segment_parser.add_argument(
        "--filter-gpt-cache-dir",
        type=Path,
        help="Optional cache directory reused across GPT foreground queries.",
    )
    kraken_segment_parser.set_defaults(func=handle_kraken_segment)

    test_parser = subparsers.add_parser(
        "test",
        help="Run OCR on a single image and print the recognised text.",
    )
    test_parser.add_argument("image", type=Path, help="Path to the image to OCR.")
    test_parser.add_argument(
        "--model",
        type=Path,
        help="Optional custom .traineddata model to use (defaults to eng).",
    )
    test_parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Directory containing tessdata files (defaults to model's folder).",
    )
    test_parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6).",
    )
    test_parser.set_defaults(func=handle_test)

    xnet_parser = subparsers.add_parser(
        "xnet",
        help="Goal-oriented OCR pipeline that sequences preprocess→segment→recognize→check→repair.",
    )
    xnet_parser.add_argument(
        "--source",
        type=Path,
        default=Path("data") / "train" / "images",
        help="Image file or directory to process (default: data/train/images).",
    )
    xnet_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPTS_DIR,
        help="Directory where page transcripts will be written (default: transcripts/raw).",
    )
    xnet_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("transcripts") / "xnet_artifacts",
        help="Directory where per-page artifacts (normalized images, crops, reports) will be written (default: transcripts/xnet_artifacts).",
    )
    xnet_parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write a per-page report.json/report.txt into the per-page artifacts folder.",
    )
    xnet_parser.add_argument(
        "--notify",
        action="store_true",
        help="Send a best-effort desktop notification/beep when the run finishes.",
    )
    # Segmentation knobs (forwarded to KrakenSegmenter)
    xnet_parser.add_argument("--kraken-model", type=Path, help="Optional Kraken segmentation model (.mlmodel).")
    xnet_parser.add_argument("--pagexml", type=Path, help="Optional directory for PAGE-XML exports.")
    xnet_parser.add_argument("--padding", type=int, default=12, help="Padding around lines before cropping.")
    xnet_parser.add_argument("--min-width", type=int, default=24, help="Minimum crop width.")
    xnet_parser.add_argument("--min-height", type=int, default=16, help="Minimum crop height.")
    xnet_parser.add_argument("--no-deskew", action="store_true", help="Disable deskew before segmentation.")
    xnet_parser.add_argument(
        "--deskew-max-angle",
        type=float,
        default=25.0,
        help="Maximum skew angle corrected automatically (default: 25).",
    )
    xnet_parser.add_argument("--deskew-allow-portrait", action="store_true", help="Keep portrait pages.")
    xnet_parser.add_argument("--deskew-allow-upside-down", action="store_true", help="Allow upside-down pages.")
    # Recognition knobs
    xnet_parser.add_argument(
        "--recognizer",
        choices=["tesseract", "kraken"],
        default="tesseract",
        help="Recognizer backend for line crops (default: tesseract).",
    )
    xnet_parser.add_argument("--tesseract-model", type=Path, help="Optional .traineddata for Tesseract.")
    xnet_parser.add_argument("--tessdata-dir", type=Path, help="Directory containing tessdata files.")
    xnet_parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode (default: 6).")
    xnet_parser.add_argument(
        "--kraken-ocr-model",
        type=Path,
        help="Kraken recognition model (.mlmodel) used when --recognizer kraken.",
    )
    xnet_parser.add_argument("--gpt-fallback", action="store_true", help="Use GPT vision fallback when needed.")
    add_gpt_arguments(xnet_parser)
    # Controller thresholds
    xnet_parser.add_argument("--min-confidence", type=float, default=55.0, help="Minimum average line confidence.")
    xnet_parser.add_argument("--min-coherence", type=float, default=60.0, help="Minimum language coherence score.")
    xnet_parser.add_argument("--min-similarity", type=float, default=0.6, help="Minimum page-level similarity to ground truth (0-1).")
    xnet_parser.add_argument("--max-retries", type=int, default=2, help="Maximum repair iterations.")
    xnet_parser.add_argument("--no-auto-tune", action="store_true", help="Disable OCR parameter auto-tuning across retries.")
    xnet_parser.add_argument("--max-images", type=int, help="Upper bound on how many images to process.")
    xnet_parser.add_argument("--gt-page-dir", type=Path, help="Directory containing page-level ground truth .gt.txt files.")
    xnet_parser.set_defaults(func=handle_xnet)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run OCR on all images inside a folder and export a CSV report.",
    )
    batch_parser.add_argument("folder", type=Path, help="Folder of images to OCR.")
    batch_parser.add_argument(
        "--model",
        type=Path,
        help="Optional custom .traineddata model to use for batch OCR.",
    )
    batch_parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Directory containing tessdata files (defaults to model's folder).",
    )
    batch_parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6).",
    )
    batch_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Where to save the CSV summary (default: results.csv).",
    )
    batch_parser.set_defaults(func=handle_batch)

    ocr_parser = subparsers.add_parser(
        "ocr",
        help="Run OCR over a file or folder using the selected engine.",
    )
    ocr_parser.add_argument(
        "--engine",
        choices=["tesseract", "kraken"],
        default="kraken",
        help="OCR engine to use (default: kraken).",
    )
    ocr_parser.add_argument(
        "--model",
        type=Path,
        help="Model to use for OCR (required for Kraken).",
    )
    ocr_parser.add_argument(
        "--in",
        dest="input_dir",
        type=Path,
        required=True,
        help="Image file or directory to process.",
    )
    ocr_parser.add_argument(
        "--out",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory where recognised text files will be written.",
    )
    ocr_parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Optional tessdata directory for Tesseract OCR.",
    )
    ocr_parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode when using --engine tesseract (default: 6).",
    )
    ocr_parser.set_defaults(func=handle_ocr)

    refine_parser = subparsers.add_parser(
        "refine",
        help="Refine OCR output by combining baseline recognition with GPT cleanup.",
    )
    refine_parser.add_argument("source", type=Path, help="Image or directory to refine.")
    refine_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REFINED_DIR,
        help="Directory where refined transcripts will be written (default: transcripts/refined).",
    )
    refine_parser.add_argument(
        "--engine",
        choices=["tesseract", "kraken"],
        default="tesseract",
        help="Baseline OCR engine providing the rough hint (default: tesseract).",
    )
    refine_parser.add_argument(
        "--tesseract-model",
        type=Path,
        help="Optional .traineddata used when extracting Tesseract tokens for hints.",
    )
    refine_parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Tessdata directory accompanying --tesseract-model when using Tesseract hints.",
    )
    refine_parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode for hint extraction (default: 6).",
    )
    refine_parser.add_argument(
        "--kraken-model",
        type=Path,
        help="Kraken model to use for the rough transcription when --engine kraken is selected.",
    )
    refine_parser.add_argument(
        "--gpt-model",
        default="gpt-4o-mini",
        help="ChatGPT model identifier used for refinement (default: gpt-4o-mini).",
    )
    refine_parser.add_argument(
        "--gpt-prompt",
        help="Override the default refinement prompt sent to ChatGPT.",
    )
    refine_parser.add_argument(
        "--gpt-cache-dir",
        type=Path,
        help="Directory for caching ChatGPT responses (optional).",
    )
    refine_parser.add_argument(
        "--gpt-max-output-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens ChatGPT may return per refinement (default: 512).",
    )
    refine_parser.add_argument(
        "--gpt-temperature",
        type=float,
        help="Optional temperature passed to ChatGPT for refinement sampling.",
    )
    refine_parser.set_defaults(func=handle_refine)

    review_parser = subparsers.add_parser(
        "review",
        help="Interactively review low-confidence OCR tokens and capture training data.",
    )
    review_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image file or directory to review.",
    )
    review_parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Confidence threshold below which tokens require review (default: 70).",
    )
    review_parser.add_argument(
        "--model",
        type=Path,
        help="Optional custom .traineddata model used during review.",
    )
    review_parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Directory containing tessdata files (defaults to model's folder).",
    )
    review_parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6).",
    )
    review_parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Where to store confirmed snippets (default: train/).",
    )
    review_parser.add_argument(
        "--auto-train",
        type=int,
        help="Automatically retrain after collecting N new samples.",
    )
    review_parser.add_argument(
        "--output-model",
        default="handwriting",
        help="Base name of the output model when auto-training (default: handwriting).",
    )
    review_parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Where to store trained models (default: models/).",
    )
    review_parser.add_argument(
        "--base-lang",
        default="eng",
        help="Base language code to fine-tune during auto-training (default: eng).",
    )
    review_parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Training iterations to run when auto-training (default: 1000).",
    )
    review_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="When auto-training, regenerate all .lstmf samples instead of reusing cached ones.",
    )
    review_parser.add_argument(
        "--unicharset-size",
        type=int,
        help="Override LSTM unicharset size when auto-training during review sessions.",
    )
    review_parser.add_argument(
        "--deserialize-check-limit",
        type=int,
        help=(
            "Maximum number of .lstmf samples to sanity-check with Tesseract before auto-training. "
            "Omit to check all samples."
        ),
    )
    review_parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable snippet previews (useful on headless systems).",
    )
    full_image_group = review_parser.add_mutually_exclusive_group()
    full_image_group.add_argument(
        "--full-image-gpt",
        dest="full_image_gpt",
        action="store_true",
        help=(
            "Request a ChatGPT transcription of the entire page before reviewing "
            "individual snippets."
        ),
    )
    full_image_group.add_argument(
        "--no-full-image-gpt",
        dest="full_image_gpt",
        action="store_false",
        help=(
            "Skip the ChatGPT full-image transcription pass and only suggest per-snippet "
            "guesses."
        ),
    )
    review_parser.set_defaults(full_image_gpt=None)
    add_gpt_arguments(review_parser)
    review_parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Tkinter-based reviewer instead of prompting in the console.",
    )
    review_parser.set_defaults(func=handle_review)

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Manually confirm transcriptions for a set of images using a GUI tool.",
    )
    annotate_parser.add_argument(
        "--engine",
        choices=["tesseract", "kraken"],
        default="kraken",
        help="Annotation engine to use for automation (default: kraken).",
    )
    annotate_parser.add_argument(
        "--seg",
        choices=["auto", "manual", "load"],
        default="auto",
        help=(
            "Segmentation mode: 'auto' for automatic segmentation with Kraken, "
            "'manual' for manual drawing, or 'load' to load from PAGE-XML files "
            "(requires --pagexml-dir). Default: auto."
        ),
    )
    annotate_parser.add_argument(
        "--export",
        choices=["lines", "pagexml"],
        default="lines",
        help="Export format for training data (default: lines).",
    )
    annotate_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image file or directory to annotate.",
    )
    annotate_parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Directory where confirmed annotations will be stored (default: train/).",
    )
    annotate_parser.add_argument(
        "--auto-train",
        type=int,
        help="Automatically retrain after collecting N new annotations.",
    )
    annotate_parser.add_argument(
        "--output-model",
        default="handwriting",
        help="Base name of the output model when auto-training (default: handwriting).",
    )
    annotate_parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Where to store trained models (default: models/).",
    )
    annotate_parser.add_argument(
        "--base-lang",
        default="eng",
        help="Base language code to fine-tune during auto-training (default: eng).",
    )
    annotate_parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Training iterations to run when auto-training (default: 1000).",
    )
    annotate_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate all .lstmf samples during auto-training instead of reusing cached ones.",
    )
    annotate_parser.add_argument(
        "--unicharset-size",
        type=int,
        help="Override LSTM unicharset size for auto-training sessions.",
    )
    annotate_parser.add_argument(
        "--deserialize-check-limit",
        type=int,
        help=(
            "Maximum number of .lstmf samples to sanity-check with Tesseract before auto-training. "
            "Omit to check all samples."
        ),
    )
    annotate_parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Path to tessdata directory containing base traineddata files.",
    )
    annotate_parser.add_argument(
        "--output-log",
        type=Path,
        help=(
            "Optional CSV file to append annotation metadata (page, transcription, timestamp)."
        ),
    )
    annotate_parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPTS_DIR,
        help=(
            "Directory where confirmed transcriptions will be written "
            "(default: transcripts/raw/)."
        ),
    )
    annotate_parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="Do not write confirmed transcriptions to the transcripts directory.",
    )
    annotate_parser.add_argument(
        "--no-prefill",
        action="store_true",
        help="Disable automatic OCR transcription prefill when annotating.",
    )
    annotate_parser.add_argument(
        "--prefill-model",
        type=Path,
        help="Optional traineddata or Kraken model used to prefill transcriptions.",
    )
    annotate_parser.add_argument(
        "--prefill-tessdata",
        type=Path,
        help="Tessdata directory to accompany --prefill-model when using Tesseract.",
    )
    annotate_parser.add_argument(
        "--prefill-psm",
        type=int,
        help="Tesseract page segmentation mode for prefill OCR (default: 6).",
    )
    annotate_parser.add_argument(
        "--pagexml-dir",
        type=Path,
        help=(
            "Directory containing pre-existing PAGE-XML annotations to load and edit. "
            "When provided, enables 'load' segmentation mode where boxes are loaded from XML files "
            "instead of auto-generated."
        ),
    )
    add_gpt_arguments(annotate_parser)
    annotate_parser.set_defaults(func=handle_annotate)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
