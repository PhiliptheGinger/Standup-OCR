"""Command-line interface for the handwriting OCR toolkit."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from src.annotation import AnnotationAutoTrainConfig, AnnotationOptions, annotate_images
from src.ocr import ocr_image
from src.gpt_ocr import GPTTranscriber, GPTTranscriptionError
from src.review import ReviewAborted, ReviewConfig, ReviewSession
from src.training import SUPPORTED_EXTENSIONS, train_model
from src.refine import DEFAULT_REFINE_PROMPT, run_refinement
from src.kraken_adapter import is_available as kraken_available, ocr as kraken_ocr, train as kraken_train


DEFAULT_TRAIN_DIR = Path("train")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_RESULTS_FILE = Path("results.csv")
DEFAULT_TRANSCRIPTS_DIR = Path("transcripts") / "raw"
DEFAULT_REFINED_DIR = Path("transcripts") / "refined"


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


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
        model_path = kraken_train(
            args.train_dir,
            Path(model_out),
            epochs=args.epochs,
            val_split=args.val_split,
            base_model=base_model,
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
        use_gpt_ocr=not args.no_gpt_ocr,
        gpt_model=args.gpt_model,
        gpt_prompt=args.gpt_prompt,
        gpt_cache_dir=args.gpt_cache_dir,
        gpt_max_output_tokens=args.gpt_max_output_tokens,
        gpt_max_images=args.gpt_max_images,
    )
    logging.info("Model saved to %s", model_path)


def handle_test(args: argparse.Namespace) -> None:
    text = ocr_image(
        args.image,
        model_path=args.model,
        tessdata_dir=args.tessdata_dir,
        psm=args.psm,
    )
    print(text)


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
    )
    session: ReviewSession
    last_trained_count = 0
    gpt_max_images = args.gpt_max_images
    transcriber: Optional[GPTTranscriber] = None

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
                use_gpt_ocr=not args.no_gpt_ocr,
                gpt_model=args.gpt_model,
                gpt_prompt=args.gpt_prompt,
                gpt_cache_dir=args.gpt_cache_dir,
                gpt_max_output_tokens=args.gpt_max_output_tokens,
                gpt_max_images=args.gpt_max_images,
            )
            logging.info("Updated model saved to %s", model_path)
            last_trained_count += args.auto_train

    session = ReviewSession(
        config,
        on_sample_saved=lambda *_args: maybe_train(),
        transcriber=transcriber,
        gpt_max_images=gpt_max_images,
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
        )

    if args.engine == "kraken" and args.seg == "auto" and not kraken_available():
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
    add_gpt_arguments(train_parser)
    train_parser.set_defaults(func=handle_train)

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
        "--no-preview",
        action="store_true",
        help="Disable snippet previews (useful on headless systems).",
    )
    add_gpt_arguments(review_parser)
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
        choices=["auto", "manual"],
        default="auto",
        help="Segmentation mode (default: auto).",
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
