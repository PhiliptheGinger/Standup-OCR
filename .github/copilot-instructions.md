## Standup-OCR — Copilot instructions

This project is a small toolkit for handwritten note OCR, review, and
fine-tuning Tesseract/handwriting models. The file below captures the
project-specific knowledge an AI assistant needs to be productive quickly.

- Big picture
  - CLI entrypoints: `main.py` (top-level) and package entry `python -m src.ocr.main`.
  - Major components:
    - `src/training.py` — orchestrates Tesseract LSTM training (prepare GT, makebox, lstmtraining, combine_tessdata).
    - `src/gpt_ocr.py` — ChatGPT multimodal transcription wrapper (handles .env loading and optional caching).
    - `src/kraken_adapter.py` — Kraken integration (CLI and Python API), ketos training, and segmentation helpers.
    - `src/annotation.py` + `src/exporters.py` — Tkinter annotation GUI and exporters for `lines` / PAGE-XML formats.
    - `src/preprocessing.py` and `src/ocr` modules contain image preprocessing and OCR runtime logic used across tools.

- Key developer workflows and commands (copyable examples visible in `README.md`)
  - Run the main OCR pipeline: `python -m src.ocr.main --agentic`
  - Re-run failed: `python -m src.ocr.main --rerun-failed`
  - Review low-confidence tokens: `python main.py review --source <img_or_folder> --train-dir train`
  - Annotate pages with GUI: `python main.py annotate --source <folder> --train-dir train`
  - Train a Tesseract model: `python main.py train --train-dir train --output-model handwriting`

- Important environment and integration points
  - Tesseract training tools required: `lstmtraining`, `combine_tessdata`, `text2image` must be on PATH for training on Windows. See README section "Windows training prerequisites".
  - `TESSDATA_PREFIX` or `--tessdata-dir` is used to locate base `.traineddata` files. `src/training.py::_resolve_tessdata_dir` shows the discovery order.
  - OpenAI: set `OPENAI_API_KEY` or put in `.env`. `src/gpt_ocr.py` implements a simple `.env` loader — prefer setting env var for CI.
  - Kraken: `kraken` and `ketos` executables are used by `src/kraken_adapter.py`. If present, the annotation tool can prefill using Kraken models.

- Project-specific conventions and patterns
  - Training image naming encodes labels: `<prefix>_<label>.png`. See `_extract_label` in `src/training.py` (dash `-` maps to space in labels).
  - Training artifacts are written under `models/<output>_training/` and final `.traineddata` to `models/`.
  - Annotation exports: `train/lines/*.png` paired with `*.gt.txt` (Kraken style) or `train/pagexml/*.xml` (PAGE-XML). See `src/exporters.py` for exact formats.
  - GPT OCR caching: `GPTTranscriber(cache_dir=...)` saves transcriptions as `<stem>.txt`.
  - Optional deps handled at runtime (OpenAI, Kraken, cv2). Search for guarded imports (try/except ImportError) in `src/`.

- Examples in the repo to reference when implementing changes
  - Training flow and makebox/LSTM calls: `src/training.py`.
  - ChatGPT OCR usage and `.env` behaviour: `src/gpt_ocr.py`.
  - Kraken CLI/API fallbacks and segmentation logic: `src/kraken_adapter.py`.
  - Annotation GUI behaviour and auto-train hook: `src/annotation.py`.
  - Export formats: `src/exporters.py` (line crops + `.gt.txt`, PAGE-XML exporter).
  - Tests demonstrating expected behaviours: `tests/test_training_paths.py` (tessdata discovery, bootstrap sample image).

- Practical hints for code edits and PRs
  - Preserve the graceful optional-dependency handling style (prefer guarded imports and helpful runtime messages). See `src/gpt_ocr.py` and `src/kraken_adapter.py`.
  - When changing training flow, include a test or update `tests/test_training_paths.py`-style fixtures for tessdata discovery and sample bootstrap logic.
  - Prefer small, focused CLI flag additions; mirror CLI help text in `README.md` if behaviour is visible to users.

If any part of this feels incomplete or you want more examples (e.g. exact log messages, sample model files under `models/finetuned_training/`, or a short checklist for adding a new OCR backend), tell me which area to expand and I will iterate.

When the user asks for a command, reply with the exact CLI invocation only (no prompt prefix) and wrap it in a fenced code block (triple backticks) so it’s directly copy/pasteable.
