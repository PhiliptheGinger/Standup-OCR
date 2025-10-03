# Standup-OCR

Standup-OCR is a small toolkit that helps capture handwritten stand-up notes,
correct OCR mistakes, and fine-tune a custom Tesseract model on the confirmed
samples.

## Command-line interface

Install the dependencies listed in `requirements.txt`, then invoke the CLI via

```bash
python main.py --help
```

### Reviewing OCR output

Use the `review` subcommand to step through low-confidence tokens produced by
`ocr_detailed`. The tool crops each region from the preprocessed image and asks
for a corrected transcription. Confirmed snippets are written to the training
directory (`train/` by default) using the `<prefix>_<label>.png` naming
convention so they can be consumed by the existing training pipeline.

```bash
python main.py review --source path/to/image_or_folder --threshold 75 \
    --train-dir train --auto-train 10
```

Key behaviour:

* `--source` accepts either a single image or a folder containing images.
* `--threshold` controls which OCR tokens require review (defaults to 70).
* `--train-dir` points to the folder where confirmed snippets will be saved.
* `--auto-train N` triggers `train_model` once every `N` newly confirmed
  samples, reusing the options passed to the command.
* Use `--no-preview` on headless systems to disable image previews. Otherwise
  the tool attempts to display each snippet using `PIL.Image.show()` and falls
  back to logging the path of a temporary PNG file when required.

During review you can:

* Press **Enter** to accept the recognised text (when available).
* Type a corrected transcription to save it under that label.
* Enter `s` to skip a snippet or `q` to end the session early.

Each confirmed snippet is recorded in `train/review_log.jsonl`, preventing the
same bounding box from being queued again in future sessions.

### Training a custom model

Once you have collected a set of labelled snippets run:

```bash
python main.py train --train-dir train --output-model handwriting
```

Use the `--model` flag on the `test`, `batch`, or `review` subcommands to
evaluate the updated model.
