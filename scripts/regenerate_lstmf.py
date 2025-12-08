"""Small helper: regenerate .lstmf files for a training directory.

Usage:
    python scripts/regenerate_lstmf.py --train-dir train --model-dir models/my_training

This script walks images in --train-dir and attempts to create their .lstmf
files inside the provided --model-dir training work directory using the
helpers in src.training. It emits a compact report at the end.
"""
from pathlib import Path
import sys
# Ensure repo root is on sys.path so "from src..." works when script is run directly.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import argparse
import logging

from src.training import _discover_images, _prepare_ground_truth, _generate_lstmf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True, help="Directory with training images")
    p.add_argument("--work-dir", required=True, help="Working model dir to write artifacts into")
    p.add_argument("--base-lang", default="eng", help="Base traineddata language (default: eng)")
    args = p.parse_args()

    train_dir = Path(args.train_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    images = _discover_images(train_dir)
    successes = []
    failures = []
    for img in images:
        try:
            gt_path = img.with_suffix('.gt.txt')
            if not gt_path.exists():
                logging.info('Skipping %s: missing .gt.txt', img.name)
                failures.append((img, 'missing-gt'))
                continue
            _, _ = _prepare_ground_truth(img, gt_path.read_text(encoding='utf-8').strip(), work_dir)
            lstmf = _generate_lstmf(img, work_dir, args.base_lang)
            if lstmf:
                successes.append((img, lstmf))
                logging.info('Generated %s', lstmf.name)
            else:
                failures.append((img, 'generation-failed'))
        except Exception as e:
            logging.exception('Error processing %s: %s', img, e)
            failures.append((img, str(e)))

    print('\n--- Regeneration report ---')
    print('Successes: %d' % len(successes))
    for img, lstmf in successes:
        print(f'  {img.name} -> {lstmf.name}')
    print('Failures: %d' % len(failures))
    for img, why in failures:
        print(f'  {img.name}: {why}')


if __name__ == '__main__':
    main()
