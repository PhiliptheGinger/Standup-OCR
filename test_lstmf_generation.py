"""Quick test script to validate LSTMF generation with new validation."""
import os
from pathlib import Path
from src.training import _generate_lstmf

os.environ["TRAIN_TIMEOUT_S"] = "60"
os.environ["MAKEBOX_TIMEOUT_S"] = "30"

work_dir = Path("models/handwriting_training")
images_dir = Path("data/train/images")
test_images = sorted(images_dir.glob("*.jpg"))[:3]

print(f"Testing LSTMF generation with {len(test_images)} images...")
for i, img in enumerate(test_images, 1):
    try:
        result = _generate_lstmf(img, work_dir, "eng")
        print(f"✓ Test {i}/3: {img.name} -> {result.name}")
    except Exception as e:
        print(f"✗ Test {i}/3: {img.name} FAILED: {e}")
