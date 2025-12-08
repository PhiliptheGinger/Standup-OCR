import sys
from pathlib import Path
from PIL import Image, ImageOps

def normalize(folder: Path):
    for p in folder.glob("*.png"):
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)
            w, h = img.size
            if h > w * 1.1:
                img = img.rotate(90, expand=True)
                img.save(p)
                print(f"Rotated {p.name} -> {img.size}")
        except Exception as e:
            print(f"Skip {p.name}: {e}")

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train/lines")
    normalize(target)