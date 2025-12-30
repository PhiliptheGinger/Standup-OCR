import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:  # pragma: no cover - script usage only
    from src.preprocessing import OrientationOptions, load_normalized_image
except ImportError:  # pragma: no cover
    from preprocessing import OrientationOptions, load_normalized_image

def normalize(folder: Path) -> None:
    options = OrientationOptions()
    for p in folder.glob("*.png"):
        try:
            normalized, meta = load_normalized_image(p, options=options)
            normalized.save(p)
            if meta.applied:
                print(f"Normalized {p.name} (angle={meta.angle:.2f}°)")
        except Exception as exc:
            print(f"Skip {p.name}: {exc}")

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train/lines")
    normalize(target)