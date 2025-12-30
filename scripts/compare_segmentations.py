#!/usr/bin/env python3
"""Compare GPT and Kraken segmentation on the same image."""

import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def compare_segmentations(image_path: Path, gpt_segments_path: Path, kraken_metadata_dir: Path):
    """Draw both GPT and Kraken boxes on the image for comparison."""
    
    # Load image
    img = Image.open(image_path)
    width, height = img.size
    
    # Apply EXIF orientation
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except:
        pass
    
    # Load GPT segments
    with open(gpt_segments_path, 'r') as f:
        gpt_segments = json.load(f)
    
    # Aggregate Kraken boxes from individual line files
    kraken_boxes = []
    image_stem = image_path.stem
    kraken_metadata_dir = Path(kraken_metadata_dir)
    
    # Find all line files for this image
    line_files = sorted(kraken_metadata_dir.glob(f"{image_stem}_line*.boxes.json"))
    print(f"Found {len(line_files)} Kraken line files")
    
    for line_file in line_files:
        try:
            with open(line_file, 'r') as f:
                line_data = json.load(f)
            
            if 'bbox_original' in line_data:
                bbox_obj = line_data['bbox_original']
            elif 'bbox' in line_data:
                bbox_obj = line_data['bbox']
            else:
                continue
            
            # Convert from {left, top, right, bottom} to [x1, y1, x2, y2]
            bbox = [bbox_obj['left'], bbox_obj['top'], bbox_obj['right'], bbox_obj['bottom']]
            kraken_boxes.append(bbox)
        except Exception as e:
            print(f"Warning: Failed to read {line_file}: {e}")
    
    # Create output image (side by side)
    combined = Image.new('RGB', (width * 2, height), color='white')
    combined.paste(img, (0, 0))
    combined.paste(img, (width, 0))
    
    draw_left = ImageDraw.Draw(combined)
    draw_right = ImageDraw.Draw(combined)
    
    print(f"Drawing {len(gpt_segments)} GPT segments in green...")
    for seg in gpt_segments:
        bbox = seg['bbox']  # [x1, y1, x2, y2]
        draw_left.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='green', width=2)
        # Add text label
        try:
            draw_left.text((bbox[0], bbox[1]), seg.get('text', '')[:20], fill='green')
        except:
            pass
    
    print(f"Drawing {len(kraken_boxes)} Kraken segments in red...")
    for bbox in kraken_boxes:
        draw_right.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='red', width=2)
    
    # Save
    output_path = image_path.parent / f"{image_path.stem}_comparison.png"
    combined.save(output_path)
    print(f"\nComparison saved to: {output_path}")
    print(f"Left side (green): GPT segmentation ({len(gpt_segments)} lines)")
    print(f"Right side (red): Kraken segmentation ({len(kraken_boxes)} lines)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare GPT and Kraken segmentation side-by-side."
    )
    
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--gpt-json", help="Path to GPT segments JSON (default: auto-detect)")
    parser.add_argument("--kraken-dir", help="Path to Kraken metadata directory (default: auto-detect)")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    
    # Auto-detect paths
    gpt_json = Path(args.gpt_json) if args.gpt_json else image_path.parent / f"{image_path.stem}.gpt_segments.json"
    
    # Find Kraken metadata directory - look in train/kraken_auto_lines_fresh or similar
    kraken_dir = None
    if args.kraken_dir:
        kraken_dir = Path(args.kraken_dir)
    else:
        # Search for metadata directory
        for search_dir in [
            Path("train/kraken_auto_lines_fresh"),
            Path("train/kraken_auto_lines"),
            Path("train/kraken_lines"),
        ]:
            if search_dir.exists():
                kraken_dir = search_dir
                break
    
    if not gpt_json.exists():
        print(f"ERROR: GPT JSON not found: {gpt_json}")
        sys.exit(1)
    
    if not kraken_dir or not kraken_dir.exists():
        print(f"ERROR: Kraken metadata directory not found: {kraken_dir}")
        sys.exit(1)
    
    compare_segmentations(image_path, gpt_json, kraken_dir)

if __name__ == "__main__":
    main()
