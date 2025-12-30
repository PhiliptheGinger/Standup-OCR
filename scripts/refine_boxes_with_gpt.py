#!/usr/bin/env python3
"""Refine Kraken box placement using ChatGPT's vision capabilities.

This script:
1. Loads a page image and its segmentation boxes (from .boxes.json)
2. Sends the image + current boxes to GPT for validation
3. Parses GPT's suggestions to correct/remove/add boxes
4. Saves corrected metadata back to .boxes.json files
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def load_env_api_key() -> Optional[str]:
    """Load OpenAI API key from .env or environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    import os
    return os.getenv("OPENAI_API_KEY")

def get_boxes_for_page(metadata_dir: Path, source_image: str) -> dict[str, dict]:
    """Load all .boxes.json files for a given source image, grouped by source."""
    boxes_by_page = {}
    
    for json_file in sorted(metadata_dir.glob("*.boxes.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        src = data.get("source_image", "")
        if src == source_image or Path(src).name == Path(source_image).name:
            if src not in boxes_by_page:
                boxes_by_page[src] = []
            boxes_by_page[src].append((json_file, data))
    
    return boxes_by_page

def draw_boxes_on_image(image: Image.Image, boxes_data: list[dict]) -> Image.Image:
    """Draw current boxes on image for GPT to review."""
    display = image.copy()
    draw = ImageDraw.Draw(display, "RGBA")
    
    for i, box_info in enumerate(boxes_data):
        bbox = box_info.get("bbox_original") or box_info.get("bbox")
        if bbox:
            left, top, right, bottom = bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]
            # Draw box with line number label
            draw.rectangle(
                [(left, top), (right, bottom)],
                outline=(255, 100, 100, 200),
                width=2,
            )
            # Draw label
            line_idx = box_info.get("line_index", i + 1)
            draw.text((left + 2, top + 2), str(line_idx), fill=(255, 100, 100, 255))
    
    return display

def ask_gpt_about_boxes(
    client: OpenAI,
    image_path: Path,
    boxes_data: list[dict],
    image_with_boxes: Image.Image,
) -> str:
    """Ask GPT to validate and suggest corrections for box placement."""
    
    import base64
    from io import BytesIO
    
    # Encode the image with boxes drawn
    buf = BytesIO()
    image_with_boxes.save(buf, format="PNG")
    b64_image = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    
    # Build current box description
    box_desc = "Current detected text line boxes:\n"
    for i, box_info in enumerate(boxes_data):
        bbox = box_info.get("bbox_original") or box_info.get("bbox")
        line_idx = box_info.get("line_index", i + 1)
        if bbox:
            box_desc += f"  Line {line_idx}: ({bbox['left']}, {bbox['top']}) to ({bbox['right']}, {bbox['bottom']})\n"
    
    prompt = f"""You are reviewing handwritten page segmentation. The image shows a handwritten page with red boxes overlaid.

{box_desc}

Please review the box placement and identify:
1. **Correct boxes**: Line numbers where the box correctly captures a line of text
2. **Wrong boxes**: Line numbers where the box misses the text, includes background, or is misaligned
3. **Missing boxes**: Areas with handwritten text that should have boxes but don't
4. **Adjust boxes**: Line numbers where the box needs to be moved/resized (suggest rough direction: left/right/up/down)

Format your response as JSON:
{{
    "correct_lines": [line_numbers_that_look_good],
    "wrong_lines": [line_numbers_to_remove],
    "missing_text_regions": ["description of where text is that needs a box"],
    "needs_adjustment": {{"line_number": "description of adjustment needed"}}
}}

Be concise but thorough. Focus on accuracy."""

    try:
        # Try newer API (v1.0+)
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except AttributeError:
        # Fallback for older API structure
        response = client.messages.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )
        return response.content[0].text

def refine_boxes_with_gpt(
    metadata_dir: Path,
    image_path: Path,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    """Refine box placement for all metadata in metadata_dir using GPT."""
    
    if OpenAI is None:
        log.error("OpenAI library not installed. Install with: pip install openai")
        sys.exit(1)
    
    api_key = load_env_api_key()
    if not api_key:
        log.error("OPENAI_API_KEY not set in environment or .env file")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    metadata_dir = Path(metadata_dir)
    image_path = Path(image_path)
    output_dir = Path(output_dir) if output_dir else metadata_dir
    
    if not image_path.exists():
        log.error(f"Image not found: {image_path}")
        sys.exit(1)
    
    if not metadata_dir.exists():
        log.error(f"Metadata directory not found: {metadata_dir}")
        sys.exit(1)
    
    # Load image
    with Image.open(image_path) as img:
        original_image = img.convert("RGB")
    
    # Collect all boxes for this page
    source_key = str(image_path)
    all_boxes_data = []
    json_files = []
    
    for json_file in sorted(metadata_dir.glob("*.boxes.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        src = data.get("source_image", "")
        # Match by source image name
        if Path(src).name == image_path.name or src == source_key:
            all_boxes_data.append(data)
            json_files.append(json_file)
    
    if not all_boxes_data:
        log.warning(f"No boxes found for {image_path.name}")
        return
    
    log.info(f"Found {len(all_boxes_data)} boxes for {image_path.name}")
    log.info("Drawing boxes and sending to GPT for review...")
    
    # Draw boxes on image
    image_with_boxes = draw_boxes_on_image(original_image, all_boxes_data)
    
    # Ask GPT for suggestions
    gpt_response = ask_gpt_about_boxes(client, image_path, all_boxes_data, image_with_boxes)
    log.info(f"\nGPT Response:\n{gpt_response}\n")
    
    # Try to parse JSON response
    try:
        suggestions = json.loads(gpt_response)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', gpt_response, re.DOTALL)
        if json_match:
            suggestions = json.loads(json_match.group(0))
        else:
            log.warning("Could not parse GPT response as JSON. Skipping corrections.")
            return
    
    # Apply suggestions
    correct_lines = set(suggestions.get("correct_lines", []))
    wrong_lines = set(suggestions.get("wrong_lines", []))
    needs_adjustment = suggestions.get("needs_adjustment", {})
    
    log.info(f"\nApplying corrections:")
    log.info(f"  Correct: {correct_lines}")
    log.info(f"  Wrong (to remove): {wrong_lines}")
    log.info(f"  Needs adjustment: {needs_adjustment}")
    
    if dry_run:
        log.info("(Dry run: not saving changes)")
        return
    
    # Update metadata files
    for json_file, data in zip(json_files, all_boxes_data):
        line_idx = data.get("line_index")
        
        if line_idx in wrong_lines:
            log.info(f"  Marking {json_file.name} (line {line_idx}) for removal")
            data["gpt_review"] = {"action": "remove", "reason": "GPT flagged as incorrect"}
        elif line_idx in correct_lines:
            log.info(f"  Confirming {json_file.name} (line {line_idx})")
            data["gpt_review"] = {"action": "keep", "reason": "GPT confirmed"}
        elif str(line_idx) in needs_adjustment:
            log.info(f"  Flagging {json_file.name} (line {line_idx}) for adjustment: {needs_adjustment[str(line_idx)]}")
            data["gpt_review"] = {"action": "adjust", "suggestion": needs_adjustment[str(line_idx)]}
        
        # Save updated metadata
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / json_file.name
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    log.info(f"\nBoxes updated and saved to {output_dir}")
    log.info("Review the 'gpt_review' field in .boxes.json files to see GPT's assessment")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Refine Kraken segmentation boxes using ChatGPT's vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refine boxes for a single image
  python scripts/refine_boxes_with_gpt.py \\
    --metadata-dir train/kraken_auto_lines_fresh \\
    --image data/train/images/001.jpg \\
    --output-dir train/kraken_auto_lines_refined

  # Dry run to see GPT's suggestions without saving
  python scripts/refine_boxes_with_gpt.py \\
    --metadata-dir train/kraken_auto_lines_fresh \\
    --image data/train/images/001.jpg \\
    --dry-run
        """,
    )
    
    parser.add_argument("--metadata-dir", required=True, help="Directory with .boxes.json files")
    parser.add_argument("--image", required=True, help="Path to the source image")
    parser.add_argument("--output-dir", help="Where to save refined .boxes.json files (default: same as metadata-dir)")
    parser.add_argument("--dry-run", action="store_true", help="Show GPT suggestions without saving")
    
    args = parser.parse_args()
    
    refine_boxes_with_gpt(
        metadata_dir=args.metadata_dir,
        image_path=args.image,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
