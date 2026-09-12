"""
generate_screenshots.py
------------------------
Renders terminal-style PNG "screenshots" of representative application
runs (one for each search mode) into output/screenshots/.

NOTE: These are auto-generated captures of the actual program output
(not mockups/fake text) -- they run the real engine and render its
real stdout into a monospace terminal-style image. When you run this
project on your own machine, feel free to replace these with real
OS screenshots of your terminal for the submission.
"""

import os
from PIL import Image, ImageDraw, ImageFont

from main import build_engine, print_ranked_results, print_phrase_results, print_proximity_results
import io
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "screenshots")


def capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


def render_terminal_image(text: str, path: str, title: str):
    font_path_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    font = None
    for fp in font_path_candidates:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 15)
            break
    if font is None:
        font = ImageFont.load_default()

    lines = text.rstrip("\n").split("\n")
    char_w, line_h = 9, 19
    width = max(len(l) for l in lines) * char_w + 40
    height = (len(lines) + 3) * line_h + 40

    img = Image.new("RGB", (width, max(height, 200)), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Fake terminal title bar
    draw.rectangle([0, 0, width, 30], fill=(50, 50, 50))
    draw.ellipse([10, 8, 24, 22], fill=(255, 95, 86))
    draw.ellipse([32, 8, 46, 22], fill=(255, 189, 46))
    draw.ellipse([54, 8, 68, 22], fill=(39, 201, 63))
    draw.text((80, 6), title, fill=(220, 220, 220), font=font)

    y = 40
    for line in lines:
        draw.text((15, y), line, fill=(0, 255, 120), font=font)
        y += line_h

    img.save(path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    documents, inv_index, pos_index, vsm = build_engine()

    header = (
        "======================================================================\n"
        "  Clothing Search Engine  (CSD358 - Assignment 1)\n"
        "======================================================================\n"
        f"Indexed {len(documents)} documents. Vocabulary size: {len(inv_index.vocabulary())}\n"
    )

    # 1. Free-text search screenshot
    out1 = header + capture(print_ranked_results, vsm, inv_index, "cotton shirt for men")
    render_terminal_image(out1, os.path.join(OUTPUT_DIR, "1_free_text_search.png"),
                           "Mode 1: Free-text VSM search")

    # 2. Phrase search screenshot
    out2 = header + capture(print_phrase_results, pos_index, "cotton shirt")
    render_terminal_image(out2, os.path.join(OUTPUT_DIR, "2_phrase_search.png"),
                           "Mode 2: Exact phrase search")

    # 3. Proximity search screenshot
    out3 = header + capture(print_proximity_results, pos_index, "stretch", "denim", 4)
    render_terminal_image(out3, os.path.join(OUTPUT_DIR, "3_proximity_search.png"),
                           "Mode 3: Proximity search")

    print(f"Screenshots written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
