from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .ocr import ocr_image_to_lines


@dataclass
class TextRegion:
    """
    Represents a piece of text on a comic/manga page.

    This is intentionally generic so we can later plug in:
    - real bounding boxes from a layout detector
    - a classifier that distinguishes dialogue vs SFX vs narration, etc.
    """
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in image coordinates
    kind: str                        # "dialogue", "sfx", "narration", "ui", ...
    reading_index: int               # order in which it should be read


def detect_text_regions(image_path: str) -> List[TextRegion]:
    """
    Placeholder layout detector.

    Current behavior:
      - Calls OCR to get cleaned lines.
      - Assigns fake vertical bounding boxes.
      - Marks everything as 'dialogue'.
      - Sets reading_index = 0, 1, 2, ... (simple top-to-bottom order).

    Later:
      - This is where we'd add panel/bubble detection, SFX classification,
        and right-to-left reading order logic for Japanese manga.
    """
    lines = ocr_image_to_lines(image_path)

    regions: List[TextRegion] = []
    x = 100
    y_start = 100
    line_height = 40

    for i, line in enumerate(lines):
        y = y_start + i * line_height
        bbox = (x, y, 300, line_height)  # fake for now

        region = TextRegion(
            text=line,
            bbox=bbox,
            kind="dialogue",
            reading_index=i,
        )
        regions.append(region)

    return regions
