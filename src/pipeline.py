from __future__ import annotations

from typing import List, Tuple

from .ocr import ocr_image_to_lines
from .translate import translate_lines_with_model, placeholder_translate


def process_page(image_path: str, use_model: bool = True) -> Tuple[List[str], List[str]]:
    """
    High-level pipeline for a single manga page.

    1. Run OCR on the image to get raw text lines.
    2. Translate those lines to English, using our custom model by default.
       If loading or running the model fails, we fall back to the placeholder
       translation so the CLI still works.

    Returns:
        (jp_lines, en_lines)
    """
    jp_lines: List[str] = ocr_image_to_lines(image_path)

    if not jp_lines:
        return [], []

    # Default: try the real model
    if use_model:
        try:
            en_lines: List[str] = translate_lines_with_model(jp_lines)
        except Exception:
            # Safe fallback if anything goes wrong with the model
            en_lines = [placeholder_translate(line) for line in jp_lines]
    else:
        # Explicitly requested to skip the model
        en_lines = [placeholder_translate(line) for line in jp_lines]

    return jp_lines, en_lines
