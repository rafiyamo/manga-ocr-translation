from __future__ import annotations

from typing import List, Tuple

from .layout import TextRegion, detect_text_regions
from .translate import translate_lines_with_model, placeholder_translate


def process_page(image_path: str, use_model: bool = True) -> Tuple[List[TextRegion], List[str]]:
    """
    High-level pipeline for a single manga page.

    1. Detect text regions on the page (currently a thin wrapper around OCR).
    2. Translate those regions' texts, using our custom model by default.
       If loading or running the model fails, we fall back to the placeholder
       translation so the CLI still works.

    Returns:
        regions: List[TextRegion] in reading order
        en_lines: List[str] translations, aligned 1:1 with regions
    """
    regions: List[TextRegion] = detect_text_regions(image_path)

    if not regions:
        return [], []

    jp_lines = [r.text for r in regions]

    if use_model:
        try:
            en_lines: List[str] = translate_lines_with_model(jp_lines)
        except Exception:
            en_lines = [placeholder_translate(line) for line in jp_lines]
    else:
        en_lines = [placeholder_translate(line) for line in jp_lines]

    return regions, en_lines
