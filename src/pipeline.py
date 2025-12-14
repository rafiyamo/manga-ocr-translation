"""
High-level pipeline that connects OCR and translation.
"""

from .ocr import placeholder_ocr
from .translate import placeholder_translate


def process_page(image_path: str) -> str:
    """
    Very early placeholder pipeline.
    For now, it ignores the image and just runs placeholder functions.
    """
    jp_text = placeholder_ocr()
    en_text = placeholder_translate(jp_text)
    return f"JP: {jp_text}\nEN: {en_text}"