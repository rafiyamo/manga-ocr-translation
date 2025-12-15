"""
OCR-related helper functions.
"""

from typing import List
from pathlib import Path

from PIL import Image
import pytesseract

# Explicitly point to the Tesseract executable on Windows:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_image_to_lines(image_path: str) -> List[str]:
    """
    Run OCR on an image and return a list of non-empty text lines.
    For now we assume mostly English text to test that OCR works.
    Later we can switch to Japanese ("jpn") or "jpn+eng".
    """
    img_path = Path(image_path)

    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path)
    raw_text = pytesseract.image_to_string(img, lang="eng")  # change to "jpn" later

    lines = [line.strip() for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

    return lines
