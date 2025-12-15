"""
High-level pipeline that connects OCR and translation.
"""

from typing import List

from .ocr import ocr_image_to_lines
from .translate import placeholder_translate


def process_page(image_path: str) -> str:
    """
    Pipeline:
    - Run OCR on the image
    - Join lines into one text block
    - Pass to placeholder_translate (for now)
    - Return a formatted string with numbered lines + translation
    """
    lines: List[str] = ocr_image_to_lines(image_path)

    if not lines:
        return "No text detected on the page."

    combined_text = "\n".join(lines)
    translated = placeholder_translate(combined_text)

    result_lines: List[str] = ["Detected text lines:"]
    for i, line in enumerate(lines, start=1):
        result_lines.append(f"[{i}] {line}")

    result_lines.append("")
    result_lines.append("Placeholder translation output:")
    result_lines.append(translated)

    return "\n".join(result_lines)
