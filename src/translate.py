"""
Translation-related helper functions.

Right now we keep a simple placeholder_translate() function so that the
pipeline runs end-to-end while we work on OCR and system design.

Later, this file will load our custom PyTorch translation model and
provide a translate_lines_with_model() function that uses that model.
"""

from typing import List


def placeholder_translate(text: str) -> str:
    """
    Temporary placeholder so the pipeline can run.

    Eventually this will be replaced (or wrapped) by a function that calls
    our own trained translation model.
    """
    return f"[EN translation of]: {text}"


def translate_lines_with_model(lines: List[str]) -> List[str]:
    """
    Stub for future model-based translation.

    Once the custom translation model is implemented and trained
    (in notebooks/train_translation_model.ipynb), this function will:

    - load the trained model weights from models/translator.pt
    - run inference on the given lines
    - return a list of English translations (same length as `lines`)
    """
    raise NotImplementedError("Model-based translation not implemented yet.")
