# ComicBubble
JP → EN manga/comic OCR & translation pipeline with a CLI and minimal web UI.

ComicBubble takes a scanned manga page, detects speech bubbles / text regions,
runs Japanese OCR, and feeds the text into a custom JP→EN translation model
trained on open parallel corpora.

---

## What ComicBubble does

- **Detects speech bubbles / text regions**
  - Uses `src/layout.py` + `src/ocr.py` to find regions that look like dialogue / text.
  - Extracts raw Japanese text from each region with a Japanese OCR backend.

- **Translates JP → EN**
  - Character-level encoder–decoder model in PyTorch (`src/translate.py`).
  - Trained on a cleaned subset of JP–EN parallel data from OPUS/JESC + Tatoeba.
  - Model weights live in `models/translator.pt`.

- **Shows results region-by-region**
  - For each detected region: bounding box, original JP text, and EN translation.
  - Available via:
    - **CLI** (`src/cli.py`) – prints everything in the terminal.
    - **Web app** (`web_app.py`) – upload a page image and see it + translations side-by-side.
   
## How to run ComicBubble locally
**1. Clone the repo**
'''
git clone https://github.com/<your-username>/manga-ocr-translation.git
cd manga-ocr-translation
'''
