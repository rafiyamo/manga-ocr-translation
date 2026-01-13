# ComicBubble🫧

Japenese → English manga/comic OCR & translation pipeline with a CLI and minimal web UI.

ComicBubble takes a scanned manga page, detects speech bubbles / text regions,
runs Japanese OCR, and feeds the text into a custom JP→EN translation model
trained on open parallel corpora.

---

## What ComicBubble does

- **Detects speech bubbles / text regions**
  - Uses `src/layout.py` + `src/ocr.py` to find regions that look like dialogue or text.
  - Extracts raw Japanese text from each region with a Japanese OCR backend.

- **Translates JP → EN**
  - Character-level encoder–decoder model in PyTorch (`src/translate.py`).
  - Trained on a cleaned subset of JP–EN parallel data from OPUS/JESC + Tatoeba.
  - Model weights are in `models/translator.pt`.

- **Shows results region-by-region**
  - For each detected region: bounding box, original JP text, and EN translation.
  - Available via:
    - **CLI** (`src/cli.py`) – prints everything in the terminal.
    - **Web app** (`web_app.py`) – upload a page image and see it with translations side-by-side.

---

## How to run ComicBubble locally

### 1. Clone the repo

```bash
git clone https://github.com/rafiyamo/manga-ocr-translation.git
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
---

### 4. Run the web app

```bash
python web_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Workflow in the browser:

1. Use the upload form to select a manga page image.
2. The **page preview** appears on the left panel.
3. On the right, you’ll see each detected region listed with:

   * Region index
   * JP text
   * EN translation
   * Region metadata 

---

## Limitations and known issues

This project is a **demo / prototype**, not a production-grade translator.
Some important limitations:

### Language support

* **Only Japanese → English is implemented.**
  All data processing and model code assumes JP as source and EN as target.
  Extending to other language pairs would require:

  * new parallel corpora,
  * new vocabularies,
  * retrained models,
  * and changes in `translate.py` and the UI.

### OCR & layout

* **Vertical text and complex page layouts** are only partially handled.

  * Vertical manga bubbles sometimes get segmented into multiple lines in a way that
    does not perfectly match natural reading order.
  * SFX, handwritten notes, and overlapping bubbles can confuse the layout heuristics.
* **Region order is approximate.**

  * Regions are sorted to be “roughly readable”, but they may not follow strict
    right-to-left / top-to-bottom manga order for every page.

### Translation quality

* The model is a **small character-level encoder–decoder**:

  * It won’t match large commercial MT systems in fluency, context handling, or idioms.
* Training data is **general JP–EN text**, not manga-specific:

  * Slang, onomatopoeia, and stylised dialogue can be mistranslated or simplified.

---

## Possible next steps / future work

* Adding **subword tokenisation** (BPE/SentencePiece) and moving away from pure character models.
* Collecting and training on **manga-style dialogue corpora** to better capture tone.
* Improving vertical-text handling and region ordering heuristics.
* Adding an overlay that draws translated text directly on the uploaded page.
* Extending to **other language pairs** (e.g., JP→FR, JP→ES) by swapping in new corpora and models.

---

## Tech stack

* **Python 3**
* **PyTorch** – translation model implementation
* **Flask** – minimal web interface
* **OCR backend** – Japanese OCR library 
* **Google Colab** – used for training runs

---

## License

MIT License

Copyright (c) 2026 Rafiya Moeez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
