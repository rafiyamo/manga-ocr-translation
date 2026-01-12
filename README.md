````markdown
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

---

## Repository layout (high-level)

```text
.
├── data/
│   ├── parallel/                  # train/dev/test TSVs (large corpora not all committed)
│   └── samples/                   # example page images
├── models/
│   └── translator.pt              # trained JP→EN model weights (15-epoch run)
├── notebooks/
│   └── train_translation_model.ipynb
├── scripts/
│   └── build_parallel_corpus.py   # combine & clean raw corpora into TSVs
├── src/
│   ├── cli.py                     # CLI entry point
│   ├── layout.py                  # text region detection
│   ├── ocr.py                     # JP OCR wrapper
│   ├── pipeline.py                # end-to-end “page → regions → translations”
│   └── translate.py               # model loading + greedy decoding helpers
├── web_app.py                     # Flask app (simple upload + results UI)
└── README.md
````

---

## How to run ComicBubble locally

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/manga-ocr-translation.git
cd manga-ocr-translation
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

> `models/translator.pt` should already exist in the repo.
> If you train a new model, just overwrite that file.

---

### 4. Run the CLI

Basic usage:

```bash
python -m src.cli --image_path data/samples/manga_page_1.png
```

What you’ll see (shape of output):

```text
Detected text regions (JP / raw OCR):
[0] kind=dialogue bbox=(100, 100, 300, 40) | 抱三
[1] kind=dialogue bbox=(100, 140, 300, 40) | け日
...

English translation (aligned with regions):
[0] JP: 抱三
    EN: right.
--------------------------------------------------
[1] JP: け日
    EN: i'm sorry.
...
```

If you want to **disable the learned model** and just check the OCR wiring /
placeholder translations, you can use:

```bash
python -m src.cli --image_path data/samples/manga_page_1.png --no_model
```

That forces the pipeline to use a dead-simple placeholder translator defined in `src/translate.py`.

---

### 5. Run the web app

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
   * Region metadata (kind + bounding box)

This is the version that’s easiest to show on a portfolio / demo.

---

## Training or improving the translator (optional)

This is only needed if you want to retrain or experiment with the model.

### Step 1 – Build train/dev/test TSVs

Place your raw corpora under:

```text
data/parallel/raw_corpora/
```

(See comments in `scripts/build_parallel_corpus.py` for expected filenames and formats.)

Then run:

```bash
python scripts/build_parallel_corpus.py
```

This script:

* Reads the raw JP–EN corpora (OPUS/JESC + Tatoeba format).
* Cleans and deduplicates sentence pairs.
* Writes:

```text
data/parallel/jp_en_train.tsv
data/parallel/jp_en_dev.tsv
data/parallel/jp_en_test.tsv
```

Each TSV has the header:

```text
src<TAB>tgt
```

where `src` is JP and `tgt` is EN.

### Step 2 – Train in Colab (recommended)

1. Open `notebooks/train_translation_model.ipynb` in Google Colab.
2. Clone this repo and copy the TSVs into the expected `data/parallel/` path (the notebook has the exact cells).
3. Run the notebook cells:

   * load data,
   * build vocab,
   * train the encoder–decoder for N epochs (e.g., 10–15).
4. At the end, download the resulting `translator.pt`.

Replace the local `models/translator.pt` with this new file.
The CLI and web app will automatically start using your new model.

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
  * and some plumbing changes in `translate.py` and the UI.

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

  * Good enough for a demo and some short, simple lines.
  * It won’t match large commercial MT systems in fluency, context handling, or idioms.
* Training data is **general JP–EN text**, not manga-specific:

  * Slang, onomatopoeia, and stylised dialogue can be mistranslated or simplified.
* Decoding is **greedy**:

  * No beam search, no length penalties, etc. – this keeps the code simple but
    sacrifices some translation quality.

---

## Possible next steps / future work

If you (or a future collaborator) want to extend ComicBubble, realistic next steps include:

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
* **OCR backend** – Japanese OCR library wrapped in `src/ocr.py`
* **Google Colab** – used for training runs

---

## License

MIT License

Copyright (c) 2026 <Your Name>

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

```
::contentReference[oaicite:0]{index=0}
```

