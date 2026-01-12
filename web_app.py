# web_app.py

from pathlib import Path
import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from src.pipeline import process_page

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    """Serve uploaded images from the uploads directory."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return redirect(url_for("index"))

        # Save the uploaded file
        save_path = UPLOAD_DIR / file.filename
        file.save(save_path)

        # Run ComicBubble pipeline
        regions, en_lines = process_page(str(save_path), use_model=True)

        bubbles = []
        for r, en in zip(regions, en_lines):
            bubbles.append(
                {
                    "jp": r.text,
                    "en": en,
                    "kind": r.kind,
                    "bbox": r.bbox,  # (x, y, w, h)
                }
            )

        # Build a URL that the browser can actually fetch
        image_url = url_for("uploaded_file", filename=file.filename)

        return render_template(
            "index.html",
            image_path=image_url,
            bubbles=bubbles,
        )

    # GET
    return render_template("index.html", image_path=None, bubbles=None)


if __name__ == "__main__":
    app.run(debug=True)
