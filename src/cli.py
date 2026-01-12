# src/cli.py

"""
Command-line interface entry point for the Manga OCR + translation demo.
"""

import argparse
from textwrap import shorten

from .pipeline import process_page


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manga OCR + JP→EN translation CLI"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input manga page image",
    )
    parser.add_argument(
        "--no_model",
        action="store_true",
        help="Use placeholder translation instead of the trained model.",
    )

    args = parser.parse_args()
    use_model = not args.no_model

    regions, en_lines = process_page(
        image_path=args.image_path,
        use_model=use_model,
    )

    if not regions:
        print("No text detected.")
        return

    print("=== RAW OCR (JP) BY REGION ===")
    for i, r in enumerate(regions, start=1):
        x, y, w, h = r.bbox
        jp_preview = shorten(r.text.replace("\n", " "), width=40, placeholder="…")
        print(
            f"[{i:02}] kind={r.kind:<14} "
            f"bbox=(x={x:3}, y={y:3}, w={w:3}, h={h:3}) | {jp_preview}"
        )

    print("\n=== JP → EN TRANSLATIONS ===")
    for i, (r, en) in enumerate(zip(regions, en_lines), start=1):
        print(f"[{i:02}] JP: {r.text}")
        print(f"     EN: {en}")
        print("-" * 60)


if __name__ == "__main__":
    main()
