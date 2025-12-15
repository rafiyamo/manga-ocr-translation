"""
Command-line interface entry point.
"""

import argparse

from .pipeline import process_page


def main() -> None:
    parser = argparse.ArgumentParser(description="Manga OCR + translation CLI")
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

    jp_lines, en_lines = process_page(
        image_path=args.image_path,
        use_model=not args.no_model,
    )

    if not jp_lines:
        print("No text detected.")
        return

    print("Detected text lines (JP / raw OCR):")
    for i, line in enumerate(jp_lines, start=1):
        print(f"[{i}] {line}")

    print("\nEnglish translation:")
    for i, line in enumerate(en_lines, start=1):
        print(f"[{i}] {line}")


if __name__ == "__main__":
    main()
