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

    regions, en_lines = process_page(
        image_path=args.image_path,
        use_model=not args.no_model,
    )

    if not regions:
        print("No text detected.")
        return

    print("Detected text regions (JP / raw OCR):")
    for r in regions:
        print(f"[{r.reading_index}] kind={r.kind} bbox={r.bbox} | {r.text}")

    print("\nEnglish translation (aligned with regions):")
    for r, en in zip(regions, en_lines):
        print(f"[{r.reading_index}] {en}")


if __name__ == "__main__":
    main()
