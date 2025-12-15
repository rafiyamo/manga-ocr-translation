"""
Command-line interface entry point.
"""

import argparse

from .pipeline import process_page


def main():
    parser = argparse.ArgumentParser(
        description="Run the manga OCR + translation pipeline (OCR + placeholder translation)."
    )
    parser.add_argument(
        "--image_path",
        required=True,
        help="Path to the input image file (PNG/JPG).",
    )

    args = parser.parse_args()

    result = process_page(args.image_path)
    print(result)


if __name__ == "__main__":
    main()
