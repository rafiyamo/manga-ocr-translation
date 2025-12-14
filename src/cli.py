"""
Command-line interface entry point.
"""

import argparse
from .pipeline import process_page


def main():
    parser = argparse.ArgumentParser(
        description="Run the manga OCR + translation pipeline (placeholder version)."
    )
    parser.add_argument(
        "--image_path",
        required=False,
        help="Path to input image (currently unused in placeholder).",
    )
    args = parser.parse_args()

    result = process_page(args.image_path or "")
    print(result)


if __name__ == "__main__":
    main()