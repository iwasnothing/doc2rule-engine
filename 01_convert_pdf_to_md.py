"""
Convert PDF files to Markdown using Microsoft's markitdown library.

Usage:
    python convert_pdf_to_md.py <input_pdf> [output_md]

If output_md is not specified, it defaults to the same name as the input
with a .md extension.

Requirements:
    pip install 'markitdown[all]'
"""

import sys
from pathlib import Path

from markitdown import MarkItDown


def convert_pdf_to_markdown(input_pdf: str, output_md: str | None = None) -> str:
    """Convert a PDF file to Markdown and save the result.

    Args:
        input_pdf: Path to the input PDF file.
        output_md: Path to the output Markdown file. If None, uses the
                   input filename with a .md extension.

    Returns:
        The path to the generated Markdown file.

    Raises:
        FileNotFoundError: If the input PDF does not exist.
        markitdown.UnsupportedFormatException: If the file format is not supported.
        markitdown.FileConversionException: If conversion fails.
    """
    input_path = Path(input_pdf)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_pdf}")

    if output_md is None:
        output_md = str(input_path.with_suffix(".md"))

    md = MarkItDown()
    result = md.convert(str(input_path))

    Path(output_md).write_text(result.text_content, encoding="utf-8")
    return output_md


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_md = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        saved_path = convert_pdf_to_markdown(input_pdf, output_md)
        print(f"Converted '{input_pdf}' -> '{saved_path}'")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
