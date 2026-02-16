"""
Convert Markdown files to PDF using markdown and xhtml2pdf.

Usage:
    python convert_md_to_pdf.py <input_md> [output_pdf]

If output_pdf is not specified, it defaults to the same name as the input
with a .pdf extension.

Requirements:
    pip install markdown xhtml2pdf
"""

import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

# Default CSS for a clean, readable PDF
DEFAULT_CSS = """
@page {
    size: A4;
    margin: 2cm;
}

body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
}

h1 {
    font-size: 22pt;
    border-bottom: 2px solid #ddd;
    padding-bottom: 6pt;
    margin-top: 24pt;
}

h2 {
    font-size: 18pt;
    border-bottom: 1px solid #eee;
    padding-bottom: 4pt;
    margin-top: 20pt;
}

h3 {
    font-size: 14pt;
    margin-top: 16pt;
}

code {
    font-family: Courier, monospace;
    background-color: #f4f4f4;
    padding: 2px 4px;
    font-size: 10pt;
}

pre {
    background-color: #f4f4f4;
    padding: 12px;
    font-size: 10pt;
    line-height: 1.4;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12pt 0;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}

th {
    background-color: #f4f4f4;
    font-weight: bold;
}

blockquote {
    border-left: 4px solid #ddd;
    padding-left: 16px;
    margin-left: 0;
    color: #666;
}

a {
    color: #0066cc;
    text-decoration: none;
}

img {
    max-width: 100%;
}
"""

# Markdown extensions to enable for richer conversion
MD_EXTENSIONS = [
    "tables",
    "fenced_code",
    "codehilite",
    "toc",
    "nl2br",
    "sane_lists",
]


def convert_markdown_to_pdf(
    input_md: str,
    output_pdf: str | None = None,
    css: str | None = None,
) -> str:
    """Convert a Markdown file to PDF and save the result.

    Args:
        input_md: Path to the input Markdown file.
        output_pdf: Path to the output PDF file. If None, uses the
                    input filename with a .pdf extension.
        css: Optional custom CSS string. If None, uses DEFAULT_CSS.

    Returns:
        The path to the generated PDF file.

    Raises:
        FileNotFoundError: If the input Markdown file does not exist.
        RuntimeError: If PDF conversion fails.
    """
    input_path = Path(input_md)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_md}")

    if output_pdf is None:
        output_pdf = str(input_path.with_suffix(".pdf"))

    if css is None:
        css = DEFAULT_CSS

    # Read the Markdown source
    md_text = input_path.read_text(encoding="utf-8")

    # Convert Markdown to HTML
    html_body = markdown.markdown(md_text, extensions=MD_EXTENSIONS)

    # Wrap in a full HTML document with styling
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Render HTML to PDF
    with open(output_pdf, "wb") as pdf_file:
        status = pisa.CreatePDF(html_doc, dest=pdf_file)

    if status.err:
        raise RuntimeError(
            f"PDF conversion failed with {status.err} error(s)"
        )

    return output_pdf


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    input_md = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        saved_path = convert_markdown_to_pdf(input_md, output_pdf)
        print(f"Converted '{input_md}' -> '{saved_path}'")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
