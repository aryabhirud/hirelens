"""Resume PDF text extraction."""
from __future__ import annotations

import io
import re

from tools.errors import ToolError

MAX_PDF_BYTES = 5 * 1024 * 1024
MIN_TEXT_CHARS = 200  # less than this almost always means a scanned/image PDF


def extract_pdf_text(data: bytes) -> tuple[str, int]:
    """Return (text, page_count). Raises ToolError with a user-friendly reason."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    if not data:
        raise ToolError("The uploaded file is empty.")
    if len(data) > MAX_PDF_BYTES:
        raise ToolError("The PDF is larger than 5 MB. Please upload a smaller resume.")
    if not data.lstrip().startswith(b"%PDF"):
        raise ToolError("This file is not a PDF. Please upload your resume as a PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ToolError("The PDF is password-protected. Please upload an unlocked copy.") from exc
        pages = [page.extract_text() or "" for page in reader.pages]
    except ToolError:
        raise
    except (PdfReadError, Exception) as exc:
        raise ToolError(f"The PDF could not be read ({type(exc).__name__}). Is the file corrupted?") from exc

    text = normalise_text("\n".join(pages))
    if len(text) < MIN_TEXT_CHARS:
        raise ToolError(
            "Almost no text could be extracted. The resume looks like a scanned image; "
            "please export it as a text-based PDF."
        )
    return text, len(pages)


def normalise_text(text: str) -> str:
    text = text.replace(" ", " ").replace("•", "•")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
