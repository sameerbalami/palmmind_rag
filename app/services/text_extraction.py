import io
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})


class UnsupportedFileError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    """Return the plain text of a .pdf or .txt file."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return _extract_pdf(data)
    if extension == ".txt":
        return data.decode("utf-8", errors="replace")
    raise UnsupportedFileError(f"Unsupported file type '{extension}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}")


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = (page.extract_text() or "" for page in reader.pages)
    return "\n\n".join(page.strip() for page in pages if page.strip())
