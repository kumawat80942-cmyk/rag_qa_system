"""
document_loader.py

Loads source documents from disk into plain text, ready for chunking.
Supports .txt and .pdf files. Extend this module if you need to support
more formats (e.g. .docx) for the project.
"""

from pathlib import Path


def load_document(file_path: str) -> str:
    """Load a document from disk and return its plain-text content.

    Args:
        file_path: Path to a .txt or .pdf file.

    Returns:
        The extracted text as a single string.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the file extension is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_txt(path)
    elif suffix == ".pdf":
        return _load_pdf(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: .txt, .pdf"
        )


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required to read PDF files. Install it with "
            "`pip install pypdf`."
        ) from e

    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)
