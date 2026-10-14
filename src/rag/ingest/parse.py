from pathlib import Path

from rag.models import ParsedDocument


def parse_pdf(pdf_path: Path, pages: list[int] | None = None) -> ParsedDocument:
    raise NotImplementedError
