import hashlib
from pathlib import Path

import pymupdf4llm

from rag.models import ParsedDocument, ParsedPage


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_pdf(pdf_path: Path, pages: list[int] | None = None) -> ParsedDocument:
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    page_chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        pages=[number - 1 for number in pages] if pages else None,
        page_chunks=True,
        use_ocr=False,
        header=False,
        footer=False,
    )
    return ParsedDocument(
        doc_id=pdf_path.stem,
        file_name=pdf_path.name,
        content_hash=file_hash(pdf_path),
        pages=[
            ParsedPage(
                page_number=chunk["metadata"]["page_number"], markdown=chunk["text"]
            )
            for chunk in page_chunks
        ],
    )
