import hashlib

from rag.ingest.clean import clean_markdown
from rag.models import Chunk, ChunkMetadata, DocumentInfo, ParsedDocument

PARAGRAPH_SEPARATOR = "\n\n"


def chunk_document(
    doc: ParsedDocument, info: DocumentInfo, max_chars: int = 1500
) -> list[Chunk]:
    paragraphs = [
        (page.page_number, paragraph)
        for page in doc.pages
        for paragraph in clean_markdown(page.markdown).split(PARAGRAPH_SEPARATOR)
        if paragraph
    ]

    groups: list[list[tuple[int, str]]] = []
    for page_number, paragraph in paragraphs:
        if (
            groups
            and group_length(groups[-1]) + len(PARAGRAPH_SEPARATOR) + len(paragraph)
            <= max_chars
        ):
            groups[-1].append((page_number, paragraph))
        else:
            groups.append([(page_number, paragraph)])

    return [to_chunk(doc, info, index, group) for index, group in enumerate(groups)]


def group_length(group: list[tuple[int, str]]) -> int:
    return len(PARAGRAPH_SEPARATOR.join(paragraph for _, paragraph in group))


def to_chunk(
    doc: ParsedDocument, info: DocumentInfo, index: int, group: list[tuple[int, str]]
) -> Chunk:
    return Chunk(
        chunk_id=hashlib.sha256(f"{doc.content_hash}:{index}".encode()).hexdigest(),
        text=PARAGRAPH_SEPARATOR.join(paragraph for _, paragraph in group),
        metadata=ChunkMetadata(
            doc_id=doc.doc_id,
            file_name=doc.file_name,
            content_hash=doc.content_hash,
            doc_type=info.doc_type,
            fund=info.fund,
            umbrella_fund=info.umbrella_fund,
            section="",
            page_start=group[0][0],
            page_end=group[-1][0],
            chunk_index=index,
        ),
    )
