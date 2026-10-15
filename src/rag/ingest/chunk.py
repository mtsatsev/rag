from rag.models import Chunk, DocumentInfo, ParsedDocument


def chunk_document(
    doc: ParsedDocument, info: DocumentInfo, max_chars: int = 1500
) -> list[Chunk]:
    raise NotImplementedError
