from pathlib import Path
from typing import Any

from rag.index.embed import Embedder
from rag.models import Chunk


class ChunkStore:
    def __init__(
        self, path: Path, embedder: Embedder, collection: str = "chunks"
    ) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def has_document(self, doc_id: str, content_hash: str) -> bool:
        raise NotImplementedError

    def replace_document(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def delete_document(self, doc_id: str) -> None:
        raise NotImplementedError

    def all_chunks(self) -> list[Chunk]:
        raise NotImplementedError

    def query(
        self, text: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[tuple[Chunk, float]]:
        raise NotImplementedError
