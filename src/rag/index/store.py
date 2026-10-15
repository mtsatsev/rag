from pathlib import Path
from typing import Any

import chromadb

from rag.index.embed import Embedder
from rag.models import Chunk, ChunkMetadata

NULLABLE_FIELDS = ("fund", "umbrella_fund")  # Chroma metadata cannot hold None


def to_chroma_metadata(metadata: ChunkMetadata) -> dict[str, Any]:
    return {
        key: "" if value is None else value
        for key, value in metadata.model_dump().items()
    }


def to_chunk(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    restored = {
        key: None if key in NULLABLE_FIELDS and value == "" else value
        for key, value in metadata.items()
    }
    return Chunk(chunk_id=chunk_id, text=text, metadata=ChunkMetadata(**restored))


class ChunkStore:
    def __init__(
        self, path: Path, embedder: Embedder, collection: str = "chunks"
    ) -> None:
        self.embedder = embedder
        self.collection = chromadb.PersistentClient(
            path=str(path)
        ).get_or_create_collection(
            collection, configuration={"hnsw": {"space": "cosine"}}
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    @property
    def document_ids(self) -> set[str]:
        raise NotImplementedError

    def has_document(self, doc_id: str, content_hash: str) -> bool:
        where = {"$and": [{"doc_id": doc_id}, {"content_hash": content_hash}]}
        return bool(self.collection.get(where=where, limit=1)["ids"])

    def replace_document(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.delete_document(chunks[0].metadata.doc_id)
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=self.embedder.embed_documents(
                [chunk.embed_text for chunk in chunks]
            ),
            documents=[chunk.text for chunk in chunks],
            metadatas=[to_chroma_metadata(chunk.metadata) for chunk in chunks],
        )

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def all_chunks(self) -> list[Chunk]:
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            to_chunk(chunk_id, text, metadata)
            for chunk_id, text, metadata in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def query(
        self, text: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[tuple[Chunk, float]]:
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (to_chunk(chunk_id, text, metadata), 1.0 - distance)
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]
