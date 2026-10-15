from typing import Protocol

from rag.models import RetrievalResult


class Retriever(Protocol):
    def retrieve(self, question: str, k: int = 5) -> RetrievalResult: ...
