import numpy as np

from rag.models import Chunk, FundQuery


class FakeEmbedder:
    """Deterministic character-histogram vectors; records every text it embeds."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.embedded.extend(texts)
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vector = np.zeros(32)
        for character in text:
            vector[ord(character) % 32] += 1
        return vector / np.linalg.norm(vector)


class FakeDense:
    def __init__(self, ranked: list[Chunk]) -> None:
        self.ranked, self.calls = ranked, []

    def query(self, text, k=5, where=None):
        self.calls.append({"text": text, "k": k, "where": where})
        return [(chunk, 1.0 - rank / 100) for rank, chunk in enumerate(self.ranked[:k])]


class FakeKeyword:
    def __init__(self, ranked: list[Chunk]) -> None:
        self.ranked, self.calls = ranked, []

    def search(self, question, k=20, fund_query=None):
        self.calls.append({"question": question, "k": k, "fund_query": fund_query})
        return [(chunk, 10.0 - rank) for rank, chunk in enumerate(self.ranked[:k])]


class FakeResolver:
    def __init__(self, fund_query: FundQuery = FundQuery()) -> None:
        self.fund_query = fund_query

    def resolve(self, question):
        return self.fund_query
