import numpy as np


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
