from typing import Protocol

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BGE_M3 = "BAAI/bge-m3"


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class BgeM3Embedder:
    def __init__(
        self, model_name: str = BGE_M3, max_seq_length: int = 1024, batch_size: int = 16
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        if device == "cuda":
            self.model.half()
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
