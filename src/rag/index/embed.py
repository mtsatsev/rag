from typing import Protocol

import numpy as np

BGE_M3 = "BAAI/bge-m3"


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class BgeM3Embedder:
    def __init__(self, model_name: str = BGE_M3, max_seq_length: int = 1024) -> None:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError
