from rag.index.store import ChunkStore
from rag.models import Chunk, FundRegistry, RetrievalResult, RetrievedChunk
from rag.retrieve.filters import to_where
from rag.retrieve.keyword import KeywordIndex
from rag.retrieve.resolver import FundResolver


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        keyword_index: KeywordIndex,
        resolver: FundResolver,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.keyword_index = keyword_index
        self.resolver = resolver
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int = 6) -> RetrievalResult:
        raise NotImplementedError("This function is not implemented yet.")


def build_retriever(store: ChunkStore, registry: FundRegistry) -> HybridRetriever:
    raise NotImplementedError("This function is not implemented yet.")
