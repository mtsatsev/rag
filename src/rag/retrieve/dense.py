from rag.index.store import ChunkStore
from rag.models import FundRegistry, RetrievalResult, RetrievedChunk
from rag.retrieve.filters import to_where
from rag.retrieve.resolver import FundResolver


class DenseRetriever:
    """Dense search restricted by the fund filter; scores are cosine similarities."""

    def __init__(self, store: ChunkStore, resolver: FundResolver) -> None:
        self.store = store
        self.resolver = resolver

    def retrieve(self, question: str, k: int = 5) -> RetrievalResult:
        fund_query = self.resolver.resolve(question)
        results = self.store.query(question, k=k, where=to_where(fund_query))
        return RetrievalResult(
            question=question,
            fund_query=fund_query,
            chunks=[
                RetrievedChunk(
                    chunk=chunk, score=score, dense_rank=rank, keyword_rank=None
                )
                for rank, (chunk, score) in enumerate(results, start=1)
            ],
        )


def build_dense_retriever(store: ChunkStore, registry: FundRegistry) -> DenseRetriever:
    """Fund resolver built from funds.yaml plus the funds currently in the store."""
    return DenseRetriever(store, FundResolver(registry, store.known_funds))
