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
        fund_query = self.resolver.resolve(question)
        dense = self.store.query(
            question, k=self.candidates, where=to_where(fund_query)
        )
        keyword = self.keyword_index.search(
            question, k=self.candidates, fund_query=fund_query
        )

        chunks: dict[str, Chunk] = {}
        ranks: dict[str, dict[str, int]] = {}
        for search, results in (("dense", dense), ("keyword", keyword)):
            for rank, (chunk, _) in enumerate(results, start=1):
                chunks[chunk.chunk_id] = chunk
                ranks.setdefault(chunk.chunk_id, {})[search] = rank

        fused = [
            RetrievedChunk(
                chunk=chunks[chunk_id],
                score=sum(1 / (self.rrf_k + rank) for rank in chunk_ranks.values()),
                dense_rank=chunk_ranks.get("dense"),
                keyword_rank=chunk_ranks.get("keyword"),
            )
            for chunk_id, chunk_ranks in ranks.items()
        ]
        fused.sort(key=lambda retrieved: retrieved.score, reverse=True)
        return RetrievalResult(
            question=question, fund_query=fund_query, chunks=fused[:k]
        )


def build_retriever(store: ChunkStore, registry: FundRegistry) -> HybridRetriever:
    """Keyword index and fund resolver built from what is currently in the store."""
    return HybridRetriever(
        store,
        KeywordIndex(store.all_chunks()),
        FundResolver(registry, store.known_funds),
    )
