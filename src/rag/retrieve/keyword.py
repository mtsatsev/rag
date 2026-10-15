from rank_bm25 import BM25Okapi

from rag.ingest.clean import match_key
from rag.models import Chunk, FundQuery
from rag.retrieve.filters import matches


class KeywordIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def search(
        self, question: str, k: int = 20, fund_query: FundQuery | None = None
    ) -> list[tuple[Chunk, float]]:
        """Best BM25 matches (score > 0) among the chunks the fund filter allows."""
        fund_query = fund_query or FundQuery()
        scores = self.bm25.get_scores(tokenize(question))
        results = [
            (chunk, float(score))
            for chunk, score in zip(self.chunks, scores)
            if score > 0 and matches(chunk.metadata, fund_query)
        ]
        return sorted(results, key=lambda result: result[1], reverse=True)[:k]


def tokenize(text: str) -> list[str]:
    return match_key(text).split()
