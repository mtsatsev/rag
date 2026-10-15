import tempfile
import unittest
from pathlib import Path

from rag.index.store import ChunkStore
from rag.models import (
    Chunk,
    ChunkMetadata,
    Fund,
    FundQuery,
    FundRef,
    FundRegistry,
    Umbrella,
)
from rag.retrieve.filters import to_where
from rag.retrieve.hybrid import HybridRetriever, build_retriever
from tests.fakes import FakeDense, FakeEmbedder, FakeKeyword, FakeResolver

ENHANCED = FundRef(
    fund="Enhanced Intelligence Global Allocation Responsible Investing",
    umbrella="Optimum Fund",
)


def make_chunk(
    chunk_id: str,
    text: str = "Текст.",
    fund: str | None = ENHANCED.fund,
    umbrella: str | None = "Optimum Fund",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            doc_id=chunk_id,
            file_name=f"{chunk_id}.pdf",
            content_hash="v1",
            doc_type="kid",
            fund=fund,
            umbrella_fund=umbrella,
            section="",
            page_start=1,
            page_end=1,
            chunk_index=0,
        ),
    )


A, B, C, D = (make_chunk(name) for name in "ABCD")


def retrieve(
    dense: list[Chunk],
    keyword: list[Chunk],
    k: int = 6,
    fund_query: FundQuery = FundQuery(),
):
    retriever = HybridRetriever(
        FakeDense(dense), FakeKeyword(keyword), FakeResolver(fund_query)
    )
    return retriever.retrieve("въпрос", k=k)


def ids(result) -> list[str]:
    return [retrieved.chunk.chunk_id for retrieved in result.chunks]


class TestHybridRetrieverFusion(unittest.TestCase):
    def test_retrieve_when_both_searches_rank_the_same_chunk_first_then_it_comes_first(
        self,
    ):
        self.assertEqual(ids(retrieve(dense=[A, B], keyword=[A, C]))[0], "A")

    def test_retrieve_when_a_chunk_is_found_by_only_one_search_then_it_is_still_returned(
        self,
    ):
        self.assertEqual(set(ids(retrieve(dense=[A, B], keyword=[C]))), {"A", "B", "C"})

    def test_retrieve_when_a_chunk_is_second_in_both_searches_then_it_beats_a_chunk_first_in_only_one(
        self,
    ):
        self.assertEqual(ids(retrieve(dense=[B, A], keyword=[C, A]))[0], "A")

    def test_retrieve_when_chunks_are_returned_then_their_rank_in_each_search_is_recorded(
        self,
    ):
        result = retrieve(dense=[A, B], keyword=[B])
        ranks = {
            retrieved.chunk.chunk_id: (retrieved.dense_rank, retrieved.keyword_rank)
            for retrieved in result.chunks
        }

        self.assertEqual(ranks, {"A": (1, None), "B": (2, 1)})

    def test_retrieve_when_more_chunks_are_found_than_k_then_returns_at_most_k(self):
        self.assertEqual(len(retrieve(dense=[A, B, C], keyword=[D], k=2).chunks), 2)

    def test_retrieve_when_chunks_are_returned_then_scores_are_sorted_from_best_to_worst(
        self,
    ):
        scores = [
            retrieved.score
            for retrieved in retrieve(dense=[A, B, C], keyword=[C, D]).chunks
        ]

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_retrieve_when_neither_search_finds_anything_then_returns_no_chunks(self):
        self.assertEqual(retrieve(dense=[], keyword=[]).chunks, [])


class TestHybridRetrieverWiring(unittest.TestCase):
    def setUp(self):
        self.fund_query = FundQuery(funds=[ENHANCED])
        self.dense, self.keyword = FakeDense([A]), FakeKeyword([A])
        self.retriever = HybridRetriever(
            self.dense, self.keyword, FakeResolver(self.fund_query), candidates=15
        )

    def test_retrieve_when_a_fund_is_resolved_then_dense_search_gets_the_matching_chroma_filter(
        self,
    ):
        self.retriever.retrieve("въпрос")

        self.assertEqual(self.dense.calls[0]["where"], to_where(self.fund_query))

    def test_retrieve_when_a_fund_is_resolved_then_keyword_search_gets_the_fund_query(
        self,
    ):
        self.retriever.retrieve("въпрос")

        self.assertEqual(self.keyword.calls[0]["fund_query"], self.fund_query)

    def test_retrieve_when_candidates_is_set_then_both_searches_return_that_many_candidates(
        self,
    ):
        self.retriever.retrieve("въпрос")

        self.assertEqual(
            (self.dense.calls[0]["k"], self.keyword.calls[0]["k"]), (15, 15)
        )

    def test_retrieve_when_called_then_the_result_carries_the_question_and_the_fund_query(
        self,
    ):
        result = self.retriever.retrieve("въпрос")

        self.assertEqual(
            (result.question, result.fund_query), ("въпрос", self.fund_query)
        )


class TestBuildRetriever(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = ChunkStore(Path(self.directory.name), FakeEmbedder())
        self.store.replace_document(
            [make_chunk("enhanced", "Входни такси до 3% от сумата.")]
        )
        self.store.replace_document(
            [
                make_chunk(
                    "lookback", "Входни такси до 3% от сумата.", fund="ČSOB Lookback 3"
                )
            ]
        )
        self.store.replace_document(
            [
                make_chunk(
                    "rights",
                    "Всеки инвеститор може да прехвърли дяловете си.",
                    None,
                    None,
                )
            ]
        )
        registry = FundRegistry(
            umbrellas=[Umbrella(name="Optimum Fund")],
            funds=[Fund(name=ENHANCED.fund, umbrella="Optimum Fund")],
            documents={},
        )
        self.retriever = build_retriever(self.store, registry)

    def tearDown(self):
        self.directory.cleanup()

    def test_build_retriever_when_question_names_a_fund_then_other_funds_identical_chunks_are_not_returned(
        self,
    ):
        result = self.retriever.retrieve(f"Какви са входните такси на {ENHANCED.fund}?")

        self.assertNotIn("lookback", ids(result))
        self.assertIn("enhanced", ids(result))

    def test_build_retriever_when_question_names_a_fund_known_only_from_the_index_then_it_is_resolved(
        self,
    ):
        result = self.retriever.retrieve("Какви са входните такси на ČSOB Lookback 3?")

        self.assertEqual(
            result.fund_query.funds,
            [FundRef(fund="ČSOB Lookback 3", umbrella="Optimum Fund")],
        )
