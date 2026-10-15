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
from rag.retrieve.dense import DenseRetriever, build_dense_retriever
from rag.retrieve.filters import to_where
from tests.fakes import FakeDense, FakeEmbedder, FakeResolver

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


A, B, C = (make_chunk(name) for name in "ABC")


def ids(result) -> list[str]:
    return [retrieved.chunk.chunk_id for retrieved in result.chunks]


class TestDenseRetriever(unittest.TestCase):
    def setUp(self):
        self.fund_query = FundQuery(funds=[ENHANCED])
        self.store = FakeDense([A, B, C])
        self.retriever = DenseRetriever(self.store, FakeResolver(self.fund_query))

    def test_retrieve_when_chunks_are_found_then_they_keep_the_store_order(self):
        self.assertEqual(ids(self.retriever.retrieve("въпрос")), ["A", "B", "C"])

    def test_retrieve_when_chunks_are_found_then_dense_rank_is_set_and_keyword_rank_is_none(
        self,
    ):
        ranks = [
            (r.dense_rank, r.keyword_rank)
            for r in self.retriever.retrieve("въпрос").chunks
        ]

        self.assertEqual(ranks, [(1, None), (2, None), (3, None)])

    def test_retrieve_when_chunks_are_found_then_the_score_is_the_store_similarity(
        self,
    ):
        scores = [r.score for r in self.retriever.retrieve("въпрос").chunks]

        self.assertEqual(
            scores, [score for _, score in self.store.query("въпрос", k=3)]
        )

    def test_retrieve_when_k_is_given_then_it_is_passed_to_the_store_and_at_most_k_are_returned(
        self,
    ):
        result = self.retriever.retrieve("въпрос", k=2)

        self.assertEqual((self.store.calls[0]["k"], len(result.chunks)), (2, 2))

    def test_retrieve_when_a_fund_is_resolved_then_the_store_gets_the_matching_chroma_filter(
        self,
    ):
        self.retriever.retrieve("въпрос")

        self.assertEqual(self.store.calls[0]["where"], to_where(self.fund_query))

    def test_retrieve_when_called_then_the_result_carries_the_question_and_the_fund_query(
        self,
    ):
        result = self.retriever.retrieve("въпрос")

        self.assertEqual(
            (result.question, result.fund_query), ("въпрос", self.fund_query)
        )

    def test_retrieve_when_nothing_is_found_then_returns_no_chunks(self):
        retriever = DenseRetriever(FakeDense([]), FakeResolver())

        self.assertEqual(retriever.retrieve("въпрос").chunks, [])


class TestBuildDenseRetriever(unittest.TestCase):
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
        self.retriever = build_dense_retriever(self.store, registry)

    def tearDown(self):
        self.directory.cleanup()

    def test_build_dense_retriever_when_question_names_a_fund_then_other_funds_identical_chunks_are_not_returned(
        self,
    ):
        result = self.retriever.retrieve(f"Какви са входните такси на {ENHANCED.fund}?")

        self.assertNotIn("lookback", ids(result))
        self.assertIn("enhanced", ids(result))

    def test_build_dense_retriever_when_question_names_a_fund_known_only_from_the_index_then_it_is_resolved(
        self,
    ):
        result = self.retriever.retrieve("Какви са входните такси на ČSOB Lookback 3?")

        self.assertEqual(
            result.fund_query.funds,
            [FundRef(fund="ČSOB Lookback 3", umbrella="Optimum Fund")],
        )
