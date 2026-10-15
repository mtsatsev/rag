import tempfile
import unittest
from pathlib import Path

from rag.index.store import ChunkStore
from rag.models import Chunk, ChunkMetadata, FundQuery, FundRef
from rag.retrieve.filters import matches, to_where
from tests.fakes import FakeEmbedder

ENHANCED = FundRef(
    fund="Enhanced Intelligence Global Allocation Responsible Investing",
    umbrella="Optimum Fund",
)
CONSERVATIVE = FundRef(
    fund="ОББ ЕкспертИйз Дефанзивен Консервативен Отговорно Инвестиращ",
    umbrella="ОББ ЕкспертИйз",
)
ALL = {
    "enhanced",
    "lookback",
    "optimum_general",
    "conservative",
    "obb_general",
    "rights",
}


def make_chunk(doc_id: str, fund: str | None, umbrella: str | None) -> Chunk:
    return Chunk(
        chunk_id=doc_id,
        text=f"Текст на {doc_id}.",
        metadata=ChunkMetadata(
            doc_id=doc_id,
            file_name=f"{doc_id}.pdf",
            content_hash="v1",
            doc_type="prospectus",
            fund=fund,
            umbrella_fund=umbrella,
            section="",
            page_start=1,
            page_end=1,
            chunk_index=0,
        ),
    )


CHUNKS = [
    make_chunk("enhanced", ENHANCED.fund, "Optimum Fund"),
    make_chunk("lookback", "ČSOB Lookback 3", "Optimum Fund"),
    make_chunk("optimum_general", None, "Optimum Fund"),
    make_chunk("conservative", CONSERVATIVE.fund, "ОББ ЕкспертИйз"),
    make_chunk("obb_general", None, "ОББ ЕкспертИйз"),
    make_chunk("rights", None, None),
]


class FilterCases:
    def allowed(self, fund_query: FundQuery) -> set[str]:
        raise NotImplementedError

    def test_filter_when_no_fund_or_umbrella_is_named_then_every_chunk_is_allowed(self):
        self.assertEqual(self.allowed(FundQuery()), ALL)

    def test_filter_when_one_fund_is_named_then_its_chunks_its_umbrella_pages_and_manager_wide_chunks_are_allowed(
        self,
    ):
        self.assertEqual(
            self.allowed(FundQuery(funds=[ENHANCED])),
            {"enhanced", "optimum_general", "rights"},
        )

    def test_filter_when_one_fund_is_named_then_other_sub_funds_of_the_same_umbrella_are_excluded(
        self,
    ):
        self.assertNotIn("lookback", self.allowed(FundQuery(funds=[ENHANCED])))

    def test_filter_when_one_fund_is_named_then_chunks_of_other_umbrellas_are_excluded(
        self,
    ):
        self.assertTrue(
            self.allowed(FundQuery(funds=[ENHANCED])).isdisjoint(
                {"conservative", "obb_general"}
            )
        )

    def test_filter_when_two_funds_of_different_umbrellas_are_named_then_both_with_their_umbrella_pages_are_allowed(
        self,
    ):
        allowed = self.allowed(FundQuery(funds=[ENHANCED, CONSERVATIVE]))

        self.assertEqual(
            allowed,
            {"enhanced", "optimum_general", "conservative", "obb_general", "rights"},
        )

    def test_filter_when_only_an_umbrella_is_named_then_all_its_chunks_and_manager_wide_chunks_are_allowed(
        self,
    ):
        allowed = self.allowed(FundQuery(umbrellas=["Optimum Fund"]))

        self.assertEqual(allowed, {"enhanced", "lookback", "optimum_general", "rights"})

    def test_filter_when_a_fund_and_another_umbrella_are_named_then_both_are_allowed(
        self,
    ):
        allowed = self.allowed(
            FundQuery(funds=[ENHANCED], umbrellas=["ОББ ЕкспертИйз"])
        )

        self.assertEqual(
            allowed,
            {"enhanced", "optimum_general", "conservative", "obb_general", "rights"},
        )


class TestMatches(FilterCases, unittest.TestCase):
    def allowed(self, fund_query: FundQuery) -> set[str]:
        return {
            chunk.chunk_id for chunk in CHUNKS if matches(chunk.metadata, fund_query)
        }


class TestToWhere(FilterCases, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.store = ChunkStore(Path(cls.directory.name), FakeEmbedder())
        for chunk in CHUNKS:
            cls.store.replace_document([chunk])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def allowed(self, fund_query: FundQuery) -> set[str]:
        results = self.store.query("Текст", k=len(CHUNKS), where=to_where(fund_query))
        return {chunk.chunk_id for chunk, _ in results}

    def test_to_where_when_no_fund_or_umbrella_is_named_then_returns_none(self):
        self.assertIsNone(to_where(FundQuery()))
