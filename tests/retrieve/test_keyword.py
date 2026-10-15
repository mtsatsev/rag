import unittest
from pathlib import Path

from rag.index.store import ChunkStore
from rag.models import Chunk, ChunkMetadata, FundQuery, FundRef
from rag.retrieve.filters import matches
from rag.retrieve.keyword import KeywordIndex
from tests.fakes import FakeEmbedder

ENHANCED = "Enhanced Intelligence Global Allocation Responsible Investing"
CONSERVATIVE = "ОББ ЕкспертИйз Дефанзивен Консервативен Отговорно Инвестиращ"
CHROMA_DIR = Path(__file__).parents[2] / ".chroma"


def make_chunk(
    chunk_id: str, text: str, fund: str | None, umbrella: str | None = "Optimum Fund"
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


CHUNKS = [
    make_chunk(
        "fees", "Входни такси до 3% от сумата при записване във фонда.", ENHANCED
    ),
    make_chunk(
        "holding", "Препоръчителен период на държане: 5 години за фонда.", ENHANCED
    ),
    make_chunk(
        "risk", "Рисковият профил на фонда е 4 от 7. ISIN: BE6316179249.", ENHANCED
    ),
    make_chunk(
        "lookback_fees",
        "Входни такси до 3% от сумата при записване във фонда.",
        "ČSOB Lookback 3",
    ),
    make_chunk(
        "rights",
        "Всеки инвеститор може да прехвърли дяловете си във фонда.",
        None,
        umbrella=None,
    ),
]
ENHANCED_QUERY = FundQuery(funds=[FundRef(fund=ENHANCED, umbrella="Optimum Fund")])


def ids(results: list[tuple[Chunk, float]]) -> list[str]:
    return [chunk.chunk_id for chunk, _ in results]


class TestKeywordIndexSearch(unittest.TestCase):
    def setUp(self):
        self.index = KeywordIndex(CHUNKS)

    def test_search_when_question_contains_a_chunks_keywords_then_that_chunk_ranks_first(
        self,
    ):
        self.assertEqual(
            ids(self.index.search("Какъв е препоръчителният период на държане?"))[0],
            "holding",
        )

    def test_search_when_question_is_in_uppercase_then_results_are_the_same_as_in_lowercase(
        self,
    ):
        self.assertEqual(
            ids(self.index.search("ВХОДНИ ТАКСИ")),
            ids(self.index.search("входни такси")),
        )

    def test_search_when_question_contains_a_latin_lookalike_then_results_are_the_same_as_without_it(
        self,
    ):
        self.assertEqual(
            ids(self.index.search("Bходни такси")),
            ids(self.index.search("Входни такси")),
        )  # Latin B

    def test_search_when_question_contains_an_isin_then_the_chunk_with_that_isin_ranks_first(
        self,
    ):
        self.assertEqual(
            ids(self.index.search("What is the risk of BE6316179249?"))[0], "risk"
        )

    def test_search_when_no_word_of_the_question_occurs_then_returns_no_results(self):
        self.assertEqual(self.index.search("dividend yield"), [])

    def test_search_when_more_chunks_match_than_k_then_returns_at_most_k_results(self):
        self.assertEqual(len(self.index.search("фонда", k=2)), 2)

    def test_search_when_several_chunks_match_then_scores_are_sorted_from_best_to_worst(
        self,
    ):
        scores = [score for _, score in self.index.search("входни такси фонда")]

        self.assertEqual(scores, sorted(scores, reverse=True))


class TestKeywordIndexFundFilter(unittest.TestCase):
    def setUp(self):
        self.index = KeywordIndex(CHUNKS)

    def test_search_when_another_funds_chunk_matches_perfectly_then_it_is_excluded_by_the_fund_filter(
        self,
    ):
        self.assertNotIn(
            "lookback_fees",
            ids(self.index.search("Входни такси до 3%", fund_query=ENHANCED_QUERY)),
        )

    def test_search_when_filtered_by_fund_then_that_funds_matching_chunk_is_still_returned(
        self,
    ):
        self.assertIn(
            "fees",
            ids(self.index.search("Входни такси до 3%", fund_query=ENHANCED_QUERY)),
        )

    def test_search_when_filtered_by_fund_then_manager_wide_chunks_are_still_returned(
        self,
    ):
        self.assertIn(
            "rights",
            ids(self.index.search("прехвърли дяловете", fund_query=ENHANCED_QUERY)),
        )

    def test_search_when_no_fund_query_is_given_then_chunks_of_every_fund_can_be_returned(
        self,
    ):
        self.assertEqual(
            set(ids(self.index.search("Входни такси до 3%"))), {"fees", "lookback_fees"}
        )


class TestKeywordIndexRealIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHROMA_DIR.exists():
            raise unittest.SkipTest(f"{CHROMA_DIR} not found; run ingestion first")
        cls.chunks = ChunkStore(CHROMA_DIR, FakeEmbedder()).all_chunks()
        if not cls.chunks:
            raise unittest.SkipTest(f"{CHROMA_DIR} is empty; run ingestion first")
        cls.index = KeywordIndex(cls.chunks)

    def test_search_when_question_contains_the_enhanced_isin_then_the_top_chunk_belongs_to_that_fund(
        self,
    ):
        top_chunk, _ = self.index.search("BE6316179249")[0]

        self.assertEqual(top_chunk.metadata.fund, ENHANCED)

    def test_search_when_question_contains_the_balanced_isin_then_the_top_chunk_is_from_its_fact_sheet(
        self,
    ):
        top_chunk, _ = self.index.search("BG9000005223")[0]

        self.assertEqual(top_chunk.metadata.doc_id, "18368")

    def test_search_when_filtered_by_fund_then_every_result_passes_the_fund_filter(
        self,
    ):
        fund_query = FundQuery(
            funds=[FundRef(fund=CONSERVATIVE, umbrella="ОББ ЕкспертИйз")]
        )

        for chunk, _ in self.index.search(
            "устойчиви инвестиции", fund_query=fund_query
        ):
            with self.subTest(chunk=chunk.chunk_id):
                self.assertTrue(matches(chunk.metadata, fund_query))
