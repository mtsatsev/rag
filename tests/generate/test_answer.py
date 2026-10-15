import json
import unittest

from rag.generate.answer import Answerer
from rag.models import (
    Chunk,
    ChunkMetadata,
    FundQuery,
    FundRef,
    LLMAnswer,
    RetrievalResult,
    RetrievedChunk,
)

ENHANCED = "Enhanced Intelligence Global Allocation Responsible Investing"
KID = "KID_BE6316179249_BG"


def make_chunk(
    chunk_id: str,
    text: str,
    page: int = 2,
    section: str = "Какви са рисковете?",
    fund: str | None = ENHANCED,
    doc_id: str = KID,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            doc_id=doc_id,
            file_name=f"{doc_id}.pdf",
            content_hash="v1",
            doc_type="kid",
            fund=fund,
            umbrella_fund="Optimum Fund" if fund else None,
            section=section,
            page_start=page,
            page_end=page,
            chunk_index=0,
        ),
    )


RISK = make_chunk(
    "risk",
    "На този продукт отредихме категория на риск 3 от 7, което означава един средно нисък.",
)
HOLDING = make_chunk(
    "holding",
    "Препоръчителен период на държане: 5 години.",
    page=3,
    section="Препоръчителен период",
)
RIGHTS = make_chunk(
    "rights",
    "Дяловете на фондовете могат да се прехвърлят свободно.",
    page=1,
    section="",
    fund=None,
    doc_id="11074",
)
FUND_QUERY = FundQuery(funds=[FundRef(fund=ENHANCED, umbrella="Optimum Fund")])


class FakeRetriever:
    def __init__(self, chunks: list[Chunk], scores: list[float] | None = None) -> None:
        scores = scores or [0.8 - 0.05 * index for index in range(len(chunks))]
        self.result_chunks = [
            RetrievedChunk(chunk=chunk, score=score, dense_rank=rank, keyword_rank=None)
            for rank, (chunk, score) in enumerate(zip(chunks, scores), start=1)
        ]

    def retrieve(self, question: str, k: int = 5) -> RetrievalResult:
        return RetrievalResult(
            question=question, fund_query=FUND_QUERY, chunks=self.result_chunks[:k]
        )


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, []

    def complete(self, messages, schema):
        self.calls.append({"messages": messages, "schema": schema})
        return self.reply


def reply(
    status: str = "answered",
    answer: str = "Категория на риск 3 от 7.",
    citations=(),
    conflicts=(),
) -> str:
    return json.dumps(
        {
            "status": status,
            "answer": answer,
            "citations": [
                {"source": source, "quote": quote} for source, quote in citations
            ],
            "conflicts": list(conflicts),
        },
        ensure_ascii=False,
    )


def answer(
    question: str,
    chunks: list[Chunk],
    llm_reply: str,
    scores: list[float] | None = None,
):
    llm = FakeLLM(llm_reply)
    result = Answerer(FakeRetriever(chunks, scores), llm).answer(question)
    return result, llm


RISK_CITATION = ("S1", "категория на риск 3 от 7")


class TestAnswererRetrievalGate(unittest.TestCase):
    def test_answer_when_nothing_is_retrieved_then_status_is_not_found_and_the_llm_is_not_called(
        self,
    ):
        result, llm = answer("Каква е категорията на риск?", [], reply())

        self.assertEqual(result.status, "not_found")
        self.assertEqual(llm.calls, [])

    def test_answer_when_best_score_is_below_min_score_then_status_is_not_found_and_the_llm_is_not_called(
        self,
    ):
        result, llm = answer(
            "Каква е категорията на риск?", [RISK], reply(), scores=[0.1]
        )

        self.assertEqual(result.status, "not_found")
        self.assertEqual(llm.calls, [])

    def test_answer_when_nothing_relevant_is_found_then_there_are_no_sources_or_funds(
        self,
    ):
        result, _ = answer("Каква е категорията на риск?", [], reply())

        self.assertEqual((result.sources, result.funds), ([], []))

    def test_answer_when_bulgarian_question_is_not_found_then_the_message_is_in_bulgarian(
        self,
    ):
        result, _ = answer("Каква е категорията на риск?", [], reply())

        self.assertRegex(result.answer, "[а-яА-Я]")

    def test_answer_when_english_question_is_not_found_then_the_message_is_in_english(
        self,
    ):
        result, _ = answer("What is the risk class?", [], reply())

        self.assertNotRegex(result.answer, "[а-яА-Я]")


class TestAnswererLLMCall(unittest.TestCase):
    def test_answer_when_chunks_are_retrieved_then_the_llm_receives_them_as_numbered_sources(
        self,
    ):
        _, llm = answer(
            "Каква е категорията на риск?",
            [RISK, HOLDING],
            reply(citations=[RISK_CITATION]),
        )

        user_message = llm.calls[0]["messages"][-1]["content"]
        self.assertIn(f"[S1] {KID}.pdf", user_message)
        self.assertIn(HOLDING.text, user_message)

    def test_answer_when_the_llm_is_called_then_its_reply_is_constrained_to_the_llm_answer_schema(
        self,
    ):
        _, llm = answer(
            "Каква е категорията на риск?", [RISK], reply(citations=[RISK_CITATION])
        )

        self.assertEqual(llm.calls[0]["schema"], LLMAnswer.model_json_schema())

    def test_answer_when_the_llm_reply_is_not_valid_json_then_status_is_not_found(self):
        result, _ = answer("Каква е категорията на риск?", [RISK], "Категорията е 3.")

        self.assertEqual((result.status, result.sources), ("not_found", []))

    def test_answer_when_the_llm_answers_with_a_valid_citation_then_its_answer_and_status_are_returned(
        self,
    ):
        result, _ = answer(
            "Каква е категорията на риск?", [RISK], reply(citations=[RISK_CITATION])
        )

        self.assertEqual(
            (result.status, result.answer), ("answered", "Категория на риск 3 от 7.")
        )


class TestAnswererCitations(unittest.TestCase):
    def test_answer_when_a_citation_uses_an_unknown_source_label_then_it_is_dropped(
        self,
    ):
        result, _ = answer(
            "Въпрос?",
            [RISK, HOLDING],
            reply(citations=[RISK_CITATION, ("S9", "5 години")]),
        )

        self.assertEqual([source.page_start for source in result.sources], [2])

    def test_answer_when_a_quote_does_not_occur_in_its_source_then_the_citation_is_dropped(
        self,
    ):
        result, _ = answer(
            "Въпрос?",
            [RISK, HOLDING],
            reply(citations=[RISK_CITATION, ("S2", "10 години")]),
        )

        self.assertEqual([source.page_start for source in result.sources], [2])

    def test_answer_when_a_quote_differs_only_in_casing_and_spacing_then_the_citation_is_kept(
        self,
    ):
        result, _ = answer(
            "Въпрос?", [RISK], reply(citations=[("S1", "КАТЕГОРИЯ  на риск 3 от 7")])
        )

        self.assertEqual(len(result.sources), 1)

    def test_answer_when_answered_but_no_citation_is_valid_then_status_becomes_not_found_without_sources(
        self,
    ):
        result, _ = answer(
            "Въпрос?", [RISK], reply(citations=[("S1", "измислен цитат")])
        )

        self.assertEqual((result.status, result.sources), ("not_found", []))

    def test_answer_when_the_llm_says_not_found_then_there_are_no_sources_even_if_it_cites(
        self,
    ):
        result, _ = answer(
            "Въпрос?",
            [RISK],
            reply(
                status="not_found", answer="Няма информация.", citations=[RISK_CITATION]
            ),
        )

        self.assertEqual(
            (result.status, result.sources, result.funds), ("not_found", [], [])
        )


class TestAnswererSources(unittest.TestCase):
    def test_answer_when_a_chunk_is_cited_then_its_source_is_built_from_the_chunk_metadata(
        self,
    ):
        result, _ = answer("Въпрос?", [RISK], reply(citations=[RISK_CITATION]))
        source = result.sources[0]

        self.assertEqual(
            (
                source.document,
                source.page_start,
                source.page_end,
                source.section,
                source.fund,
            ),
            (f"{KID}.pdf", 2, 2, "Какви са рисковете?", ENHANCED),
        )

    def test_answer_when_the_same_source_is_cited_twice_then_it_appears_once(self):
        result, _ = answer(
            "Въпрос?", [RISK], reply(citations=[RISK_CITATION, ("S1", "средно нисък")])
        )

        self.assertEqual(len(result.sources), 1)

    def test_answer_when_a_chunk_is_retrieved_but_not_cited_then_it_is_not_a_source(
        self,
    ):
        result, _ = answer("Въпрос?", [RISK, HOLDING], reply(citations=[RISK_CITATION]))

        self.assertEqual([source.page_start for source in result.sources], [2])

    def test_answer_when_chunks_are_cited_then_funds_are_the_distinct_funds_of_the_cited_chunks(
        self,
    ):
        citations = [RISK_CITATION, ("S2", "5 години"), ("S3", "прехвърлят свободно")]

        result, _ = answer(
            "Въпрос?", [RISK, HOLDING, RIGHTS], reply(citations=citations)
        )

        self.assertEqual(result.funds, [ENHANCED])

    def test_answer_when_the_llm_reports_conflicts_then_they_are_passed_through(self):
        llm_reply = reply(
            citations=[RISK_CITATION],
            conflicts=["Документите посочват различни категории."],
        )

        result, _ = answer("Въпрос?", [RISK], llm_reply)

        self.assertEqual(result.conflicts, ["Документите посочват различни категории."])
