import re

from pydantic import ValidationError

from rag.generate.llm import LLM
from rag.generate.prompt import build_messages, chunk_for_label
from rag.ingest.clean import match_key
from rag.models import Answer, Chunk, LLMAnswer, Source
from rag.retrieve.protocol import Retriever

CYRILLIC = re.compile(r"[а-яА-Я]")
NOT_FOUND_MESSAGE = {
    "bg": "Тази информация не е намерена в предоставените документи.",
    "en": "This information could not be found in the provided documents.",
}


class Answerer:
    def __init__(
        self, retriever: Retriever, llm: LLM, k: int = 5, min_score: float = 0.35
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.min_score = min_score

    def answer(self, question: str) -> Answer:
        result = self.retriever.retrieve(question, k=self.k)
        if not result.chunks or result.chunks[0].score < self.min_score:
            return not_found(question)

        chunks = [retrieved.chunk for retrieved in result.chunks]
        reply = self.llm.complete(
            build_messages(question, chunks, result.fund_query),
            LLMAnswer.model_json_schema(),
        )
        try:
            llm_answer = LLMAnswer.model_validate_json(reply)
        except ValidationError:
            return not_found(question)
        if llm_answer.status == "not_found":
            return Answer(
                answer=llm_answer.answer, status="not_found", funds=[], sources=[]
            )

        cited = cited_chunks(llm_answer, chunks)
        if llm_answer.status == "answered" and not cited:
            return not_found(question)
        return Answer(
            answer=llm_answer.answer,
            status=llm_answer.status,
            funds=list(
                dict.fromkeys(c.metadata.fund for c in cited if c.metadata.fund)
            ),
            sources=unique([to_source(chunk) for chunk in cited]),
            conflicts=llm_answer.conflicts,
        )


def cited_chunks(llm_answer: LLMAnswer, chunks: list[Chunk]) -> list[Chunk]:
    """Chunks behind valid citations: known label and a quote that occurs in that chunk."""
    cited: list[Chunk] = []
    for citation in llm_answer.citations:
        chunk = chunk_for_label(citation.source, chunks)
        quote = match_key(citation.quote)
        if chunk and quote and quote in match_key(chunk.text) and chunk not in cited:
            cited.append(chunk)
    return cited


def unique(sources: list[Source]) -> list[Source]:
    return [
        source for index, source in enumerate(sources) if source not in sources[:index]
    ]


def to_source(chunk: Chunk) -> Source:
    metadata = chunk.metadata
    return Source(
        document=metadata.file_name,
        page_start=metadata.page_start,
        page_end=metadata.page_end,
        section=metadata.section,
        fund=metadata.fund,
    )


def not_found(question: str) -> Answer:
    language = "bg" if CYRILLIC.search(question) else "en"
    return Answer(
        answer=NOT_FOUND_MESSAGE[language], status="not_found", funds=[], sources=[]
    )
