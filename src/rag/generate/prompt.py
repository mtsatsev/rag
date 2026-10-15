from rag.models import Chunk, FundQuery

SOURCE_LABEL = "S{number}"

SYSTEM_PROMPT = """\
You are an assistant for relationship managers and support staff at an investment fund manager.
You answer questions about investment funds using ONLY the numbered sources you are given.

Rules:
1. Use only facts stated in the sources. Never use outside knowledge, never guess, never calculate new figures.
2. Every fact in your answer must be supported by a citation: the source label (e.g. "S2") and a short
   quote copied word for word from that source (at most 25 words, in the source's original language).
3. Each source belongs to one fund (or to no specific fund). Never combine information about different funds
   as if it were about one fund. If the question is about one fund, ignore sources about other funds.
4. If the sources do not contain the answer, set status to "not_found", give a one-sentence answer saying
   the information is not in the documents, and give no citations.
5. If the question could refer to several funds and the sources do not make clear which one is meant,
   set status to "ambiguous" and explain which funds the sources cover.
6. If sources contradict each other, report both versions with their citations and describe the
   contradiction in "conflicts". Do not silently choose one.
7. Otherwise set status to "answered".
8. Answer in the language of the question (Bulgarian or English), even if the sources are in another language.
   Keep fund names exactly as written in the sources. Be concise: a few sentences or a short list.

Reply with JSON only, in this shape:
{"status": "answered" | "not_found" | "ambiguous",
 "answer": "...",
 "citations": [{"source": "S1", "quote": "..."}],
 "conflicts": ["..."]}
"""


def source_label(number: int) -> str:
    return SOURCE_LABEL.format(number=number)


def format_source(number: int, chunk: Chunk) -> str:
    metadata = chunk.metadata
    pages = (
        f"p. {metadata.page_start}"
        if metadata.page_start == metadata.page_end
        else f"pp. {metadata.page_start}-{metadata.page_end}"
    )
    header = " | ".join(
        part
        for part in (
            f"[{source_label(number)}] {metadata.file_name}, {pages}",
            f"Document type: {metadata.doc_type}",
            f"Fund: {metadata.fund or 'not fund-specific'}",
            f"Umbrella: {metadata.umbrella_fund}" if metadata.umbrella_fund else "",
            f"Section: {metadata.section}" if metadata.section else "",
        )
        if part
    )
    return f"{header}\n{chunk.text}"


def describe_fund_query(fund_query: FundQuery | None) -> str:
    if fund_query is None or (not fund_query.funds and not fund_query.umbrellas):
        return "No specific fund was recognised in the question."
    named = [ref.fund for ref in fund_query.funds] + [
        f"{umbrella} (umbrella fund)" for umbrella in fund_query.umbrellas
    ]
    return "The question is about: " + "; ".join(named) + "."



def build_user_prompt(
    question: str, chunks: list[Chunk], fund_query: FundQuery | None = None
) -> str:
    sources = "\n\n".join(
        format_source(number, chunk) for number, chunk in enumerate(chunks, start=1)
    )
    return (
        f"Question: {question}\n"
        f"{describe_fund_query(fund_query)}\n\n"
        f"Sources:\n\n{sources}\n\n"
        "Answer the question using only these sources, following the rules. Reply with JSON only."
    )


def build_messages(
    question: str, chunks: list[Chunk], fund_query: FundQuery | None = None
) -> list[dict[str, str]]:
    """Chat messages for the LLM; source S{i} is chunks[i - 1]."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, chunks, fund_query)},
    ]


def chunk_for_label(label: str, chunks: list[Chunk]) -> Chunk | None:
    """Map a source label such as 'S2' back to its chunk; None if the label is unknown."""
    number = label.strip().upper().removeprefix("S")
    if not number.isdigit() or not 1 <= int(number) <= len(chunks):
        return None
    return chunks[int(number) - 1]
