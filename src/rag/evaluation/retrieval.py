from collections.abc import Callable
from pathlib import Path
from typing import Literal

import polars as pl
import yaml
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from rag.models import Chunk, FundQuery
from rag.retrieve.dense import DenseRetriever
from rag.retrieve.hybrid import HybridRetriever

Retrieve = Callable[[str, int], list[Chunk]]  # (question, k) -> ranked chunks


class ExpectedSource(BaseModel):
    doc_id: str = Field(
        description="File stem of the document that answers the question"
    )
    pages: tuple[int, int] = Field(description="Inclusive page range of the answer")


class EvalQuestion(BaseModel):
    id: str = Field(description="Short unique id")
    question: str = Field(description="The question as a user would ask it")
    language: Literal["bg", "en"] = Field(description="Language of the question")
    category: str = Field(
        description="e.g. single_fund, cross_lingual, multi_doc, unanswerable"
    )
    funds: list[str] = Field(
        default=[], description="Canonical funds the question is about"
    )
    sources: list[ExpectedSource] = Field(
        default=[],
        description="Where the answer is; empty = not answerable from the documents",
    )

    @property
    def answerable(self) -> bool:
        return bool(self.sources)


def load_questions(path: Path) -> list[EvalQuestion]:
    return [
        EvalQuestion.model_validate(item)
        for item in yaml.safe_load(path.read_text("utf-8"))
    ]


def retrieval_variants(retriever: HybridRetriever) -> dict[str, Retrieve]:
    """Ablations of one retriever: each search alone, with and without the fund filter, and hybrid."""
    store, keyword_index, resolver = (
        retriever.store,
        retriever.keyword_index,
        retriever.resolver,
    )

    def dense(question: str, k: int, use_filter: bool = True) -> list[Chunk]:
        if use_filter:
            return [
                r.chunk
                for r in DenseRetriever(store, resolver).retrieve(question, k).chunks
            ]
        return [chunk for chunk, _ in store.query(question, k=k)]

    def keyword(question: str, k: int, use_filter: bool = True) -> list[Chunk]:
        fund_query = resolver.resolve(question) if use_filter else FundQuery()
        return [
            chunk
            for chunk, _ in keyword_index.search(question, k=k, fund_query=fund_query)
        ]

    return {
        "dense_no_filter": lambda question, k: dense(question, k, use_filter=False),
        "keyword_no_filter": lambda question, k: keyword(question, k, use_filter=False),
        "dense": dense,
        "keyword": keyword,
        "hybrid": lambda question, k: [
            retrieved.chunk for retrieved in retriever.retrieve(question, k=k).chunks
        ],
    }


def is_relevant(chunk: Chunk, source: ExpectedSource) -> bool:
    """Same document and overlapping page range."""
    first, last = source.pages
    return (
        chunk.metadata.doc_id == source.doc_id
        and chunk.metadata.page_start <= last
        and chunk.metadata.page_end >= first
    )


def is_fund_allowed(chunk: Chunk, question: EvalQuestion) -> bool:
    """A chunk is on-fund if it belongs to an asked fund or is not fund-specific."""
    return chunk.metadata.fund is None or chunk.metadata.fund in question.funds


class RetrievalEvaluator:
    def __init__(
        self, questions: list[EvalQuestion], ks: tuple[int, ...] = (1, 3, 5, 10)
    ) -> None:
        self.questions = questions
        self.ks = ks
        self.results: dict[
            str, dict[str, list[Chunk]]
        ] = {}  # variant -> question id -> chunks

    def run(self, variants: dict[str, Retrieve]) -> "RetrievalEvaluator":
        """Retrieve max(ks) chunks per question for every variant; reports use these results."""
        for name, retrieve in variants.items():
            self.results[name] = {
                question.id: retrieve(question.question, max(self.ks))
                for question in self.questions
            }
        return self

    def per_question(self, variant: str) -> pl.DataFrame:
        """One row per question with hit@k, recall@k, reciprocal rank and fund purity@k."""
        rows = []
        for question in self.questions:
            chunks = self.results[variant][question.id]
            row = {
                "id": question.id,
                "category": question.category,
                "language": question.language,
                "answerable": question.answerable,
                "first_relevant_rank": _first_relevant_rank(chunks, question),
            }
            row["reciprocal_rank"] = (
                1 / row["first_relevant_rank"] if row["first_relevant_rank"] else 0.0
            )
            for k in self.ks:
                top = chunks[:k]
                found = [
                    any(is_relevant(chunk, source) for chunk in top)
                    for source in question.sources
                ]
                row[f"hit@{k}"] = any(found) if question.answerable else None
                row[f"recall@{k}"] = (
                    sum(found) / len(found) if question.answerable else None
                )
                row[f"purity@{k}"] = (
                    sum(is_fund_allowed(chunk, question) for chunk in top) / len(top)
                    if question.funds and top
                    else None
                )
            if not question.answerable:
                row["reciprocal_rank"] = None
            rows.append(row)
        return pl.DataFrame(rows)

    def summary(self, by: str | None = None) -> pl.DataFrame:
        """Mean of every metric per variant, optionally split by 'category' or 'language'."""
        metrics = [
            "reciprocal_rank",
            *(f"{m}@{k}" for m in ("hit", "recall", "purity") for k in self.ks),
        ]
        frames = [
            self.per_question(variant).with_columns(pl.lit(variant).alias("variant"))
            for variant in self.results
        ]
        keys = ["variant", by] if by else ["variant"]
        return (
            pl.concat(frames, how="diagonal_relaxed")
            .group_by(keys, maintain_order=True)
            .agg(
                pl.len().alias("questions"),
                *(pl.col(metric).cast(pl.Float64).mean() for metric in metrics),
            )
            .rename({"reciprocal_rank": "mrr"})
        )

    def failures(self, variant: str, k: int | None = None) -> pl.DataFrame:
        """Answerable questions with no relevant chunk in the top k, with what was retrieved instead."""
        k = k or max(self.ks)
        rows = []
        for question in self.questions:
            top = self.results[variant][question.id][:k]
            if question.answerable and not any(
                is_relevant(chunk, source)
                for chunk in top
                for source in question.sources
            ):
                rows.append(
                    {
                        "id": question.id,
                        "question": question.question,
                        "expected": [
                            f"{s.doc_id} p{s.pages[0]}-{s.pages[1]}"
                            for s in question.sources
                        ],
                        "retrieved": [
                            f"{chunk.metadata.doc_id} p{chunk.metadata.page_start}-{chunk.metadata.page_end}"
                            for chunk in top
                        ],
                        "retrieved_funds": [chunk.metadata.fund for chunk in top],
                    }
                )
        return pl.DataFrame(rows)

    def retrieved(self, variant: str, question_id: str) -> pl.DataFrame:
        """What one variant retrieved for one question, marked relevant / on-fund."""
        question = next(
            question for question in self.questions if question.id == question_id
        )
        return pl.DataFrame(
            [
                {
                    "rank": rank,
                    "doc_id": chunk.metadata.doc_id,
                    "pages": f"{chunk.metadata.page_start}-{chunk.metadata.page_end}",
                    "fund": chunk.metadata.fund,
                    "section": chunk.metadata.section,
                    "relevant": any(
                        is_relevant(chunk, source) for source in question.sources
                    ),
                    "on_fund": is_fund_allowed(chunk, question)
                    if question.funds
                    else None,
                    "text": chunk.text[:200],
                }
                for rank, chunk in enumerate(
                    self.results[variant][question_id], start=1
                )
            ]
        )

    def plot_metric(self, metric: str = "hit") -> Figure:
        """Metric@k curves, one line per variant (metric: 'hit', 'recall' or 'purity')."""
        summary = self.summary()
        figure, axis = plt.subplots(figsize=(8, 4))
        for row in summary.iter_rows(named=True):
            axis.plot(
                self.ks,
                [row[f"{metric}@{k}"] for k in self.ks],
                marker="o",
                label=row["variant"],
            )
        axis.set(
            xlabel="k",
            ylabel=f"{metric}@k",
            ylim=(0, 1.05),
            xticks=list(self.ks),
            title=f"{metric}@k per variant",
        )
        axis.legend()
        return figure


def _first_relevant_rank(chunks: list[Chunk], question: EvalQuestion) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if any(is_relevant(chunk, source) for source in question.sources):
            return rank
    return None
