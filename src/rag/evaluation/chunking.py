import re

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer

from rag.ingest.chunk import PARAGRAPH_SEPARATOR, is_heading
from rag.models import Chunk, ParsedDocument

ARTIFACTS = {
    "bold": re.compile(r"\*\*"),
    "underline": re.compile(r"</?u>"),
    "br": re.compile(r"<br>"),
    "picture_marker": re.compile(r"<!--"),
    "running_header": re.compile(r"Optimum Fund Ref\."),
    "page_number_line": re.compile(r"^\d{1,3}$", re.MULTILINE),
    "mixed_script_word": re.compile(r"\b(?=\w*[Ѐ-ӿ])(?=\w*[A-Za-z])\w+\b"),
}
NO_FUND = "(no fund)"


class ChunkerEvaluator:
    def __init__(
        self,
        chunks: list[Chunk],
        documents: list[ParsedDocument],
        max_chars: int = 1500,
        tiny_chars: int = 200,
    ) -> None:
        self.chunks = chunks
        self.documents = documents
        self.max_chars = max_chars
        self.tiny_chars = tiny_chars
        self._vectors = None

    def frame(self) -> pl.DataFrame:
        """One row per chunk."""
        return pl.DataFrame(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "doc_id": chunk.metadata.doc_id,
                    "doc_type": chunk.metadata.doc_type,
                    "umbrella_fund": chunk.metadata.umbrella_fund,
                    "fund": chunk.metadata.fund,
                    "section": chunk.metadata.section,
                    "page_start": chunk.metadata.page_start,
                    "page_end": chunk.metadata.page_end,
                    "length": len(chunk.text),
                    "body_length": len(_without_leading_headings(chunk.text)),
                    "text": chunk.text,
                }
                for chunk in self.chunks
            ]
        )

    def length_stats(self) -> pl.DataFrame:
        """Chunk length distribution per document, incl. oversized and tiny chunks."""
        return (
            self.frame()
            .group_by("doc_id")
            .agg(
                pl.len().alias("chunks"),
                pl.col("length").min().alias("min"),
                pl.col("length").median().alias("median"),
                pl.col("length").quantile(0.95).alias("p95"),
                pl.col("length").max().alias("max"),
                (pl.col("length") > self.max_chars).sum().alias("over_max"),
                (pl.col("body_length") > self.max_chars).sum().alias("body_over_max"),
                (pl.col("length") < self.tiny_chars).sum().alias("tiny"),
            )
            .sort("doc_id")
        )

    def coverage(self) -> pl.DataFrame:
        """Share of chunks with a fund / section, and pages covered per document."""
        page_counts = pl.DataFrame(
            {
                "doc_id": [doc.doc_id for doc in self.documents],
                "pages_parsed": [len(doc.pages) for doc in self.documents],
            }
        )
        parsed_pages = {
            doc.doc_id: {page.page_number for page in doc.pages}
            for doc in self.documents
        }
        covered = pl.DataFrame(
            [
                {"doc_id": chunk.metadata.doc_id, "page": page}
                for chunk in self.chunks
                for page in range(
                    chunk.metadata.page_start, chunk.metadata.page_end + 1
                )
                if page in parsed_pages.get(chunk.metadata.doc_id, set())
            ]
        )
        return (
            self.frame()
            .group_by("doc_id")
            .agg(
                pl.len().alias("chunks"),
                pl.col("fund").is_not_null().mean().alias("share_with_fund"),
                (pl.col("section") != "").mean().alias("share_with_section"),
            )
            .join(
                covered.group_by("doc_id").agg(
                    pl.col("page").n_unique().alias("pages_covered")
                ),
                on="doc_id",
            )
            .join(page_counts, on="doc_id", how="left")
            .sort("doc_id")
        )

    def artifacts(self) -> pl.DataFrame:
        """Leftover markup, mixed-script words and page furniture per document."""
        counts = pl.DataFrame(
            [
                {
                    "doc_id": chunk.metadata.doc_id,
                    **{
                        name: len(pattern.findall(chunk.text))
                        for name, pattern in ARTIFACTS.items()
                    },
                }
                for chunk in self.chunks
            ]
        )
        return (
            counts.group_by("doc_id")
            .agg(pl.col(name).sum() for name in ARTIFACTS)
            .sort("doc_id")
        )

    def artifact_examples(self, name: str, limit: int = 10) -> pl.DataFrame:
        """Chunks containing a given artifact, to inspect what survived."""
        pattern = ARTIFACTS[name]
        return (
            self.frame()
            .filter(
                pl.Series(
                    [pattern.search(chunk.text) is not None for chunk in self.chunks]
                )
            )
            .select("doc_id", "page_start", "section", "text")
            .head(limit)
        )

    def fund_breakdown(self) -> pl.DataFrame:
        """Chunks and page range per (umbrella, fund)."""
        return (
            self.frame()
            .group_by("umbrella_fund", "fund")
            .agg(
                pl.len().alias("chunks"),
                pl.col("doc_id").unique().sort().alias("documents"),
                pl.col("page_start").min().alias("first_page"),
                pl.col("page_end").max().alias("last_page"),
            )
            .sort("umbrella_fund", "first_page", nulls_last=True)
        )

    def vectors(self) -> np.ndarray:
        """TF-IDF vectors of the chunk texts (L2-normalised, so dot product = cosine)."""
        if self._vectors is None:
            self._vectors = TfidfVectorizer(sublinear_tf=True).fit_transform(
                chunk.text for chunk in self.chunks
            )
        return self._vectors

    def near_duplicates(self, threshold: float = 0.9, vectors=None) -> pl.DataFrame:
        """Pairs of nearly identical chunks that belong to different funds."""
        pairs = [
            {
                "similarity": similarity,
                "fund_a": self.chunks[i].metadata.fund or NO_FUND,
                "fund_b": self.chunks[j].metadata.fund or NO_FUND,
                "doc_a": self.chunks[i].metadata.doc_id,
                "page_a": self.chunks[i].metadata.page_start,
                "doc_b": self.chunks[j].metadata.doc_id,
                "page_b": self.chunks[j].metadata.page_start,
                "section_a": self.chunks[i].metadata.section,
                "text_a": self.chunks[i].text[:200],
            }
            for i, j, similarity in self._cross_fund_pairs(threshold, vectors)
        ]
        return (
            pl.DataFrame(pairs).sort("similarity", descending=True)
            if pairs
            else pl.DataFrame()
        )

    def near_duplicate_rate(self, threshold: float = 0.9, vectors=None) -> float:
        """Share of chunks that have a near-duplicate in a different fund."""
        affected = {
            index
            for i, j, _ in self._cross_fund_pairs(threshold, vectors)
            for index in (i, j)
        }
        return len(affected) / len(self.chunks)

    def _cross_fund_pairs(
        self, threshold: float, vectors=None
    ) -> list[tuple[int, int, float]]:
        vectors = self.vectors() if vectors is None else vectors
        similarity = vectors @ vectors.T
        similarity = (
            similarity.toarray()
            if hasattr(similarity, "toarray")
            else np.asarray(similarity)
        )
        rows, cols = np.nonzero(np.triu(similarity, k=1) >= threshold)
        return [
            (int(i), int(j), float(similarity[i, j]))
            for i, j in zip(rows, cols)
            if self.chunks[i].metadata.fund != self.chunks[j].metadata.fund
        ]

    def plot_lengths(self, bins: int = 50) -> Figure:
        """Histogram of chunk lengths with the max_chars and tiny_chars limits."""
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.hist([len(chunk.text) for chunk in self.chunks], bins=bins)
        axis.axvline(
            self.max_chars,
            color="red",
            linestyle="--",
            label=f"max_chars={self.max_chars}",
        )
        axis.axvline(
            self.tiny_chars,
            color="orange",
            linestyle="--",
            label=f"tiny<{self.tiny_chars}",
        )
        axis.set(
            xlabel="characters", ylabel="chunks", title="Chunk length distribution"
        )
        axis.legend()
        return figure

    def plot_map(
        self, color_by: str = "doc_type", vectors=None, top: int = 12, seed: int = 42
    ) -> Figure:
        """2-D UMAP projection of the chunks, coloured by a metadata column.

        Pass `vectors` (e.g. bge-m3 embeddings) to map those instead of TF-IDF.
        Only the `top` most frequent values get their own colour; the rest are 'other'.
        """
        from umap import UMAP

        vectors = self.vectors() if vectors is None else vectors
        points = UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(
            vectors
        )

        labels = [
            str(getattr(chunk.metadata, color_by) or NO_FUND) for chunk in self.chunks
        ]
        frequent = [label for label, _ in _most_common(labels, top)]
        labels = [label if label in frequent else "other" for label in labels]

        figure, axis = plt.subplots(figsize=(10, 8))
        for label in [*frequent, "other"]:
            mask = np.array([value == label for value in labels])
            if mask.any():
                axis.scatter(
                    points[mask, 0], points[mask, 1], s=8, label=label[:50], alpha=0.7
                )
        axis.set(title=f"Chunks coloured by {color_by}", xticks=[], yticks=[])
        axis.legend(fontsize=7, markerscale=2, loc="best")
        return figure


def _without_leading_headings(text: str) -> str:
    paragraphs = text.split(PARAGRAPH_SEPARATOR)
    while paragraphs and is_heading(paragraphs[0]):
        paragraphs.pop(0)
    return PARAGRAPH_SEPARATOR.join(paragraphs)


def _most_common(labels: list[str], top: int) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top]
