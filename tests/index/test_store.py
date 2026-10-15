import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag.index.store import ChunkStore
from rag.models import Chunk, ChunkMetadata


class FakeEmbedder:
    """Deterministic character-histogram vectors; records every text it embeds."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.embedded.extend(texts)
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vector = np.zeros(32)
        for character in text:
            vector[ord(character) % 32] += 1
        return vector / np.linalg.norm(vector)


def make_chunk(
    doc_id: str,
    index: int,
    text: str,
    fund: str | None = "Алфа",
    content_hash: str = "v1",
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-{content_hash}-{index}",
        text=text,
        metadata=ChunkMetadata(
            doc_id=doc_id,
            file_name=f"{doc_id}.pdf",
            content_hash=content_hash,
            doc_type="kid",
            fund=fund,
            umbrella_fund="Optimum Fund" if fund else None,
            section="Разходи",
            page_start=1,
            page_end=2,
            chunk_index=index,
        ),
    )


KID_CHUNKS = [
    make_chunk("kid", 0, "Входни такси до 3% от сумата."),
    make_chunk("kid", 1, "Препоръчителен период на държане: 5 години."),
]
RIGHTS_CHUNKS = [
    make_chunk("rights", 0, "Всеки инвеститор може да запише дялове.", fund=None)
]


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.embedder = FakeEmbedder()
        self.store = ChunkStore(Path(self.directory.name), self.embedder)

    def tearDown(self):
        self.directory.cleanup()


class TestChunkStoreDocuments(StoreTestCase):
    def test_replace_document_when_new_document_is_added_then_count_equals_its_chunks(
        self,
    ):
        self.store.replace_document(KID_CHUNKS)

        self.assertEqual(self.store.count, 2)

    def test_replace_document_when_same_document_is_added_twice_then_no_duplicates_are_stored(
        self,
    ):
        self.store.replace_document(KID_CHUNKS)
        self.store.replace_document(KID_CHUNKS)

        self.assertEqual(self.store.count, 2)

    def test_replace_document_when_new_version_is_added_then_only_the_new_chunks_remain(
        self,
    ):
        new_version = [
            make_chunk("kid", 0, "Входни такси до 2% от сумата.", content_hash="v2")
        ]
        self.store.replace_document(KID_CHUNKS)

        self.store.replace_document(new_version)

        self.assertEqual(
            [chunk.chunk_id for chunk in self.store.all_chunks()], ["kid-v2-0"]
        )

    def test_delete_document_when_document_is_deleted_then_other_documents_are_untouched(
        self,
    ):
        self.store.replace_document(KID_CHUNKS)
        self.store.replace_document(RIGHTS_CHUNKS)

        self.store.delete_document("kid")

        self.assertEqual(
            [chunk.metadata.doc_id for chunk in self.store.all_chunks()], ["rights"]
        )

    def test_has_document_when_same_content_hash_is_indexed_then_returns_true(self):
        self.store.replace_document(KID_CHUNKS)

        self.assertTrue(self.store.has_document("kid", "v1"))

    def test_has_document_when_content_hash_differs_then_returns_false(self):
        self.store.replace_document(KID_CHUNKS)

        self.assertFalse(self.store.has_document("kid", "v2"))

    def test_has_document_when_document_was_never_indexed_then_returns_false(self):
        self.assertFalse(self.store.has_document("kid", "v1"))


class TestChunkStoreRoundTrip(StoreTestCase):
    def test_all_chunks_when_chunks_are_read_back_then_they_equal_the_originals(self):
        self.store.replace_document(KID_CHUNKS)

        self.assertEqual(
            sorted(self.store.all_chunks(), key=lambda chunk: chunk.chunk_id),
            KID_CHUNKS,
        )

    def test_all_chunks_when_chunk_has_no_fund_then_fund_fields_are_read_back_as_none(
        self,
    ):
        self.store.replace_document(RIGHTS_CHUNKS)

        metadata = self.store.all_chunks()[0].metadata

        self.assertEqual((metadata.fund, metadata.umbrella_fund), (None, None))

    def test_replace_document_when_chunks_are_stored_then_embed_text_is_embedded_not_text(
        self,
    ):
        self.store.replace_document(KID_CHUNKS)

        self.assertEqual(
            self.embedder.embedded, [chunk.embed_text for chunk in KID_CHUNKS]
        )


class TestChunkStoreQuery(StoreTestCase):
    def test_query_when_query_equals_a_chunks_embed_text_then_that_chunk_comes_first(
        self,
    ):
        self.store.replace_document(KID_CHUNKS)
        self.store.replace_document(RIGHTS_CHUNKS)

        results = self.store.query(KID_CHUNKS[1].embed_text, k=3)

        self.assertEqual(results[0][0].chunk_id, KID_CHUNKS[1].chunk_id)

    def test_query_when_results_are_returned_then_scores_are_sorted_from_best_to_worst(
        self,
    ):
        self.store.replace_document(KID_CHUNKS)
        self.store.replace_document(RIGHTS_CHUNKS)

        scores = [score for _, score in self.store.query("Входни такси", k=3)]

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_query_when_k_is_smaller_than_the_store_then_at_most_k_results_are_returned(
        self,
    ):
        self.store.replace_document(KID_CHUNKS)
        self.store.replace_document(RIGHTS_CHUNKS)

        self.assertEqual(len(self.store.query("Входни такси", k=2)), 2)

    def test_query_when_filtered_by_fund_then_only_that_funds_chunks_are_returned(self):
        self.store.replace_document(KID_CHUNKS)
        self.store.replace_document(RIGHTS_CHUNKS)

        results = self.store.query(
            RIGHTS_CHUNKS[0].embed_text, k=3, where={"fund": "Алфа"}
        )

        self.assertEqual({chunk.metadata.fund for chunk, _ in results}, {"Алфа"})
