import unittest

import numpy as np
from huggingface_hub import try_to_load_from_cache

from rag.index.embed import BGE_M3, BgeM3Embedder


def require_cached_model(model_name: str) -> None:
    if not isinstance(try_to_load_from_cache(model_name, "config.json"), str):
        raise unittest.SkipTest(
            f"{model_name} is not downloaded; run ingestion once to fetch it"
        )


class TestBgeM3Embedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_cached_model(BGE_M3)
        cls.embedder = BgeM3Embedder()

    def test_embed_documents_when_texts_are_embedded_then_returns_one_1024_dimensional_vector_per_text(
        self,
    ):
        vectors = self.embedder.embed_documents(
            ["Входни такси до 3%.", "Entry costs up to 3%."]
        )

        self.assertEqual(vectors.shape, (2, 1024))

    def test_embed_documents_when_texts_are_embedded_then_vectors_are_normalised(self):
        vectors = self.embedder.embed_documents(
            ["Входни такси до 3%.", "Entry costs up to 3%."]
        )

        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)

    def test_embed_query_when_question_is_embedded_then_returns_a_single_1024_dimensional_vector(
        self,
    ):
        self.assertEqual(
            self.embedder.embed_query("Какви са разходите?").shape, (1024,)
        )

    def test_embed_query_when_bulgarian_question_matches_an_english_passage_then_it_is_closer_than_an_unrelated_one(
        self,
    ):
        question = self.embedder.embed_query("Какви са входните такси?")
        related, unrelated = self.embedder.embed_documents(
            [
                "Entry costs: up to 3% of the amount you pay in when entering this investment.",
                "The board of directors of the management company meets four times a year.",
            ]
        )

        self.assertGreater(question @ related, question @ unrelated)
