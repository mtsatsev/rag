import unittest

from rag.ingest.chunk import chunk_document
from rag.models import DocumentInfo, ParsedDocument, ParsedPage

INFO = DocumentInfo(doc_type="kid", fund="Test Fund", umbrella_fund="Test Umbrella")


def make_document(*pages: tuple[int, str]) -> ParsedDocument:
    return ParsedDocument(
        doc_id="test",
        file_name="test.pdf",
        content_hash="abc123",
        pages=[
            ParsedPage(page_number=number, markdown=markdown)
            for number, markdown in pages
        ],
    )


class TestChunkDocumentPages(unittest.TestCase):
    def test_chunk_document_when_all_paragraphs_are_on_one_page_then_page_start_equals_page_end(
        self,
    ):
        doc = make_document((1, "Първи абзац.\n\nВтори абзац."))

        chunks = chunk_document(doc, INFO)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            (chunks[0].metadata.page_start, chunks[0].metadata.page_end), (1, 1)
        )

    def test_chunk_document_when_chunk_continues_onto_the_next_page_then_page_end_is_the_next_page(
        self,
    ):
        doc = make_document(
            (1, "Абзац на първа страница."), (2, "Абзац на втора страница.")
        )

        chunks = chunk_document(doc, INFO)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            (chunks[0].metadata.page_start, chunks[0].metadata.page_end), (1, 2)
        )

    def test_chunk_document_when_chunks_split_at_a_page_break_then_each_chunk_has_its_own_page(
        self,
    ):
        doc = make_document((1, "A" * 40), (2, "B" * 40))

        chunks = chunk_document(doc, INFO, max_chars=50)

        self.assertEqual(
            [(chunk.metadata.page_start, chunk.metadata.page_end) for chunk in chunks],
            [(1, 1), (2, 2)],
        )

    def test_chunk_document_when_only_selected_pages_were_parsed_then_original_page_numbers_are_used(
        self,
    ):
        doc = make_document(
            (25, "Абзац на страница 25."), (26, "Абзац на страница 26.")
        )

        chunks = chunk_document(doc, INFO)

        self.assertEqual(
            (chunks[0].metadata.page_start, chunks[0].metadata.page_end), (25, 26)
        )

    def test_chunk_document_when_a_page_is_empty_after_cleaning_then_it_does_not_extend_the_page_range(
        self,
    ):
        doc = make_document(
            (1, "Абзац на първа страница."),
            (2, "<!-- Start of picture text --><!-- End of picture text -->"),
        )

        chunks = chunk_document(doc, INFO)

        self.assertEqual(
            (chunks[0].metadata.page_start, chunks[0].metadata.page_end), (1, 1)
        )

    def test_chunk_document_when_document_produces_several_chunks_then_chunk_index_counts_from_zero(
        self,
    ):
        doc = make_document((1, "\n\n".join(letter * 40 for letter in "ABC")))

        chunks = chunk_document(doc, INFO, max_chars=50)

        self.assertEqual([chunk.metadata.chunk_index for chunk in chunks], [0, 1, 2])
