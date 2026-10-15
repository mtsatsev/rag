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


def is_heading(paragraph: str) -> bool:
    return paragraph.startswith("#")


def section_of(chunks, text: str) -> str:
    return next(chunk.metadata.section for chunk in chunks if text in chunk.text)


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


class TestChunkDocumentSections(unittest.TestCase):
    def test_chunk_document_when_there_is_no_heading_then_section_is_empty(self):
        chunks = chunk_document(make_document((1, "Абзац без заглавие.")), INFO)

        self.assertEqual(chunks[0].metadata.section, "")

    def test_chunk_document_when_paragraph_is_under_a_heading_then_section_is_that_heading(
        self,
    ):
        chunks = chunk_document(
            make_document((1, "# Резюме\n\nТекст на резюмето.")), INFO
        )

        self.assertEqual(section_of(chunks, "Текст на резюмето."), "Резюме")

    def test_chunk_document_when_heading_has_markdown_emphasis_then_section_has_no_markers(
        self,
    ):
        chunks = chunk_document(
            make_document((1, "### **<u>Цел</u>**\n\nТекст на целта.")), INFO
        )

        self.assertEqual(section_of(chunks, "Текст на целта."), "Цел")

    def test_chunk_document_when_subheading_follows_a_heading_then_section_is_the_heading_path(
        self,
    ):
        doc = make_document(
            (1, "# Резюме\n\nТекст на резюмето.\n\n## Цел\n\nТекст на целта.")
        )

        chunks = chunk_document(doc, INFO, max_chars=30)

        self.assertEqual(section_of(chunks, "Текст на целта."), "Резюме > Цел")

    def test_chunk_document_when_sibling_subheading_follows_then_it_replaces_the_previous_one(
        self,
    ):
        doc = make_document((1, "# А\n\n## Б\n\nТекст под Б.\n\n## В\n\nТекст под В."))

        chunks = chunk_document(doc, INFO, max_chars=20)

        self.assertEqual(section_of(chunks, "Текст под В."), "А > В")

    def test_chunk_document_when_new_top_level_heading_follows_then_deeper_headings_are_dropped(
        self,
    ):
        doc = make_document((1, "# А\n\n## Б\n\nТекст под Б.\n\n# В\n\nТекст под В."))

        chunks = chunk_document(doc, INFO, max_chars=20)

        self.assertEqual(section_of(chunks, "Текст под В."), "В")

    def test_chunk_document_when_heading_levels_are_skipped_then_shallower_heading_drops_the_deeper_one(
        self,
    ):
        doc = make_document(
            (1, "# А\n\n#### Б\n\nТекст под Б.\n\n## В\n\nТекст под В.")
        )

        chunks = chunk_document(doc, INFO, max_chars=20)

        self.assertEqual(section_of(chunks, "Текст под Б."), "А > Б")
        self.assertEqual(section_of(chunks, "Текст под В."), "А > В")

    def test_chunk_document_when_section_continues_on_the_next_page_then_it_keeps_the_heading_from_the_previous_page(
        self,
    ):
        doc = make_document(
            (1, "# Разходи\n\nТекст на страница 1."), (2, "Текст на страница 2.")
        )

        chunks = chunk_document(doc, INFO, max_chars=25)

        self.assertEqual(section_of(chunks, "Текст на страница 2."), "Разходи")

    def test_chunk_document_when_chunk_packs_several_subsections_then_section_is_the_path_at_its_first_paragraph(
        self,
    ):
        doc = make_document(
            (1, "# А\n\n### Б\n\nТекст под Б.\n\n### В\n\nТекст под В.")
        )

        chunks = chunk_document(doc, INFO)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata.section, "А")


class TestChunkDocumentBoundaries(unittest.TestCase):
    def test_chunk_document_when_top_level_heading_follows_text_then_it_starts_a_new_chunk(
        self,
    ):
        chunks = chunk_document(
            make_document((1, "Текст преди.\n\n# Разходи\n\nТекст за разходите.")), INFO
        )

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["Текст преди.", "# Разходи\n\nТекст за разходите."],
        )

    def test_chunk_document_when_second_level_heading_follows_text_then_it_starts_a_new_chunk(
        self,
    ):
        chunks = chunk_document(
            make_document(
                (1, "# Резюме\n\nТекст на резюмето.\n\n## Цел\n\nТекст на целта.")
            ),
            INFO,
        )

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["# Резюме\n\nТекст на резюмето.", "## Цел\n\nТекст на целта."],
        )

    def test_chunk_document_when_small_third_level_sections_follow_each_other_then_they_are_packed_into_one_chunk(
        self,
    ):
        text = "## Основни данни\n\n#### Име\n\nOptimum Fund\n\n#### Дата на учредяване\n\n29 June 2006"

        chunks = chunk_document(make_document((1, text)), INFO)

        self.assertEqual([chunk.text for chunk in chunks], [text])

    def test_chunk_document_when_headings_follow_each_other_then_they_stay_together_with_their_content(
        self,
    ):
        text = "# Information concerning the sub-fund X\n\n## 1. Basic details\n\nТекст на раздела."

        chunks = chunk_document(make_document((1, text)), INFO)

        self.assertEqual([chunk.text for chunk in chunks], [text])

    def test_chunk_document_when_heading_fits_but_its_content_does_not_then_heading_moves_to_the_next_chunk(
        self,
    ):
        doc = make_document((1, "Текст на първия абзац.\n\n### Б\n\nТекст под Б."))

        chunks = chunk_document(doc, INFO, max_chars=30)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["Текст на първия абзац.", "### Б\n\nТекст под Б."],
        )

    def test_chunk_document_when_chunks_are_packed_then_no_chunk_ends_with_a_heading(
        self,
    ):
        text = "\n\n".join(f"### Заглавие {i}\n\n{'Текст ' * 4}" for i in range(6))

        chunks = chunk_document(make_document((1, text)), INFO, max_chars=60)

        for chunk in chunks:
            with self.subTest(chunk=chunk.metadata.chunk_index):
                self.assertFalse(is_heading(chunk.text.split("\n\n")[-1]))

    def test_chunk_document_when_section_is_longer_than_max_chars_then_it_is_split_into_chunks_with_the_same_section(
        self,
    ):
        doc = make_document(
            (1, "## Разходи\n\n" + "\n\n".join(letter * 40 for letter in "АБВ"))
        )

        chunks = chunk_document(doc, INFO, max_chars=60)

        self.assertEqual(len(chunks), 3)
        self.assertEqual([chunk.metadata.section for chunk in chunks], ["Разходи"] * 3)
        self.assertTrue(chunks[0].text.startswith("## Разходи"))
