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


SCENARIO_TABLE = [
    "|Сценарий|1 година|5 години|",
    "|---|---|---|",
    "|Кризисен|6 060 EUR|5 690 EUR|",
    "|Песимистичен|8 410 EUR|10 000 EUR|",
    "|Умерен|10 350 EUR|12 270 EUR|",
    "|Благоприятен|14 280 EUR|15 010 EUR|",
]
TABLE_HEADER = SCENARIO_TABLE[:2]  # header row + separator row


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


class TestChunkDocumentSize(unittest.TestCase):
    def test_chunk_document_when_all_paragraphs_are_shorter_than_max_chars_then_no_chunk_exceeds_max_chars(
        self,
    ):
        doc = make_document(
            (
                1,
                "\n\n".join(
                    letter * length
                    for letter, length in zip("АБВГДЕ", [10, 35, 20, 45, 5, 30])
                ),
            )
        )

        chunks = chunk_document(doc, INFO, max_chars=50)

        for chunk in chunks:
            with self.subTest(chunk=chunk.metadata.chunk_index):
                self.assertLessEqual(len(chunk.text), 50)

    def test_chunk_document_when_paragraph_is_longer_than_max_chars_then_it_is_split_at_line_boundaries(
        self,
    ):
        doc = make_document((1, "\n".join(["А" * 30, "Б" * 30, "В" * 30])))

        chunks = chunk_document(doc, INFO, max_chars=70)

        self.assertEqual(
            [chunk.text for chunk in chunks], ["А" * 30 + "\n" + "Б" * 30, "В" * 30]
        )

    def test_chunk_document_when_table_is_longer_than_max_chars_then_every_data_row_stays_intact_in_exactly_one_chunk(
        self,
    ):
        doc = make_document((1, "\n".join(SCENARIO_TABLE)))

        chunks = chunk_document(doc, INFO, max_chars=80)

        self.assertGreater(len(chunks), 1)
        for row in SCENARIO_TABLE[2:]:
            with self.subTest(row=row):
                self.assertEqual(
                    sum(row in chunk.text.split("\n") for chunk in chunks), 1
                )

    def test_chunk_document_when_single_line_is_longer_than_max_chars_then_it_is_kept_whole(
        self,
    ):
        chunks = chunk_document(make_document((1, "А" * 100)), INFO, max_chars=50)

        self.assertEqual([chunk.text for chunk in chunks], ["А" * 100])

    def test_chunk_document_when_oversized_table_is_split_then_every_piece_keeps_the_section_and_page(
        self,
    ):
        doc = make_document(
            (3, "## Сценарии за резултатите\n\n" + "\n".join(SCENARIO_TABLE))
        )

        chunks = chunk_document(doc, INFO, max_chars=80)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            with self.subTest(chunk=chunk.metadata.chunk_index):
                self.assertEqual(chunk.metadata.section, "Сценарии за резултатите")
                self.assertEqual(
                    (chunk.metadata.page_start, chunk.metadata.page_end), (3, 3)
                )


class TestChunkDocumentTableHeader(unittest.TestCase):
    def test_chunk_document_when_table_is_split_then_every_piece_starts_with_the_header_and_separator_rows(
        self,
    ):
        chunks = chunk_document(
            make_document((1, "\n".join(SCENARIO_TABLE))), INFO, max_chars=80
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            with self.subTest(chunk=chunk.metadata.chunk_index):
                self.assertEqual(chunk.text.split("\n")[:2], TABLE_HEADER)

    def test_chunk_document_when_table_is_split_then_header_is_not_repeated_within_a_piece(
        self,
    ):
        chunks = chunk_document(
            make_document((1, "\n".join(SCENARIO_TABLE))), INFO, max_chars=80
        )

        for chunk in chunks:
            with self.subTest(chunk=chunk.metadata.chunk_index):
                self.assertEqual(chunk.text.split("\n").count(TABLE_HEADER[0]), 1)

    def test_chunk_document_when_table_is_split_then_every_piece_including_its_repeated_header_fits_max_chars(
        self,
    ):
        chunks = chunk_document(
            make_document((1, "\n".join(SCENARIO_TABLE))), INFO, max_chars=80
        )

        for chunk in chunks:
            with self.subTest(chunk=chunk.metadata.chunk_index):
                self.assertLessEqual(len(chunk.text), 80)

    def test_chunk_document_when_table_fits_max_chars_then_it_stays_one_piece_with_a_single_header(
        self,
    ):
        table = "\n".join(SCENARIO_TABLE)

        chunks = chunk_document(make_document((1, table)), INFO)

        self.assertEqual([chunk.text for chunk in chunks], [table])

    def test_chunk_document_when_oversized_paragraph_is_not_a_table_then_no_line_is_repeated(
        self,
    ):
        lines = ["А" * 30, "Б" * 30, "В" * 30]

        chunks = chunk_document(
            make_document((1, "\n".join(lines))), INFO, max_chars=40
        )

        self.assertEqual([chunk.text for chunk in chunks], lines)


class TestChunkDocumentSentences(unittest.TestCase):
    def test_chunk_document_when_single_line_is_longer_than_max_chars_then_it_is_split_at_sentence_ends(
        self,
    ):
        line = "Първо изречение е тук. Второ изречение е тук. Трето изречение е тук."

        chunks = chunk_document(make_document((1, line)), INFO, max_chars=50)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["Първо изречение е тук. Второ изречение е тук.", "Трето изречение е тук."],
        )

    def test_chunk_document_when_sentences_end_with_question_or_exclamation_marks_then_they_are_split_there_too(
        self,
    ):
        line = "Какво е фондът? Фондът е подфонд! Това е краят."

        chunks = chunk_document(make_document((1, line)), INFO, max_chars=40)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["Какво е фондът? Фондът е подфонд!", "Това е краят."],
        )

    def test_chunk_document_when_period_is_followed_by_a_lowercase_word_then_it_is_not_treated_as_a_sentence_end(
        self,
    ):
        line = "Управлява се от Кей Би Си, т.е. от управляващото дружество. Второто изречение е тук."

        chunks = chunk_document(make_document((1, line)), INFO, max_chars=40)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            [
                "Управлява се от Кей Би Си, т.е. от управляващото дружество.",
                "Второто изречение е тук.",
            ],
        )

    def test_chunk_document_when_long_line_is_inside_a_multi_line_paragraph_then_only_that_line_is_split_into_sentences(
        self,
    ):
        paragraph = "Кратък ред.\nПърво изречение е тук. Второ изречение е тук. Трето изречение е тук."

        chunks = chunk_document(make_document((1, paragraph)), INFO, max_chars=50)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            [
                "Кратък ред.",
                "Първо изречение е тук. Второ изречение е тук.",
                "Трето изречение е тук.",
            ],
        )

    def test_chunk_document_when_long_line_is_a_table_row_then_it_is_not_split_into_sentences(
        self,
    ):
        row = "|Разходи за вход|Процентът е 3%. Таксата се плаща веднъж. Процентът е 3%. Таксата се плаща веднъж. |"

        chunks = chunk_document(make_document((1, row)), INFO, max_chars=50)

        self.assertEqual([chunk.text for chunk in chunks], [row])


class TestChunkDocumentIdentity(unittest.TestCase):
    TEXT = "\n\n".join(letter * 40 for letter in "АБВ")

    def test_chunk_document_when_same_document_is_chunked_twice_then_chunk_ids_are_equal(
        self,
    ):
        first = chunk_document(make_document((1, self.TEXT)), INFO, max_chars=50)
        second = chunk_document(make_document((1, self.TEXT)), INFO, max_chars=50)

        self.assertEqual(
            [chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second]
        )

    def test_chunk_document_when_document_has_several_chunks_then_chunk_ids_are_unique(
        self,
    ):
        chunks = chunk_document(make_document((1, self.TEXT)), INFO, max_chars=50)

        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))

    def test_chunk_document_when_file_content_changes_then_all_chunk_ids_change(self):
        original = chunk_document(make_document((1, self.TEXT)), INFO, max_chars=50)
        replaced_doc = make_document((1, self.TEXT)).model_copy(
            update={"content_hash": "def456"}
        )

        replaced = chunk_document(replaced_doc, INFO, max_chars=50)

        self.assertFalse(
            {chunk.chunk_id for chunk in original}
            & {chunk.chunk_id for chunk in replaced}
        )

    def test_chunk_document_when_document_is_chunked_then_document_fields_are_copied_to_every_chunk(
        self,
    ):
        chunks = chunk_document(make_document((1, self.TEXT)), INFO, max_chars=50)

        for chunk in chunks:
            with self.subTest(chunk=chunk.metadata.chunk_index):
                self.assertEqual(
                    (
                        chunk.metadata.doc_id,
                        chunk.metadata.file_name,
                        chunk.metadata.content_hash,
                    ),
                    ("test", "test.pdf", "abc123"),
                )

    def test_chunk_document_when_document_is_chunked_then_document_info_fields_are_copied_to_every_chunk(
        self,
    ):
        chunks = chunk_document(make_document((1, self.TEXT)), INFO, max_chars=50)

        for chunk in chunks:
            with self.subTest(chunk=chunk.metadata.chunk_index):
                self.assertEqual(
                    (
                        chunk.metadata.doc_type,
                        chunk.metadata.fund,
                        chunk.metadata.umbrella_fund,
                    ),
                    ("kid", "Test Fund", "Test Umbrella"),
                )

    def test_chunk_document_when_document_is_not_fund_specific_then_fund_fields_are_none(
        self,
    ):
        info = DocumentInfo(doc_type="investor_rights", fund=None, umbrella_fund=None)

        chunks = chunk_document(make_document((1, self.TEXT)), info, max_chars=50)

        for chunk in chunks:
            with self.subTest(chunk=chunk.metadata.chunk_index):
                self.assertEqual(
                    (chunk.metadata.fund, chunk.metadata.umbrella_fund), (None, None)
                )


class TestChunkDocumentText(unittest.TestCase):
    def test_chunk_document_when_page_has_markdown_emphasis_then_chunk_text_is_cleaned(
        self,
    ):
        chunks = chunk_document(
            make_document((1, "**Умерен** сценарий и <u>важна</u> _бележка_.")), INFO
        )

        self.assertEqual(chunks[0].text, "Умерен сценарий и важна бележка.")

    def test_chunk_document_when_page_has_picture_markers_and_br_tags_then_chunk_text_has_neither(
        self,
    ):
        page = "<!-- Start of picture text -->ISIN код: BG9000005223<br>Подфонд на договорен фонд<!-- End of picture text -->"

        chunks = chunk_document(make_document((1, page)), INFO)

        self.assertEqual(
            chunks[0].text, "ISIN код: BG9000005223 Подфонд на договорен фонд"
        )

    def test_chunk_document_when_page_has_latin_lookalikes_then_chunk_text_is_pure_cyrillic(
        self,
    ):
        chunks = chunk_document(make_document((1, "Валута: eвро")), INFO)  # Latin e

        self.assertEqual(chunks[0].text, "Валута: евро")

    def test_chunk_document_when_chunk_has_fund_and_section_then_embed_text_starts_with_both(
        self,
    ):
        chunks = chunk_document(
            make_document((1, "## Разходи\n\nТекст за разходите.")), INFO
        )

        self.assertEqual(
            chunks[0].embed_text,
            "Test Fund | Разходи\n\n## Разходи\n\nТекст за разходите.",
        )

    def test_chunk_document_when_chunk_has_fund_but_no_section_then_embed_text_starts_with_fund_only(
        self,
    ):
        chunks = chunk_document(make_document((1, "Текст без заглавие.")), INFO)

        self.assertEqual(chunks[0].embed_text, "Test Fund\n\nТекст без заглавие.")

    def test_chunk_document_when_chunk_has_neither_fund_nor_section_then_embed_text_equals_text(
        self,
    ):
        info = DocumentInfo(doc_type="investor_rights", fund=None, umbrella_fund=None)

        chunks = chunk_document(make_document((1, "Текст без заглавие.")), info)

        self.assertEqual(chunks[0].embed_text, chunks[0].text)
