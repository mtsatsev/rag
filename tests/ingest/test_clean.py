import unittest

from rag.ingest.clean import clean_markdown, match_key


class TestCleanMarkdownEmphasis(unittest.TestCase):
    def test_clean_markdown_when_text_is_bold_then_markers_are_removed_and_text_is_kept(
        self,
    ):
        self.assertEqual(clean_markdown("**Умерен**"), "Умерен")

    def test_clean_markdown_when_text_is_italic_then_markers_are_removed_and_text_is_kept(
        self,
    ):
        self.assertEqual(clean_markdown("_Executive Director_"), "Executive Director")

    def test_clean_markdown_when_text_is_bold_and_italic_then_all_markers_are_removed(
        self,
    ):
        self.assertEqual(
            clean_markdown("**_Date of incorporation_**"), "Date of incorporation"
        )

    def test_clean_markdown_when_text_is_underlined_then_tags_are_removed_and_text_is_kept(
        self,
    ):
        self.assertEqual(clean_markdown("<u>www.kbc.be/kid.</u>"), "www.kbc.be/kid.")

    def test_clean_markdown_when_emphasis_is_inside_a_sentence_then_surrounding_text_is_kept(
        self,
    ):
        self.assertEqual(
            clean_markdown("Препоръчителен период на държане: **5 години** за фонда"),
            "Препоръчителен период на държане: 5 години за фонда",
        )

    def test_clean_markdown_when_underscore_is_inside_a_word_then_it_is_kept(self):
        self.assertEqual(
            clean_markdown("see file fund_report_2025.pdf"),
            "see file fund_report_2025.pdf",
        )

    def test_clean_markdown_when_asterisk_marks_a_footnote_then_it_is_kept(self):
        self.assertEqual(
            clean_markdown("_*Вижте речник на термините_"), "*Вижте речник на термините"
        )


class TestCleanMarkdownStructure(unittest.TestCase):
    def test_clean_markdown_when_line_is_a_heading_then_heading_marker_is_kept(self):
        self.assertEqual(clean_markdown("### **Цел**"), "### Цел")

    def test_clean_markdown_when_heading_is_bold_and_underlined_then_only_heading_marker_remains(
        self,
    ):
        self.assertEqual(clean_markdown("# **<u>Резюме</u>**"), "# Резюме")

    def test_clean_markdown_when_text_has_paragraph_breaks_then_they_are_kept(self):
        self.assertEqual(
            clean_markdown("Първи абзац.\n\nВтори абзац."),
            "Първи абзац.\n\nВтори абзац.",
        )

    def test_clean_markdown_when_cell_contains_br_then_it_is_replaced_by_a_space(self):
        self.assertEqual(clean_markdown("|10 350 EUR<br>3,50%|"), "|10 350 EUR 3,50%|")

    def test_clean_markdown_when_line_is_a_table_row_then_it_stays_a_single_row_with_all_cells(
        self,
    ):
        row = "|**Умерен**|**Какво бихте получили**<br>Средногодишна възвръщаемост|**10 350 EUR**<br>3,50%|"

        self.assertEqual(
            clean_markdown(row),
            "|Умерен|Какво бихте получили Средногодишна възвръщаемост|10 350 EUR 3,50%|",
        )

    def test_clean_markdown_when_line_is_a_table_separator_then_it_is_kept(self):
        self.assertEqual(clean_markdown("|---|---|---|"), "|---|---|---|")

    def test_clean_markdown_when_text_is_wrapped_in_picture_markers_then_markers_are_removed_and_text_is_kept(
        self,
    ):
        text = "<!-- Start of picture text -->\nISIN код: BG9000005223<!-- End of picture text -->"

        cleaned = clean_markdown(text)

        self.assertIn("ISIN код: BG9000005223", cleaned)
        self.assertNotIn("<!--", cleaned)
        self.assertNotIn("picture text", cleaned)


class TestCleanMarkdownHomoglyphs(unittest.TestCase):
    # Inputs contain a Latin look-alike letter (noted in each comment); expected outputs are pure Cyrillic.

    def test_clean_markdown_when_cyrillic_word_starts_with_latin_uppercase_lookalike_then_it_becomes_cyrillic(
        self,
    ):
        self.assertEqual(clean_markdown("Tози документ"), "Този документ")  # Latin T

    def test_clean_markdown_when_cyrillic_word_contains_latin_lowercase_lookalike_then_it_becomes_cyrillic(
        self,
    ):
        self.assertEqual(
            clean_markdown("бъдещoто представяне"), "бъдещото представяне"
        )  # Latin o

    def test_clean_markdown_when_cyrillic_word_ends_with_latin_lookalike_then_it_becomes_cyrillic(
        self,
    ):
        self.assertEqual(
            clean_markdown("на дистрибуторa"), "на дистрибутора"
        )  # Latin a

    def test_clean_markdown_when_lookalike_word_is_in_a_sentence_then_only_that_word_changes(
        self,
    ):
        self.assertEqual(clean_markdown("Валута: eвро"), "Валута: евро")  # Latin e

    def test_clean_markdown_when_lookalike_word_is_inside_a_table_cell_then_it_becomes_cyrillic(
        self,
    ):
        self.assertEqual(
            clean_markdown("|Mакс. входна такса|3%|"), "|Макс. входна такса|3%|"
        )  # Latin M

    def test_clean_markdown_when_word_is_english_then_it_is_unchanged(self):
        self.assertEqual(
            clean_markdown("Optimum Fund Prospectus"), "Optimum Fund Prospectus"
        )

    def test_clean_markdown_when_english_name_is_inside_bulgarian_text_then_it_is_unchanged(
        self,
    ):
        text = "Кей Би Си Асет Мениджмънт Н.В. (KBC Asset Management N.V.)"

        self.assertEqual(clean_markdown(text), text)

    def test_clean_markdown_when_token_is_an_isin_next_to_cyrillic_then_it_is_unchanged(
        self,
    ):
        self.assertEqual(
            clean_markdown("ISIN-код: BE6316179249"), "ISIN-код: BE6316179249"
        )


class TestCleanMarkdownWhitespace(unittest.TestCase):
    def test_clean_markdown_when_line_has_trailing_spaces_then_they_are_removed(self):
        self.assertEqual(
            clean_markdown("Резюме \nПодфондът насърчава "),
            "Резюме\nПодфондът насърчава",
        )

    def test_clean_markdown_when_words_are_separated_by_multiple_spaces_then_a_single_space_remains(
        self,
    ):
        self.assertEqual(
            clean_markdown("as stated in the  acounts of  the Bevek"),
            "as stated in the acounts of the Bevek",
        )

    def test_clean_markdown_when_heading_has_multiple_spaces_then_heading_marker_and_single_spaces_remain(
        self,
    ):
        self.assertEqual(
            clean_markdown("## 1.    Basic details"), "## 1. Basic details"
        )

    def test_clean_markdown_when_text_has_three_or_more_newlines_then_a_single_blank_line_remains(
        self,
    ):
        self.assertEqual(
            clean_markdown("Първи абзац.\n\n\n\nВтори абзац."),
            "Първи абзац.\n\nВтори абзац.",
        )

    def test_clean_markdown_when_text_has_leading_and_trailing_blank_lines_then_they_are_stripped(
        self,
    ):
        self.assertEqual(
            clean_markdown("\n\nТекст на страницата.\n\n"), "Текст на страницата."
        )

    def test_clean_markdown_when_text_has_a_non_breaking_space_then_it_becomes_a_regular_space(
        self,
    ):
        self.assertEqual(clean_markdown("10 350 EUR"), "10 350 EUR")

    def test_clean_markdown_when_lines_are_separated_by_a_single_newline_then_it_is_kept(
        self,
    ):
        self.assertEqual(
            clean_markdown("Година на стартиране: 2022\nВалута: евро"),
            "Година на стартиране: 2022\nВалута: евро",
        )


class TestMatchKey(unittest.TestCase):
    def test_match_key_when_fund_names_differ_only_in_casing_then_keys_are_equal(self):
        self.assertEqual(match_key("ОББ ЕкспертИйз"), match_key("ОББ Експертийз"))

    def test_match_key_when_text_is_all_uppercase_then_key_equals_lowercase_variant(
        self,
    ):
        self.assertEqual(match_key("OPTIMUM FUND"), match_key("optimum fund"))

    def test_match_key_when_cyrillic_word_contains_latin_lookalike_then_key_equals_pure_cyrillic_variant(
        self,
    ):
        self.assertEqual(match_key("Дефанзивeн"), match_key("Дефанзивен"))  # Latin e

    def test_match_key_when_fund_name_is_wrapped_in_quotes_then_key_equals_unquoted_variant(
        self,
    ):
        self.assertEqual(
            match_key("„ОББ ЕкспертИйз Дефанзивен Балансиран Отговорно Инвестиращ“"),
            match_key("ОББ ЕкспертИйз Дефанзивен Балансиран Отговорно Инвестиращ"),
        )

    def test_match_key_when_name_has_line_breaks_or_repeated_spaces_then_key_equals_single_spaced_variant(
        self,
    ):
        self.assertEqual(
            match_key("Enhanced Intelligence Global Allocation Responsible\nInvesting"),
            match_key("Enhanced  Intelligence Global Allocation Responsible Investing"),
        )

    def test_match_key_when_fund_names_differ_by_one_word_then_keys_are_different(self):
        self.assertNotEqual(
            match_key("ОББ ЕкспертИйз Дефанзивен Консервативен Отговорно Инвестиращ"),
            match_key("ОББ ЕкспертИйз Дефанзивен Балансиран Отговорно Инвестиращ"),
        )
