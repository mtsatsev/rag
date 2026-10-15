import unittest

from rag.ingest.clean import clean_markdown


class TestCleanMarkdownEmphasis(unittest.TestCase):
    def test_clean_markdown_when_text_is_bold_then_markers_are_removed_and_text_is_kept(self):
        self.assertEqual(clean_markdown("**Умерен**"), "Умерен")

    def test_clean_markdown_when_text_is_italic_then_markers_are_removed_and_text_is_kept(self):
        self.assertEqual(clean_markdown("_Executive Director_"), "Executive Director")

    def test_clean_markdown_when_text_is_bold_and_italic_then_all_markers_are_removed(self):
        self.assertEqual(clean_markdown("**_Date of incorporation_**"), "Date of incorporation")

    def test_clean_markdown_when_text_is_underlined_then_tags_are_removed_and_text_is_kept(self):
        self.assertEqual(clean_markdown("<u>www.kbc.be/kid.</u>"), "www.kbc.be/kid.")

    def test_clean_markdown_when_emphasis_is_inside_a_sentence_then_surrounding_text_is_kept(self):
        self.assertEqual(
            clean_markdown("Препоръчителен период на държане: **5 години** за фонда"),
            "Препоръчителен период на държане: 5 години за фонда",
        )

    def test_clean_markdown_when_underscore_is_inside_a_word_then_it_is_kept(self):
        self.assertEqual(clean_markdown("see file fund_report_2025.pdf"), "see file fund_report_2025.pdf")

    def test_clean_markdown_when_asterisk_marks_a_footnote_then_it_is_kept(self):
        self.assertEqual(clean_markdown("_*Вижте речник на термините_"), "*Вижте речник на термините")


class TestCleanMarkdownStructure(unittest.TestCase):
    def test_clean_markdown_when_line_is_a_heading_then_heading_marker_is_kept(self):
        self.assertEqual(clean_markdown("### **Цел**"), "### Цел")

    def test_clean_markdown_when_heading_is_bold_and_underlined_then_only_heading_marker_remains(self):
        self.assertEqual(clean_markdown("# **<u>Резюме</u>**"), "# Резюме")

    def test_clean_markdown_when_text_has_paragraph_breaks_then_they_are_kept(self):
        self.assertEqual(clean_markdown("Първи абзац.\n\nВтори абзац."), "Първи абзац.\n\nВтори абзац.")

    def test_clean_markdown_when_cell_contains_br_then_it_is_replaced_by_a_space(self):
        self.assertEqual(clean_markdown("|10 350 EUR<br>3,50%|"), "|10 350 EUR 3,50%|")

    def test_clean_markdown_when_line_is_a_table_row_then_it_stays_a_single_row_with_all_cells(self):
        row = "|**Умерен**|**Какво бихте получили**<br>Средногодишна възвръщаемост|**10 350 EUR**<br>3,50%|"

        self.assertEqual(
            clean_markdown(row),
            "|Умерен|Какво бихте получили Средногодишна възвръщаемост|10 350 EUR 3,50%|",
        )

    def test_clean_markdown_when_line_is_a_table_separator_then_it_is_kept(self):
        self.assertEqual(clean_markdown("|---|---|---|"), "|---|---|---|")

    def test_clean_markdown_when_text_is_wrapped_in_picture_markers_then_markers_are_removed_and_text_is_kept(self):
        text = "<!-- Start of picture text -->\nISIN код: BG9000005223<!-- End of picture text -->"

        cleaned = clean_markdown(text)

        self.assertIn("ISIN код: BG9000005223", cleaned)
        self.assertNotIn("<!--", cleaned)
        self.assertNotIn("picture text", cleaned)