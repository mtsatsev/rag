import unittest

from rag.ingest.parse import file_hash, parse_pdf
from tests.documents import KID_PDF, PROSPECTUS_PDF, RIGHTS_PDF, require_pdf


class TestParsePdfDocumentIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rights_path = require_pdf(RIGHTS_PDF)
        cls.kid_path = require_pdf(KID_PDF)
        cls.rights = parse_pdf(cls.rights_path)

    def test_parse_pdf_when_document_is_parsed_then_doc_id_is_the_file_stem(self):
        self.assertEqual(self.rights.doc_id, "11074")

    def test_parse_pdf_when_document_is_parsed_then_file_name_is_kept(self):
        self.assertEqual(self.rights.file_name, "11074.pdf")

    def test_parse_pdf_when_same_file_is_parsed_twice_then_content_hash_is_equal(self):
        self.assertEqual(
            parse_pdf(self.rights_path).content_hash, self.rights.content_hash
        )

    def test_parse_pdf_when_files_differ_then_content_hashes_differ(self):
        self.assertNotEqual(
            parse_pdf(self.kid_path).content_hash, self.rights.content_hash
        )


class TestParsePdfMissingFile(unittest.TestCase):
    def test_parse_pdf_when_path_does_not_exist_then_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_pdf(require_pdf(RIGHTS_PDF).with_name("does_not_exist.pdf"))


class TestParsePdfPages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kid = parse_pdf(require_pdf(KID_PDF))
        cls.prospectus_path = require_pdf(PROSPECTUS_PDF)

    def test_parse_pdf_when_document_has_multiple_pages_then_returns_one_page_per_pdf_page_in_order(
        self,
    ):
        self.assertEqual([page.page_number for page in self.kid.pages], [1, 2, 3, 4])

    def test_parse_pdf_when_pdf_has_a_text_layer_then_every_page_has_text(self):
        for page in self.kid.pages:
            with self.subTest(page=page.page_number):
                self.assertTrue(page.markdown.strip())

    def test_parse_pdf_when_only_selected_pages_are_parsed_then_original_page_numbers_are_kept(
        self,
    ):
        doc = parse_pdf(self.prospectus_path, pages=[4, 25])

        self.assertEqual([page.page_number for page in doc.pages], [4, 25])
        self.assertIn("Information concerning the sub-fund", doc.pages[1].markdown)


class TestParsePdfPlainText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rights = parse_pdf(require_pdf(RIGHTS_PDF))

    def test_parse_pdf_when_page_is_plain_cyrillic_text_then_text_is_extracted_intact(
        self,
    ):
        self.assertIn(
            "Всеки инвеститор може да запише дялове във фонд",
            self.rights.pages[0].markdown,
        )


class TestParsePdfTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kid = parse_pdf(require_pdf(KID_PDF))

    def test_parse_pdf_when_page_contains_a_table_then_it_is_rendered_as_markdown_rows(
        self,
    ):
        table_rows = [
            line
            for line in self.kid.pages[1].markdown.splitlines()
            if line.startswith("|")
        ]

        self.assertTrue(table_rows)

    def test_parse_pdf_when_page_contains_a_table_then_cells_of_a_row_stay_on_the_same_line(
        self,
    ):
        table_rows = [
            line
            for line in self.kid.pages[1].markdown.splitlines()
            if line.startswith("|")
        ]
        moderate_row = next((row for row in table_rows if "Умерен" in row), None)

        self.assertIsNotNone(moderate_row)
        self.assertIn("10 350 EUR", moderate_row)
        self.assertIn("12 270 EUR", moderate_row)


class TestParsePdfHeadings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kid = parse_pdf(require_pdf(KID_PDF))

    def test_parse_pdf_when_page_has_titles_then_they_are_marked_as_markdown_headings(
        self,
    ):
        headings = [
            line
            for line in self.kid.pages[0].markdown.splitlines()
            if line.startswith("#")
        ]

        self.assertTrue(any("Какъв е този продукт?" in heading for heading in headings))


class TestParsePdfPageFurniture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = parse_pdf(require_pdf(PROSPECTUS_PDF), pages=[4]).pages[0].markdown

    def test_parse_pdf_when_page_has_a_running_header_then_header_is_dropped(self):
        self.assertNotIn("Optimum Fund Ref.", self.page)

    def test_parse_pdf_when_page_has_a_page_number_footer_then_page_number_is_dropped(
        self,
    ):
        self.assertNotEqual(self.page.rstrip().splitlines()[-1].strip(), "4")

    def test_parse_pdf_when_header_and_footer_are_dropped_then_body_text_is_kept(self):
        self.assertIn("The issued capital amounts to 35.754.192 euros", self.page)


class TestFileHash(unittest.TestCase):
    def test_file_hash_when_same_file_is_hashed_twice_then_hashes_are_equal(self):
        path = require_pdf(RIGHTS_PDF)

        self.assertEqual(file_hash(path), file_hash(path))

    def test_file_hash_when_files_differ_then_hashes_differ(self):
        self.assertNotEqual(
            file_hash(require_pdf(RIGHTS_PDF)), file_hash(require_pdf(KID_PDF))
        )

    def test_file_hash_when_file_is_parsed_then_it_equals_the_parsed_content_hash(self):
        path = require_pdf(RIGHTS_PDF)

        self.assertEqual(file_hash(path), parse_pdf(path).content_hash)
