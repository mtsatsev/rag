import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from rag.ingest.funds import (
    FUNDS_YAML,
    detect_fund,
    document_info,
    fund_detector,
    load_registry,
)
from rag.models import DocumentEntry, Fund, FundRegistry, Umbrella
from tests.documents import DATA_DIR

ENHANCED = "Enhanced Intelligence Global Allocation Responsible Investing"

REGISTRY = FundRegistry(
    umbrellas=[Umbrella(name="Optimum Fund"), Umbrella(name="ОББ ЕкспертИйз")],
    funds=[
        Fund(name=ENHANCED, umbrella="Optimum Fund", isin="BE6316179249"),
        Fund(
            name="ОББ ЕкспертИйз Дефанзивен Консервативен Отговорно Инвестиращ",
            umbrella="ОББ ЕкспертИйз",
        ),
    ],
    documents={
        "prospectus.pdf": DocumentEntry(
            doc_type="prospectus", umbrella="Optimum Fund", multi_fund=True
        ),
        "kid.pdf": DocumentEntry(doc_type="kid", fund=ENHANCED),
        "rights.pdf": DocumentEntry(doc_type="investor_rights"),
    },
)

REGISTRY_YAML = f"""
umbrellas:
  - name: Optimum Fund
    aliases: [Optimum Fund]
funds:
  - name: {ENHANCED}
    umbrella: Optimum Fund
    isin: BE6316179249
documents:
  kid.pdf:
    doc_type: kid
    fund: {ENHANCED}
"""


def write_yaml(text: str) -> Path:
    file = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    file.write(text)
    file.close()
    return Path(file.name)


class TestLoadRegistry(unittest.TestCase):
    def test_load_registry_when_yaml_is_valid_then_umbrellas_funds_and_documents_are_loaded(
        self,
    ):
        registry = load_registry(write_yaml(REGISTRY_YAML))

        self.assertEqual(
            [umbrella.name for umbrella in registry.umbrellas], ["Optimum Fund"]
        )
        self.assertEqual([fund.name for fund in registry.funds], [ENHANCED])
        self.assertEqual(registry.documents["kid.pdf"].fund, ENHANCED)

    def test_load_registry_when_document_names_an_unknown_fund_then_raises_validation_error(
        self,
    ):
        with self.assertRaises(ValidationError):
            load_registry(
                write_yaml(
                    REGISTRY_YAML.replace(
                        f"    fund: {ENHANCED}", "    fund: Unknown Fund"
                    )
                )
            )

    def test_load_registry_when_fund_names_an_unknown_umbrella_then_raises_validation_error(
        self,
    ):
        with self.assertRaises(ValidationError):
            load_registry(
                write_yaml(
                    REGISTRY_YAML.replace(
                        "    umbrella: Optimum Fund", "    umbrella: Unknown Umbrella"
                    )
                )
            )

    def test_load_registry_when_document_names_an_unknown_umbrella_then_raises_validation_error(
        self,
    ):
        yaml_text = (
            REGISTRY_YAML
            + "  prospectus.pdf:\n    doc_type: prospectus\n    umbrella: Unknown Umbrella\n"
        )

        with self.assertRaises(ValidationError):
            load_registry(write_yaml(yaml_text))


class TestDocumentInfo(unittest.TestCase):
    def test_document_info_when_document_is_about_one_fund_then_umbrella_is_taken_from_that_fund(
        self,
    ):
        info = document_info(REGISTRY, "kid.pdf")

        self.assertEqual(
            (info.doc_type, info.fund, info.umbrella_fund),
            ("kid", ENHANCED, "Optimum Fund"),
        )

    def test_document_info_when_document_is_umbrella_level_then_fund_is_none_and_umbrella_is_set(
        self,
    ):
        info = document_info(REGISTRY, "prospectus.pdf")

        self.assertEqual(
            (info.doc_type, info.fund, info.umbrella_fund),
            ("prospectus", None, "Optimum Fund"),
        )

    def test_document_info_when_document_is_manager_wide_then_fund_and_umbrella_are_none(
        self,
    ):
        info = document_info(REGISTRY, "rights.pdf")

        self.assertEqual(
            (info.doc_type, info.fund, info.umbrella_fund),
            ("investor_rights", None, None),
        )

    def test_document_info_when_file_is_not_listed_then_raises_key_error_mentioning_funds_yaml(
        self,
    ):
        with self.assertRaisesRegex(KeyError, "funds.yaml"):
            document_info(REGISTRY, "new_document.pdf")


class TestFundDetector(unittest.TestCase):
    def test_fund_detector_when_document_is_multi_fund_then_returns_a_detector(self):
        detector = fund_detector(REGISTRY, "prospectus.pdf")

        self.assertIsNotNone(detector)
        self.assertEqual(
            detector(f"# Information concerning the sub-fund {ENHANCED}"), ENHANCED
        )

    def test_fund_detector_when_document_is_single_fund_then_returns_none(self):
        self.assertIsNone(fund_detector(REGISTRY, "kid.pdf"))


class TestDetectFund(unittest.TestCase):
    def test_detect_fund_when_paragraph_is_a_sub_fund_heading_then_returns_the_fund_name(
        self,
    ):
        self.assertEqual(
            detect_fund(REGISTRY, f"# Information concerning the sub-fund {ENHANCED}"),
            ENHANCED,
        )

    def test_detect_fund_when_paragraph_is_an_annex_title_then_returns_the_name_without_the_umbrella_prefix(
        self,
    ):
        self.assertEqual(
            detect_fund(REGISTRY, f"Annex for Optimum Fund {ENHANCED}"), ENHANCED
        )

    def test_detect_fund_when_registered_fund_is_written_in_different_casing_then_returns_the_canonical_name(
        self,
    ):
        self.assertEqual(
            detect_fund(
                REGISTRY, f"# Information concerning the sub-fund {ENHANCED.upper()}"
            ),
            ENHANCED,
        )

    def test_detect_fund_when_sub_fund_is_not_registered_then_returns_the_name_as_found(
        self,
    ):
        self.assertEqual(
            detect_fund(
                REGISTRY, "# Information concerning the sub-fund ČSOB Lookback 3"
            ),
            "ČSOB Lookback 3",
        )

    def test_detect_fund_when_heading_is_about_the_umbrella_then_returns_none(self):
        self.assertIsNone(detect_fund(REGISTRY, "# Information concerning the Bevek"))

    def test_detect_fund_when_paragraph_is_an_ordinary_heading_then_returns_none(self):
        self.assertIsNone(detect_fund(REGISTRY, "## 1. Basic details"))

    def test_detect_fund_when_sentence_mentions_a_sub_fund_heading_then_returns_none(
        self,
    ):
        paragraph = f"See the heading “Information concerning the sub-fund {ENHANCED}” for details."

        self.assertIsNone(detect_fund(REGISTRY, paragraph))


class TestRealRegistry(unittest.TestCase):
    def test_load_registry_when_real_funds_yaml_is_loaded_then_it_is_valid(self):
        registry = load_registry(FUNDS_YAML)

        self.assertTrue(registry.documents)

    def test_load_registry_when_data_folder_has_pdfs_then_every_pdf_is_listed_in_funds_yaml(
        self,
    ):
        pdfs = sorted(path.name for path in DATA_DIR.glob("*.pdf"))
        if not pdfs:
            self.skipTest(f"no PDFs in {DATA_DIR}")

        registry = load_registry(FUNDS_YAML)

        self.assertEqual([name for name in pdfs if name not in registry.documents], [])
