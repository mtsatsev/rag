import tempfile
import unittest
from pathlib import Path

import pymupdf

from rag.index.store import ChunkStore
from rag.ingest.runner import ingest
from rag.models import DocumentEntry, Fund, FundRegistry, Umbrella
from tests.fakes import FakeEmbedder

REGISTRY = FundRegistry(
    umbrellas=[Umbrella(name="Optimum Fund")],
    funds=[Fund(name="Alpha", umbrella="Optimum Fund")],
    documents={
        "kid.pdf": DocumentEntry(doc_type="kid", fund="Alpha"),
        "rights.pdf": DocumentEntry(doc_type="investor_rights"),
    },
)


def write_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    document.new_page().insert_textbox(pymupdf.Rect(72, 200, 520, 500), f"{text} " * 3)
    document.save(path)
    document.close()


class IngestTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.data_dir = root / "data"
        self.data_dir.mkdir()
        write_pdf(
            self.data_dir / "kid.pdf",
            "Entry costs are up to 3 percent of the amount invested.",
        )
        write_pdf(
            self.data_dir / "rights.pdf",
            "Every investor may subscribe to units of the fund.",
        )
        self.embedder = FakeEmbedder()
        self.store = ChunkStore(root / "chroma", self.embedder)

    def tearDown(self):
        self.directory.cleanup()

    def chunk_texts(self, doc_id: str) -> str:
        return " ".join(
            chunk.text
            for chunk in self.store.all_chunks()
            if chunk.metadata.doc_id == doc_id
        )


class TestIngestFirstRun(IngestTestCase):
    def test_ingest_when_store_is_empty_then_every_pdf_is_indexed(self):
        report = ingest(self.data_dir, self.store, REGISTRY)

        self.assertEqual(sorted(report.indexed), ["kid", "rights"])
        self.assertEqual(self.store.document_ids, {"kid", "rights"})

    def test_ingest_when_document_is_listed_with_a_fund_then_its_chunks_get_that_fund(
        self,
    ):
        ingest(self.data_dir, self.store, REGISTRY)

        funds = {
            chunk.metadata.fund
            for chunk in self.store.all_chunks()
            if chunk.metadata.doc_id == "kid"
        }

        self.assertEqual(funds, {"Alpha"})

    def test_ingest_when_folder_contains_non_pdf_files_then_they_are_ignored(self):
        (self.data_dir / "notes.txt").write_text("not a document")

        report = ingest(self.data_dir, self.store, REGISTRY)

        self.assertEqual(sorted(report.indexed), ["kid", "rights"])

    def test_ingest_when_pdf_is_not_listed_in_the_registry_then_raises_key_error_mentioning_funds_yaml(
        self,
    ):
        write_pdf(self.data_dir / "unknown.pdf", "A document nobody registered.")

        with self.assertRaisesRegex(KeyError, "funds.yaml"):
            ingest(self.data_dir, self.store, REGISTRY)


class TestIngestRerun(IngestTestCase):
    def test_ingest_when_nothing_changed_since_the_last_run_then_every_pdf_is_skipped(
        self,
    ):
        ingest(self.data_dir, self.store, REGISTRY)

        report = ingest(self.data_dir, self.store, REGISTRY)

        self.assertEqual(
            (report.indexed, sorted(report.skipped), report.removed),
            ([], ["kid", "rights"], []),
        )

    def test_ingest_when_nothing_changed_since_the_last_run_then_nothing_is_embedded_again(
        self,
    ):
        ingest(self.data_dir, self.store, REGISTRY)
        embedded_after_first_run = len(self.embedder.embedded)

        ingest(self.data_dir, self.store, REGISTRY)

        self.assertEqual(len(self.embedder.embedded), embedded_after_first_run)

    def test_ingest_when_a_pdf_is_replaced_then_only_that_pdf_is_reindexed(self):
        ingest(self.data_dir, self.store, REGISTRY)
        write_pdf(
            self.data_dir / "kid.pdf",
            "Entry costs are now up to 2 percent of the amount invested.",
        )

        report = ingest(self.data_dir, self.store, REGISTRY)

        self.assertEqual((report.indexed, report.skipped), (["kid"], ["rights"]))

    def test_ingest_when_a_pdf_is_replaced_then_its_old_text_is_gone_and_the_new_text_is_stored(
        self,
    ):
        ingest(self.data_dir, self.store, REGISTRY)
        write_pdf(
            self.data_dir / "kid.pdf",
            "Entry costs are now up to 2 percent of the amount invested.",
        )

        ingest(self.data_dir, self.store, REGISTRY)

        self.assertIn("2 percent", self.chunk_texts("kid"))
        self.assertNotIn("3 percent", self.chunk_texts("kid"))

    def test_ingest_when_a_pdf_is_removed_from_the_folder_then_its_chunks_are_removed(
        self,
    ):
        ingest(self.data_dir, self.store, REGISTRY)
        (self.data_dir / "rights.pdf").unlink()

        report = ingest(self.data_dir, self.store, REGISTRY)

        self.assertEqual(report.removed, ["rights"])
        self.assertEqual(self.store.document_ids, {"kid"})
