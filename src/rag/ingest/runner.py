from pathlib import Path

from rag.index.store import ChunkStore
from rag.ingest.chunk import chunk_document
from rag.ingest.funds import document_info, fund_detector
from rag.ingest.parse import file_hash, parse_pdf
from rag.models import FundRegistry, IngestReport


def ingest(data_dir: Path, store: ChunkStore, registry: FundRegistry) -> IngestReport:
    report = IngestReport()
    pdfs = sorted(data_dir.glob("*.pdf"))
    for pdf in pdfs:
        if store.has_document(pdf.stem, file_hash(pdf)):
            report.skipped.append(pdf.stem)
            continue
        chunks = chunk_document(
            parse_pdf(pdf),
            document_info(registry, pdf.name),
            detect_fund=fund_detector(registry, pdf.name),
        )
        store.replace_document(chunks)
        report.indexed.append(pdf.stem)
    for doc_id in sorted(store.document_ids - {pdf.stem for pdf in pdfs}):
        store.delete_document(doc_id)
        report.removed.append(doc_id)
    return report
