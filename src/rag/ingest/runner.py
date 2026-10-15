from pathlib import Path

from rag.index.store import ChunkStore
from rag.models import FundRegistry, IngestReport


def ingest(data_dir: Path, store: ChunkStore, registry: FundRegistry) -> IngestReport:
    raise NotImplementedError
