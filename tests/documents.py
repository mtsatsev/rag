import os
import unittest
from pathlib import Path

DATA_DIR = Path(
    os.environ.get("RAG_DATA_DIR", Path(__file__).parents[1] / "src" / "data")
)

RIGHTS_PDF = "11074.pdf"
KID_PDF = "KID_BE6316179249_BG.pdf"
PROSPECTUS_PDF = "FU_BF065_EN.pdf"


def require_pdf(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        raise unittest.SkipTest(
            f"{name} not found in {DATA_DIR}; see README for where to put the documents"
        )
    return path
