from collections.abc import Callable
from pathlib import Path

from rag.models import DocumentInfo, FundRegistry

FUNDS_YAML = Path(__file__).parents[3] / "config" / "funds.yaml"


def load_registry(path: Path = FUNDS_YAML) -> FundRegistry:
    raise NotImplementedError


def document_info(registry: FundRegistry, file_name: str) -> DocumentInfo:
    raise NotImplementedError


def fund_detector(
    registry: FundRegistry, file_name: str
) -> Callable[[str], str | None] | None:
    raise NotImplementedError


def detect_fund(registry: FundRegistry, paragraph: str) -> str | None:
    raise NotImplementedError
