import functools
import re
from collections.abc import Callable
from pathlib import Path

import yaml

from rag.ingest.clean import match_key
from rag.models import DocumentInfo, FundRegistry

FUNDS_YAML = Path(__file__).parents[3] / "config" / "funds.yaml"
SUB_FUND_HEADING = re.compile(r"^#{1,6} Information concerning the sub-fund (.+)$")
ANNEX_TITLE = re.compile(r"^Annex for (.+)$")


def load_registry(path: Path = FUNDS_YAML) -> FundRegistry:
    return FundRegistry.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def document_info(registry: FundRegistry, file_name: str) -> DocumentInfo:
    entry = registry.documents.get(file_name)
    if entry is None:
        raise KeyError(f"{file_name} is not listed in funds.yaml")
    umbrella = entry.umbrella
    if entry.fund is not None:
        umbrella = next(
            fund.umbrella for fund in registry.funds if fund.name == entry.fund
        )
    return DocumentInfo(
        doc_type=entry.doc_type, fund=entry.fund, umbrella_fund=umbrella
    )


def fund_detector(
    registry: FundRegistry, file_name: str
) -> Callable[[str], str | None] | None:
    if registry.documents[file_name].multi_fund:
        return functools.partial(detect_fund, registry)
    return None


def detect_fund(registry: FundRegistry, paragraph: str) -> str | None:
    match = SUB_FUND_HEADING.match(paragraph) or ANNEX_TITLE.match(paragraph)
    if match is None:
        return None
    name = match.group(1)
    for umbrella in registry.umbrellas:
        name = name.removeprefix(f"{umbrella.name} ")
    canonical = {
        match_key(alias): fund.name
        for fund in registry.funds
        for alias in (fund.name, *fund.aliases)
    }
    return canonical.get(match_key(name), name)
