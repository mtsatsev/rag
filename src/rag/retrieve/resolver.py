from collections.abc import Iterable

from rag.ingest.clean import match_key
from rag.models import FundQuery, FundRef, FundRegistry


class FundResolver:
    def __init__(
        self, registry: FundRegistry, known_funds: Iterable[tuple[str, str | None]] = ()
    ) -> None:
        names: list[tuple[str, FundRef | str]] = []
        for fund in registry.funds:
            ref = FundRef(fund=fund.name, umbrella=fund.umbrella)
            for alias in (
                fund.name,
                *fund.aliases,
                *([fund.isin] if fund.isin else []),
            ):
                names.append((match_key(alias), ref))
        for fund, umbrella in known_funds:
            names.append((match_key(fund), FundRef(fund=fund, umbrella=umbrella)))
        for umbrella in registry.umbrellas:
            for alias in (umbrella.name, *umbrella.aliases):
                names.append((match_key(alias), umbrella.name))
        self.names = sorted(names, key=lambda item: len(item[0]), reverse=True)

    def resolve(self, question: str) -> FundQuery:
        text = f" {match_key(question)} "
        funds: list[FundRef] = []
        umbrellas: list[str] = []
        for key, target in self.names:
            needle = f" {key} "
            if not key or needle not in text:
                continue
            text = text.replace(needle, " ")
            if isinstance(target, FundRef):
                if target not in funds:
                    funds.append(target)
            elif target not in umbrellas:
                umbrellas.append(target)
        return FundQuery(funds=funds, umbrellas=umbrellas)
