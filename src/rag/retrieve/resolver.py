from collections.abc import Iterable

from rag.models import FundQuery, FundRegistry


class FundResolver:
    def __init__(
        self, registry: FundRegistry, known_funds: Iterable[tuple[str, str | None]] = ()
    ) -> None:
        """`known_funds` are (fund, umbrella) pairs found in the index, e.g. prospectus sub-funds."""
        raise NotImplementedError

    def resolve(self, question: str) -> FundQuery:
        raise NotImplementedError
