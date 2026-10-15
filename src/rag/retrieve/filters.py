from typing import Any

from rag.models import ChunkMetadata, FundQuery

NO_VALUE = ""  # how the store saves None in Chroma metadata


def to_where(fund_query: FundQuery) -> dict[str, Any] | None:
    """Chroma `where` filter for the chunks a question may be answered from; None = no filter."""
    if not fund_query.funds and not fund_query.umbrellas:
        return None
    funds, fund_umbrellas = _funds_and_their_umbrellas(fund_query)
    conditions: list[dict[str, Any]] = [
        {"$and": [{"fund": NO_VALUE}, {"umbrella_fund": NO_VALUE}]}  # manager-wide
    ]
    if funds:
        conditions.append({"fund": {"$in": funds}})
    if fund_umbrellas:
        conditions.append(
            {"$and": [{"fund": NO_VALUE}, {"umbrella_fund": {"$in": fund_umbrellas}}]}
        )
    if fund_query.umbrellas:
        conditions.append({"umbrella_fund": {"$in": fund_query.umbrellas}})
    return {"$or": conditions}


def matches(metadata: ChunkMetadata, fund_query: FundQuery) -> bool:
    """Same rule as `to_where`, as a Python check (used by keyword search)."""
    if not fund_query.funds and not fund_query.umbrellas:
        return True
    funds, fund_umbrellas = _funds_and_their_umbrellas(fund_query)
    fund, umbrella = metadata.fund, metadata.umbrella_fund
    return (
        (fund is None and umbrella is None)
        or fund in funds
        or (fund is None and umbrella in fund_umbrellas)
        or umbrella in fund_query.umbrellas
    )


def _funds_and_their_umbrellas(fund_query: FundQuery) -> tuple[list[str], list[str]]:
    funds = [ref.fund for ref in fund_query.funds]
    umbrellas = sorted({ref.umbrella for ref in fund_query.funds if ref.umbrella})
    return funds, umbrellas
