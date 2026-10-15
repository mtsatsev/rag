import unittest

from rag.models import Fund, FundQuery, FundRef, FundRegistry, Umbrella
from rag.retrieve.resolver import FundResolver

ENHANCED = "Enhanced Intelligence Global Allocation Responsible Investing"
CONSERVATIVE = "ОББ ЕкспертИйз Дефанзивен Консервативен Отговорно Инвестиращ"
BALANCED = "ОББ ЕкспертИйз Дефанзивен Балансиран Отговорно Инвестиращ"

REGISTRY = FundRegistry(
    umbrellas=[
        Umbrella(name="Optimum Fund", aliases=["Optimum Fund"]),
        Umbrella(name="ОББ ЕкспертИйз", aliases=["ОББ ЕкспертИйз", "UBB Expertiz"]),
    ],
    funds=[
        Fund(
            name=ENHANCED,
            umbrella="Optimum Fund",
            isin="BE6316179249",
            aliases=[f"Optimum Fund {ENHANCED}"],
        ),
        Fund(name=CONSERVATIVE, umbrella="ОББ ЕкспертИйз"),
        Fund(name=BALANCED, umbrella="ОББ ЕкспертИйз", isin="BG9000005223"),
    ],
    documents={},
)
KNOWN_FUNDS = [
    (ENHANCED, "Optimum Fund"),
    ("ČSOB Lookback 3", "Optimum Fund"),
    ("ČSOB Lookback 4", "Optimum Fund"),
]


class TestFundResolverSingleFund(unittest.TestCase):
    def setUp(self):
        self.resolver = FundResolver(REGISTRY, KNOWN_FUNDS)

    def test_resolve_when_question_contains_the_exact_fund_name_then_returns_that_fund_with_its_umbrella(
        self,
    ):
        query = self.resolver.resolve(f"What are the entry costs of {ENHANCED}?")

        self.assertEqual(
            query, FundQuery(funds=[FundRef(fund=ENHANCED, umbrella="Optimum Fund")])
        )

    def test_resolve_when_bulgarian_fund_name_is_in_an_english_question_then_returns_that_fund(
        self,
    ):
        query = self.resolver.resolve(
            f"What is the investment objective of {CONSERVATIVE}?"
        )

        self.assertEqual(
            query.funds, [FundRef(fund=CONSERVATIVE, umbrella="ОББ ЕкспертИйз")]
        )

    def test_resolve_when_fund_name_is_written_in_different_casing_then_returns_the_canonical_name(
        self,
    ):
        query = self.resolver.resolve(
            "Каква е целта на ОББ Експертийз Дефанзивен Консервативен Отговорно Инвестиращ?"
        )

        self.assertEqual([ref.fund for ref in query.funds], [CONSERVATIVE])

    def test_resolve_when_fund_name_contains_a_latin_lookalike_then_returns_the_canonical_name(
        self,
    ):
        query = self.resolver.resolve(
            "Каква е целта на ОББ ЕкспертИйз Дефанзивeн Консервативен Отговорно Инвестиращ?"
        )  # Latin e

        self.assertEqual([ref.fund for ref in query.funds], [CONSERVATIVE])

    def test_resolve_when_fund_name_is_wrapped_in_quotes_then_returns_that_fund(self):
        query = self.resolver.resolve(f"Какво е представянето на „{BALANCED}“?")

        self.assertEqual([ref.fund for ref in query.funds], [BALANCED])

    def test_resolve_when_question_contains_an_isin_then_returns_the_fund_with_that_isin(
        self,
    ):
        query = self.resolver.resolve(
            "What was the performance of BG9000005223 in 2024?"
        )

        self.assertEqual([ref.fund for ref in query.funds], [BALANCED])

    def test_resolve_when_question_uses_a_registered_alias_then_returns_the_canonical_fund(
        self,
    ):
        query = self.resolver.resolve(f"Describe the risks of Optimum Fund {ENHANCED}.")

        self.assertEqual(query.funds, [FundRef(fund=ENHANCED, umbrella="Optimum Fund")])

    def test_resolve_when_fund_is_only_known_from_the_index_then_returns_it_with_its_umbrella(
        self,
    ):
        query = self.resolver.resolve("What is the maturity date of ČSOB Lookback 3?")

        self.assertEqual(
            query.funds, [FundRef(fund="ČSOB Lookback 3", umbrella="Optimum Fund")]
        )


class TestFundResolverOverlappingNames(unittest.TestCase):
    def setUp(self):
        self.resolver = FundResolver(REGISTRY, KNOWN_FUNDS)

    def test_resolve_when_fund_name_contains_the_umbrella_name_then_umbrella_is_not_reported_separately(
        self,
    ):
        query = self.resolver.resolve(
            f"What is the investment objective of {CONSERVATIVE}?"
        )

        self.assertEqual(query.umbrellas, [])

    def test_resolve_when_alias_contains_the_umbrella_name_then_umbrella_is_not_reported_separately(
        self,
    ):
        query = self.resolver.resolve(f"Describe the risks of Optimum Fund {ENHANCED}.")

        self.assertEqual(query.umbrellas, [])

    def test_resolve_when_fund_names_differ_by_one_word_then_only_the_mentioned_fund_is_returned(
        self,
    ):
        query = self.resolver.resolve(f"Какво е представянето на {BALANCED}?")

        self.assertEqual([ref.fund for ref in query.funds], [BALANCED])

    def test_resolve_when_a_longer_number_follows_the_fund_name_then_the_shorter_fund_does_not_match(
        self,
    ):
        query = self.resolver.resolve("What is the maturity date of ČSOB Lookback 30?")

        self.assertEqual(query.funds, [])


class TestFundResolverUmbrellasAndSeveralFunds(unittest.TestCase):
    def setUp(self):
        self.resolver = FundResolver(REGISTRY, KNOWN_FUNDS)

    def test_resolve_when_only_an_umbrella_is_named_then_returns_it_without_funds(self):
        query = self.resolver.resolve("Who is the auditor of Optimum Fund?")

        self.assertEqual(query, FundQuery(umbrellas=["Optimum Fund"]))

    def test_resolve_when_umbrella_is_named_by_an_alias_then_returns_the_canonical_umbrella(
        self,
    ):
        query = self.resolver.resolve("Who manages UBB Expertiz?")

        self.assertEqual(query.umbrellas, ["ОББ ЕкспертИйз"])

    def test_resolve_when_two_funds_are_named_then_both_are_returned(self):
        query = self.resolver.resolve(
            "Compare the costs of ČSOB Lookback 3 and ČSOB Lookback 4."
        )

        self.assertEqual(
            {ref.fund for ref in query.funds}, {"ČSOB Lookback 3", "ČSOB Lookback 4"}
        )

    def test_resolve_when_the_same_fund_is_named_twice_then_it_is_returned_once(self):
        query = self.resolver.resolve(
            f"Is {ENHANCED} riskier than the average? {ENHANCED} invests in equities."
        )

        self.assertEqual(len(query.funds), 1)

    def test_resolve_when_no_fund_or_umbrella_is_named_then_returns_an_empty_query(
        self,
    ):
        query = self.resolver.resolve("Can I transfer my fund units to another person?")

        self.assertEqual(query, FundQuery())
