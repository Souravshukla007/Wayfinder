"""Property-based test: every flight and hotel option carries a rationale.

Feature: wayfinder-travel-planner, Property 13: Every flight and hotel option
carries a rationale.

*For any* produced plan, every flight option and every hotel option in the
final plan SHALL have a non-empty rationale field.

Validates: Requirements 1.4, 17.4

The plan is produced by the Coordinator's merge step
(:func:`app.orchestration.nodes.coordinator.merge_final_plan`) — the single
layer that turns the deterministic tools' raw provider options into the
``FlightDTO`` / ``HotelDTO`` entries the user sees, attaching a rationale to
each (Requirements 1.4, 17.4). The decision-transparency guardrail
(Requirement 1.3) means any option for which no rationale can be derived is
*withheld* rather than emitted with a blank rationale, so the invariant we
assert — every surfaced option carries a non-empty rationale — is exactly the
contract the merge must uphold.

The generators deliberately straddle the guardrail: provider ``tag`` values
include normal tags, the empty string, and whitespace-only strings, so the test
exercises both the "rationale derivable -> surfaced with rationale" path and the
"no rationale -> withheld" path. A counterexample here (a surfaced option whose
rationale is blank) would reveal a real decision-transparency bug to report, not
something to paper over in the test.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from app.orchestration.nodes.coordinator import ToolBundle, merge_final_plan
from app.providers.base import FlightOption, HotelOption

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
# A price as a display string mirroring the mock fixtures (e.g. "₹12,000").
# The exact figure is irrelevant to the rationale invariant; we keep it
# digit-bearing so the merge's price parsing stays well-defined.
_price = st.integers(min_value=0, max_value=500_000).map(lambda n: f"\u20b9{n:,}")

# Provider tags span the guardrail boundary: real tags, the empty string, and
# whitespace-only strings (which carry no rationale and must trigger withholding),
# plus free text that may or may not be blank after stripping.
_tag = st.one_of(
    st.sampled_from(
        ["Best balance", "Cheapest", "Shortest", "Best value", "Splurge", "Most convenient"]
    ),
    st.just(""),
    st.just("   "),
    st.text(max_size=24),
)

_flight = st.builds(
    FlightOption,
    airline=st.text(min_size=1, max_size=24),
    price=_price,
    duration=st.text(min_size=1, max_size=12),
    tag=_tag,
)

_hotel = st.builds(
    HotelOption,
    name=st.text(min_size=1, max_size=24),
    rating=st.integers(min_value=0, max_value=5),
    price=_price.map(lambda p: f"{p}/n"),
    distance=st.text(max_size=18),
    tag=_tag,
)


@st.composite
def _tool_bundles(draw: st.DrawFn) -> ToolBundle:
    """Draw a ToolBundle with 0-6 flight and 0-6 hotel options.

    The size ranges include the empty case (no options to surface) and several
    options at once, so the property holds whether the chosen destination has
    many candidate flights/hotels or none.
    """
    flights = draw(st.lists(_flight, min_size=0, max_size=6))
    hotels = draw(st.lists(_hotel, min_size=0, max_size=6))
    return ToolBundle(
        destination=draw(st.text(min_size=1, max_size=16)),
        flights=flights,
        hotels=hotels,
        events=[],
        weather=None,
    )


def _is_blank(text: str) -> bool:
    """A provider tag yields no rationale when it is empty/whitespace-only."""
    return not (text or "").strip()


# ---------------------------------------------------------------------------
# Property 13: Every flight and hotel option carries a rationale
# Feature: wayfinder-travel-planner, Property 13
# Validates: Requirements 1.4, 17.4
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(bundle=_tool_bundles())
def test_every_flight_and_hotel_option_carries_a_rationale(bundle: ToolBundle) -> None:
    """Every flight/hotel option in the produced plan has a non-empty rationale.

    **Validates: Requirements 1.4, 17.4**

    Two guarantees are checked together:

    1. Every surfaced ``FlightDTO`` / ``HotelDTO`` carries a non-empty rationale
       (the named property).
    2. The guardrail accounting balances: the count of options surfaced plus the
       count withheld equals the number of provider options, and an option is
       withheld exactly when its source tag carries no rationale (Req 1.3). This
       proves nothing was silently dropped or emitted with a blank rationale.
    """
    plan, withheld = merge_final_plan(
        scored=[], itinerary=[], bundle=bundle, constraints=None
    )

    # (1) The core property: no surfaced option has a blank rationale.
    for flight in plan.flights:
        assert flight.rationale and flight.rationale.strip(), (
            f"flight {flight.airline!r} surfaced with a blank rationale"
        )
    for hotel in plan.hotels:
        assert hotel.rationale and hotel.rationale.strip(), (
            f"hotel {hotel.name!r} surfaced with a blank rationale"
        )

    # (2) Guardrail accounting: surfaced + withheld == provided, split exactly on
    # whether the provider tag carried a rationale.
    withheld_flights = [w for w in withheld if w.kind == "flight"]
    withheld_hotels = [w for w in withheld if w.kind == "hotel"]

    expected_flight_blanks = sum(1 for f in bundle.flights if _is_blank(f.tag))
    expected_hotel_blanks = sum(1 for h in bundle.hotels if _is_blank(h.tag))

    assert len(plan.flights) + len(withheld_flights) == len(bundle.flights)
    assert len(plan.hotels) + len(withheld_hotels) == len(bundle.hotels)
    assert len(withheld_flights) == expected_flight_blanks
    assert len(withheld_hotels) == expected_hotel_blanks
    assert len(plan.flights) == len(bundle.flights) - expected_flight_blanks
    assert len(plan.hotels) == len(bundle.hotels) - expected_hotel_blanks
