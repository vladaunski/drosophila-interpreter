"""Tests for semantic sensory routing across all expanded modalities."""

import pytest

from drosophila_interpreter.config import SensoryModality
from drosophila_interpreter.router import SensoryRouter


@pytest.fixture(scope="module")
def router() -> SensoryRouter:
    return SensoryRouter()


def test_empty_string_yields_all_zeros(router: SensoryRouter) -> None:
    currents = router.route("")
    assert all(val == 0.0 for val in currents.values())


def test_acoustic_routing(router: SensoryRouter) -> None:
    currents = router.route("LOUD SHOUTING DIRECTLY AT YOU!")
    assert currents[SensoryModality.ACOUSTIC_VIBRATION] > 0.3
    assert currents[SensoryModality.SUGAR] == 0.0


def test_multimodal_conflict_routing(router: SensoryRouter) -> None:
    # A prompt with competing signals (sugar + threat)
    currents = router.route("A fly swatter slams down on a plate of sweet honey")
    assert currents[SensoryModality.SUGAR] > 0.2
    assert currents[SensoryModality.LOOMING_THREAT] > 0.2