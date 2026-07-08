"""Tests for the domain types."""

from __future__ import annotations

import pytest

from auto_model_selector.models import Decision, Model


def test_model_values_are_cli_aliases() -> None:
    assert Model.HAIKU.value == "haiku"
    assert Model.SONNET.value == "sonnet"
    assert Model.OPUS.value == "opus"


def test_decision_accepts_boundary_confidence() -> None:
    assert Decision(Model.SONNET, 0.0, "r", "heuristic").confidence == 0.0
    assert Decision(Model.SONNET, 1.0, "r", "heuristic").confidence == 1.0


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_decision_rejects_out_of_range_confidence(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        Decision(Model.SONNET, bad, "r", "heuristic")
