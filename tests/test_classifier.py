"""Tests for the heuristic and hybrid classifiers."""

from __future__ import annotations

from auto_model_selector.classifier import heuristic_route, route
from auto_model_selector.config import RouterConfig
from auto_model_selector.models import Decision, Model


class _StubLLM:
    """A recording stub standing in for the LLM tie-breaker."""

    def __init__(self, decision: Decision | None) -> None:
        self._decision = decision
        self.calls: list[str] = []

    def classify(self, prompt: str) -> Decision | None:
        self.calls.append(prompt)
        return self._decision


def test_hard_task_routes_to_opus() -> None:
    decision = heuristic_route("Refactor the scheduler to fix a race condition under load")
    assert decision.model is Model.OPUS
    assert decision.confidence >= 0.6


def test_trivial_task_routes_to_haiku() -> None:
    decision = heuristic_route("rename foo to bar")
    assert decision.model is Model.HAIKU


def test_everyday_task_defaults_to_sonnet() -> None:
    decision = heuristic_route("Add a GET endpoint that returns the current user")
    assert decision.model is Model.SONNET


def test_short_prompt_alone_is_haiku() -> None:
    assert heuristic_route("fix the build").model is Model.HAIKU


def test_short_but_hard_prompt_is_not_haiku() -> None:
    # "debug" is an Opus keyword; it must override the short-prompt Haiku signal.
    assert heuristic_route("debug deadlock").model is Model.OPUS


def test_long_prompt_leans_opus() -> None:
    long_prompt = " ".join(["please"] * 90)
    assert heuristic_route(long_prompt).model is Model.OPUS


def test_code_heavy_prompt_leans_opus() -> None:
    prompt = "Why does this fail?\n```py\nx=1\n```\nand\n```py\ny=2\n```"
    assert heuristic_route(prompt).model is Model.OPUS


def test_confidence_grows_with_margin() -> None:
    weak = heuristic_route("please debug this")
    strong = heuristic_route("debug the race condition and security vulnerability in the algorithm")
    assert strong.confidence > weak.confidence


def test_route_skips_llm_when_confident() -> None:
    stub = _StubLLM(Decision(Model.HAIKU, 0.7, "stub", "llm"))
    decision = route("architect a distributed migration end to end", RouterConfig(), stub)
    assert decision.model is Model.OPUS
    assert stub.calls == []


def test_route_consults_llm_when_ambiguous() -> None:
    stub = _StubLLM(Decision(Model.OPUS, 0.7, "stub", "llm"))
    decision = route("Add a GET endpoint that returns the current user", RouterConfig(), stub)
    assert decision.model is Model.OPUS
    assert decision.method == "llm"
    assert len(stub.calls) == 1


def test_route_falls_back_when_llm_declines() -> None:
    stub = _StubLLM(None)
    decision = route("Add a GET endpoint that returns the current user", RouterConfig(), stub)
    assert decision.model is Model.SONNET
    assert decision.method == "heuristic"


def test_route_without_llm_returns_heuristic() -> None:
    decision = route("Add a GET endpoint that returns the current user")
    assert decision.model is Model.SONNET
