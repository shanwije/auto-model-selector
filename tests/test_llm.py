"""Tests for the optional Haiku classifier, using fake clients (no network)."""

from __future__ import annotations

from typing import Any, ClassVar

from auto_model_selector.llm import HaikuClassifier
from auto_model_selector.models import Model


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _FakeMessages:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _Response:
        self.kwargs = kwargs
        return _Response(self._reply)


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.messages = _FakeMessages(reply)


class _RaisingClient:
    class messages:  # noqa: N801 -- mimics the SDK's attribute shape
        @staticmethod
        def create(**_: Any) -> Any:
            raise RuntimeError("network down")


def test_valid_reply_maps_to_model() -> None:
    clf = HaikuClassifier(client=_FakeClient("opus"))
    decision = clf.classify("some hard task")
    assert decision is not None
    assert decision.model is Model.OPUS
    assert decision.method == "llm"


def test_reply_is_normalized() -> None:
    clf = HaikuClassifier(client=_FakeClient("  Haiku\n"))
    decision = clf.classify("trivial")
    assert decision is not None
    assert decision.model is Model.HAIKU


def test_unrecognized_reply_returns_none() -> None:
    clf = HaikuClassifier(client=_FakeClient("gpt-9"))
    assert clf.classify("task") is None


def test_client_error_fails_open() -> None:
    clf = HaikuClassifier(client=_RaisingClient())
    assert clf.classify("task") is None


def test_missing_key_and_no_client_returns_none(monkeypatch: Any) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    clf = HaikuClassifier(client=None)
    assert clf.classify("task") is None


def test_empty_response_content_returns_none() -> None:
    class _EmptyResponse:
        content: ClassVar[list[object]] = []

    class _EmptyMessages:
        def create(self, **_: Any) -> _EmptyResponse:
            return _EmptyResponse()

    class _EmptyClient:
        def __init__(self) -> None:
            self.messages = _EmptyMessages()

    clf = HaikuClassifier(client=_EmptyClient())
    assert clf.classify("task") is None
