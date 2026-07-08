"""Tests for the ``route`` command-line interface."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from auto_model_selector import cli


def test_prints_model_alias(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["rename", "foo", "to", "bar"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "haiku"


def test_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["--json", "refactor", "the", "race", "condition"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "opus"
    assert set(payload) == {"model", "confidence", "reason", "method"}


def test_explain_writes_reason_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--explain", "rename", "foo"])
    captured = capsys.readouterr()
    assert captured.out.strip() == "haiku"
    assert "reason:" in captured.err


def test_reads_task_from_stdin(monkeypatch: Any, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("rename foo to bar"))
    code = cli.main([])
    assert code == 0
    assert capsys.readouterr().out.strip() == "haiku"


def test_no_task_is_usage_error(monkeypatch: Any, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    code = cli.main([])
    assert code == 2
    assert "no task" in capsys.readouterr().err


def test_llm_flag_wires_tie_breaker(monkeypatch: Any, capsys: pytest.CaptureFixture[str]) -> None:
    from auto_model_selector.models import Decision, Model

    class _StubLLM:
        def classify(self, prompt: str) -> Decision:
            return Decision(Model.OPUS, 0.7, "stub", "llm")

    monkeypatch.setattr("auto_model_selector.llm.HaikuClassifier", _StubLLM)
    # Ambiguous everyday task -> heuristics return Sonnet -> LLM tie-breaker picks Opus.
    code = cli.main(["--llm", "add a GET endpoint returning the current user"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "opus"
