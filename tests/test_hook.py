"""Tests for the ``UserPromptSubmit`` hook.

The overriding property under test: the hook never blocks and always exits 0,
whatever the input.
"""

from __future__ import annotations

import io
import json
from typing import Any

from auto_model_selector import hook


def _run(event: dict[str, Any] | str) -> tuple[int, dict[str, Any] | None]:
    """Run the hook with an event and return (exit_code, parsed_output_or_None)."""
    raw = event if isinstance(event, str) else json.dumps(event)
    stdout = io.StringIO()
    code = hook.run(io.StringIO(raw), stdout)
    text = stdout.getvalue()
    return code, (json.loads(text) if text else None)


def test_hard_task_emits_opus_advisory() -> None:
    code, out = _run({"user_prompt": "refactor to fix the race condition across the codebase"})
    assert code == 0
    assert out is not None
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "Opus" in ctx
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


def test_trivial_task_emits_haiku_advisory() -> None:
    code, out = _run({"user_prompt": "rename foo to bar"})
    assert code == 0
    assert out is not None
    assert "Haiku" in out["hookSpecificOutput"]["additionalContext"]


def test_everyday_task_stays_silent() -> None:
    code, out = _run({"user_prompt": "Add a GET endpoint returning the current user"})
    assert code == 0
    assert out is None


def test_never_blocks() -> None:
    _, out = _run({"user_prompt": "refactor to fix the race condition across the codebase"})
    assert out is not None
    assert "decision" not in out


def test_empty_prompt_is_noop() -> None:
    assert _run({"user_prompt": "   "}) == (0, None)


def test_missing_prompt_is_noop() -> None:
    assert _run({"session_id": "abc"}) == (0, None)


def test_malformed_json_is_noop() -> None:
    assert _run("not json at all {{{") == (0, None)


def test_empty_stdin_is_noop() -> None:
    assert _run("") == (0, None)


def test_main_fails_open(monkeypatch: Any) -> None:
    def _boom(*_: Any, **__: Any) -> int:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(hook, "run", _boom)
    assert hook.main() == 0


def test_llm_path_consulted_when_enabled(monkeypatch: Any) -> None:
    from auto_model_selector.models import Decision, Model

    class _StubLLM:
        def classify(self, prompt: str) -> Decision:
            return Decision(Model.OPUS, 0.7, "stub", "llm")

    monkeypatch.setenv("AMS_HOOK_USE_LLM", "1")
    monkeypatch.setattr("auto_model_selector.llm.HaikuClassifier", _StubLLM)
    # An ambiguous everyday prompt: heuristics return Sonnet, so the LLM is consulted.
    _, out = _run({"user_prompt": "Add a GET endpoint returning the current user"})
    assert out is not None
    assert "Opus" in out["hookSpecificOutput"]["additionalContext"]
