"""Claude Code ``UserPromptSubmit`` hook: recommend a model, never block.

Claude Code hooks cannot switch the active model — that is a hard limitation of
the hook API. This hook therefore only *advises*: it injects a one-line
recommendation as ``additionalContext`` when a non-default tier looks clearly
better, and otherwise stays silent.

Two invariants keep it safe for autonomous ("auto") mode:

1. It never emits ``decision: "block"`` and always exits ``0``. A prompt is
   never rejected or delayed by a routing failure.
2. Every code path is wrapped so that any unexpected error fails open — the
   hook prints nothing and exits ``0`` rather than propagating.

The LLM tie-breaker is off by default here to keep the hook instant and
network-free; set ``AMS_HOOK_USE_LLM=1`` to enable it.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TextIO

from .classifier import route
from .config import RouterConfig
from .models import Decision, Model

_ADVISORY_TABLE = {
    Model.OPUS: (
        "This looks like a hard task; Opus may handle it better. Switch with /model opus."
    ),
    Model.HAIKU: (
        "This looks trivial; Haiku would be faster and cheaper. Switch with /model haiku."
    ),
}


def _read_event(stream: TextIO) -> dict[str, object]:
    """Parse the hook's JSON stdin payload.

    Args:
        stream: The input stream carrying the event JSON.

    Returns:
        The decoded event, or an empty dict if input is missing or malformed.
    """
    raw = stream.read()
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _advisory_for(decision: Decision, config: RouterConfig) -> str | None:
    """Return advisory text worth surfacing, or ``None`` to stay silent.

    Only confident, non-default (Opus/Haiku) recommendations are surfaced, to
    avoid nagging on every prompt.

    Args:
        decision: The routing decision.
        config: The active configuration, for the confidence gate.

    Returns:
        A short advisory string, or ``None``.
    """
    if decision.model is Model.SONNET:
        return None
    if decision.confidence < config.confidence_threshold:
        return None
    return _ADVISORY_TABLE.get(decision.model)


def _emit(context: str, stream: TextIO) -> None:
    """Write a well-formed ``additionalContext`` payload to ``stream``."""
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": f"[auto-model-selector] {context}",
        }
    }
    json.dump(payload, stream)


def run(stdin: TextIO, stdout: TextIO) -> int:
    """Execute the hook against the given streams.

    Args:
        stdin: Stream carrying the event JSON.
        stdout: Stream to write any advisory payload to.

    Returns:
        Always ``0``. The hook is advisory and must never block a prompt.
    """
    event = _read_event(stdin)
    prompt = event.get("user_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return 0

    llm = None
    if os.environ.get("AMS_HOOK_USE_LLM") == "1":
        from .llm import HaikuClassifier

        llm = HaikuClassifier()

    config = RouterConfig()
    decision = route(prompt, config, llm)
    advisory = _advisory_for(decision, config)
    if advisory is not None:
        _emit(advisory, stdout)
    return 0


def main() -> int:
    """Console-script entry point. Fails open on any unexpected error.

    Returns:
        Always ``0``.
    """
    try:
        return run(sys.stdin, sys.stdout)
    except Exception:  # noqa: BLE001 -- fail open: a routing bug must never block a prompt
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
