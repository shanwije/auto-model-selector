"""Optional Haiku-backed tie-breaker for ambiguous prompts.

This module is imported lazily and depends on the ``anthropic`` package, which
is an optional extra. It is built to *never* raise: any failure — missing
dependency, missing key, network error, or an unparseable reply — resolves to
``None`` so the caller falls back to the heuristic decision. That fail-open
contract is what lets the hook stay non-blocking.
"""

from __future__ import annotations

import os

from .models import Decision, Model

_SYSTEM_PROMPT = (
    "You route a coding task to the cheapest capable Claude model. "
    "Reply with exactly one word: 'haiku' for trivial/mechanical tasks, "
    "'sonnet' for everyday coding, or 'opus' for hard reasoning, architecture, "
    "debugging subtle issues, or whole-system changes. Output only the word."
)

_VALID = {m.value: m for m in Model}


class HaikuClassifier:
    """Classifies ambiguous prompts with a single small Claude call.

    Attributes are injected rather than read from globals so the classifier can
    be tested with a fake client.
    """

    def __init__(
        self,
        client: object | None = None,
        *,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 8,
        timeout: float = 8.0,
    ) -> None:
        """Build a classifier.

        Args:
            client: An object exposing ``messages.create(...)`` compatible with
                the Anthropic SDK. If ``None``, a real client is constructed on
                first use (requires the ``llm`` extra and an API key).
            model: The model id used for classification.
            max_tokens: Upper bound on the reply length.
            timeout: Per-request timeout in seconds.
        """
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout

    def _ensure_client(self) -> object | None:
        """Return a usable client, building a real one if needed, else ``None``."""
        if self._client is not None:
            return self._client
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        try:
            from anthropic import Anthropic
        except ImportError:
            return None
        self._client = Anthropic(timeout=self._timeout)
        return self._client

    def classify(self, prompt: str) -> Decision | None:
        """Classify ``prompt`` with the LLM, or return ``None`` on any failure.

        Args:
            prompt: The user's task text.

        Returns:
            A :class:`Decision` with method ``"llm"``, or ``None`` if the model
            was unavailable or its reply could not be interpreted.
        """
        client = self._ensure_client()
        if client is None:
            return None

        try:
            response = client.messages.create(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            word = _extract_text(response).strip().lower()
        except Exception:  # noqa: BLE001 -- fail open: any error means "no decision"
            return None

        model = _VALID.get(word)
        if model is None:
            return None
        return Decision(
            model=model,
            confidence=0.7,
            reason="LLM tie-breaker on ambiguous prompt",
            method="llm",
        )


def _extract_text(response: object) -> str:
    """Pull the first text block out of an Anthropic-style response.

    Args:
        response: The object returned by ``messages.create``.

    Returns:
        The text of the first content block, or ``""`` if none is present.
    """
    content = getattr(response, "content", None)
    if not content:
        return ""
    first = content[0]
    return str(getattr(first, "text", "") or "")
