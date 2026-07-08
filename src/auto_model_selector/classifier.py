"""Task classification: map a prompt to a recommended model.

The heuristic layer is pure, fast, and deterministic. The hybrid entry point
adds an optional LLM tie-breaker for prompts the heuristics find ambiguous, and
always degrades safely to the heuristic result if the LLM is absent or fails.
"""

from __future__ import annotations

import re
from typing import Protocol

from .config import RouterConfig
from .models import Decision, Model

_WORD_RE = re.compile(r"\b\w+\b")
_FENCE = "```"


class LLMClassifier(Protocol):
    """A tie-breaker that classifies ambiguous prompts.

    Implementations must never raise: a classifier that cannot decide (network
    error, missing credentials, unparseable reply) returns ``None`` so the
    caller can fall back to the heuristic result.
    """

    def classify(self, prompt: str) -> Decision | None:
        """Classify ``prompt`` or return ``None`` if no decision can be made."""
        ...


def _count_words(text: str) -> int:
    """Return the number of word tokens in ``text``."""
    return len(_WORD_RE.findall(text))


def _matched_keywords(haystack: str, keywords: frozenset[str]) -> list[str]:
    """Return the keywords that appear as substrings of ``haystack``."""
    return [kw for kw in keywords if kw in haystack]


def _confidence(margin: int) -> float:
    """Map a score margin to a confidence in ``[0.6, 1.0]``.

    Args:
        margin: The winning tier's score minus the runner-up's; at least 1.

    Returns:
        A confidence that grows with the margin and saturates at ``1.0``.
    """
    return min(1.0, 0.6 + 0.15 * (margin - 1))


def heuristic_route(prompt: str, config: RouterConfig | None = None) -> Decision:
    """Route a prompt using keyword and length heuristics only.

    Args:
        prompt: The user's task text.
        config: Tuning knobs; defaults to :class:`RouterConfig`.

    Returns:
        A :class:`Decision`. Opus and Haiku are chosen only on a positive,
        winning score; otherwise the decision is Sonnet with low confidence,
        which the hybrid router treats as ambiguous.
    """
    config = config or RouterConfig()
    lowered = prompt.lower()
    n_words = _count_words(prompt)

    opus_hits = _matched_keywords(lowered, config.opus_keywords)
    haiku_hits = _matched_keywords(lowered, config.haiku_keywords)

    is_long = n_words >= config.long_prompt_words
    is_short = n_words <= config.short_prompt_words
    is_code_heavy = prompt.count(_FENCE) >= 2

    opus_score = len(opus_hits) + int(is_long) + int(is_code_heavy)
    # A short prompt only counts toward Haiku when nothing marks it as hard.
    haiku_score = len(haiku_hits) + int(is_short and opus_score == 0)

    if opus_score > haiku_score and opus_score > 0:
        signals = (
            opus_hits
            + (["long prompt"] if is_long else [])
            + (["code-heavy"] if is_code_heavy else [])
        )
        return Decision(
            model=Model.OPUS,
            confidence=_confidence(opus_score - haiku_score),
            reason=f"complex-task signals: {', '.join(signals)}",
            method="heuristic",
        )

    if haiku_score > opus_score and haiku_score > 0:
        signals = haiku_hits + (["short prompt"] if is_short else [])
        return Decision(
            model=Model.HAIKU,
            confidence=_confidence(haiku_score - opus_score),
            reason=f"trivial-task signals: {', '.join(signals)}",
            method="heuristic",
        )

    # No clear winner: default to Sonnet, flagged low-confidence so the hybrid
    # router knows it may consult the LLM.
    reason = "no decisive signal" if not (opus_hits or haiku_hits) else "conflicting signals"
    return Decision(
        model=Model.SONNET,
        confidence=0.5 if (opus_hits or haiku_hits) else 0.4,
        reason=reason,
        method="heuristic",
    )


def route(
    prompt: str,
    config: RouterConfig | None = None,
    llm: LLMClassifier | None = None,
) -> Decision:
    """Route a prompt, consulting the LLM only when heuristics are unsure.

    Args:
        prompt: The user's task text.
        config: Tuning knobs; defaults to :class:`RouterConfig`.
        llm: Optional tie-breaker for ambiguous prompts. If it declines
            (returns ``None``) or is absent, the heuristic result stands.

    Returns:
        The final :class:`Decision`.
    """
    config = config or RouterConfig()
    decision = heuristic_route(prompt, config)

    if decision.confidence >= config.confidence_threshold:
        return decision
    if llm is None:
        return decision

    llm_decision = llm.classify(prompt)
    return llm_decision if llm_decision is not None else decision
