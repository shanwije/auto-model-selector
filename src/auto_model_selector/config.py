"""Routing configuration.

Everything that governs *which* task maps to *which* model lives here, so the
behaviour can be tuned per environment without touching the classifier. The
defaults are conservative: they only escalate to Opus or de-escalate to Haiku
on clear signals, and otherwise fall through to Sonnet.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Substrings that signal genuinely hard work: deep reasoning, whole-system
# changes, or subtle correctness. Matched case-insensitively as substrings, so
# "concurren" covers "concurrent"/"concurrency".
_DEFAULT_OPUS_KEYWORDS: frozenset[str] = frozenset(
    {
        "architect",
        "architecture",
        "design a",
        "redesign",
        "refactor",
        "debug",
        "root cause",
        "race condition",
        "concurren",
        "deadlock",
        "security",
        "vulnerab",
        "optimize",
        "optimise",
        "algorithm",
        "complexity",
        "migrate",
        "migration",
        "trade-off",
        "tradeoff",
        "distributed",
        "prove",
        "proof",
        "investigate",
        "diagnose",
        "end to end",
        "end-to-end",
        "entire codebase",
        "whole repo",
        "across the codebase",
    }
)

# Substrings that signal trivial, mechanical work that a small fast model handles
# well.
_DEFAULT_HAIKU_KEYWORDS: frozenset[str] = frozenset(
    {
        "rename",
        "typo",
        "spelling",
        "reformat",
        "format this",
        "lint",
        "bump version",
        "changelog",
        "add a comment",
        "what is",
        "what's",
        "define",
        "list the",
        "print",
        "echo",
        "remove unused",
        "one-liner",
        "one liner",
    }
)


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Tunable thresholds and keyword sets for the classifier.

    Attributes:
        opus_keywords: Substrings that push a task toward Opus.
        haiku_keywords: Substrings that push a task toward Haiku.
        long_prompt_words: Word count at or above which a prompt counts as long
            (a weak Opus signal).
        short_prompt_words: Word count at or below which a prompt counts as short
            (a weak Haiku signal).
        confidence_threshold: Minimum heuristic confidence to accept without
            consulting the optional LLM classifier.
        default_model_is_sonnet: Kept for documentation; the fallback tier is
            always Sonnet.
    """

    opus_keywords: frozenset[str] = _DEFAULT_OPUS_KEYWORDS
    haiku_keywords: frozenset[str] = _DEFAULT_HAIKU_KEYWORDS
    long_prompt_words: int = 80
    short_prompt_words: int = 6
    confidence_threshold: float = 0.6
    default_model_is_sonnet: bool = field(default=True)

    def __post_init__(self) -> None:
        """Validate threshold coherence.

        Raises:
            ValueError: If the word thresholds overlap or the confidence
                threshold is out of range.
        """
        if self.short_prompt_words >= self.long_prompt_words:
            msg = "short_prompt_words must be below long_prompt_words"
            raise ValueError(msg)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            msg = "confidence_threshold must be in [0.0, 1.0]"
            raise ValueError(msg)

    def with_overrides(self, **changes: object) -> RouterConfig:
        """Return a copy with the given fields replaced.

        Args:
            **changes: Field values to override.

        Returns:
            A new :class:`RouterConfig`; the original is unchanged.
        """
        return replace(self, **changes)  # type: ignore[arg-type]
