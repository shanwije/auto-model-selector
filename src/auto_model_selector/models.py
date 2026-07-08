"""Domain types for model routing decisions.

These types are the shared vocabulary between the classifier, the CLI, and the
Claude Code hook. They are deliberately free of any I/O so they can be tested
and reused in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class Model(StrEnum):
    """A Claude Code model tier.

    The values are the aliases Claude Code accepts for ``--model`` and the
    ``model`` setting, so a :class:`Model` can be handed straight to the CLI.
    """

    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


Method = Literal["heuristic", "llm", "fallback"]
"""How a :class:`Decision` was reached.

``heuristic`` — decided by keyword/length rules alone.
``llm`` — decided by the optional Haiku classifier.
``fallback`` — the safe default returned when no signal was conclusive.
"""


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of routing a single task.

    Attributes:
        model: The recommended model tier.
        confidence: Confidence in ``model``, in the inclusive range 0.0 to 1.0.
        reason: A short, human-readable justification for the choice.
        method: Which mechanism produced the decision.
    """

    model: Model
    confidence: float
    reason: str
    method: Method

    def __post_init__(self) -> None:
        """Validate the confidence bound.

        Raises:
            ValueError: If ``confidence`` is outside the range 0.0 to 1.0.
        """
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence must be in [0.0, 1.0], got {self.confidence}"
            raise ValueError(msg)
