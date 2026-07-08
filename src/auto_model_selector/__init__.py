"""Task-aware model routing for Claude Code.

Public API:
    - :func:`~auto_model_selector.classifier.route` — hybrid routing.
    - :func:`~auto_model_selector.classifier.heuristic_route` — rules only.
    - :class:`~auto_model_selector.config.RouterConfig` — tuning knobs.
    - :class:`~auto_model_selector.models.Model` / :class:`~auto_model_selector.models.Decision`.
"""

from __future__ import annotations

from .classifier import heuristic_route, route
from .config import RouterConfig
from .models import Decision, Method, Model

__all__ = [
    "Decision",
    "Method",
    "Model",
    "RouterConfig",
    "heuristic_route",
    "route",
]

__version__ = "0.1.0"
