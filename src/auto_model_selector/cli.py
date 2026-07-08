"""Command-line entry point: print the model to launch Claude Code with.

Launch time is the one place model routing genuinely takes effect, so this CLI
is the primary way to use the router::

    claude --model "$(route 'refactor the auth module for thread safety')"

By default it prints just the model alias so it composes in a shell. ``--json``
and ``--explain`` expose the full decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .classifier import route
from .config import RouterConfig
from .models import Decision


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="route",
        description="Pick a Claude Code model for a task.",
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="The task text. If omitted, the task is read from stdin.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full decision as JSON instead of just the model alias.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the model alias followed by the reason on stderr.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable the Haiku tie-breaker for ambiguous tasks (needs the 'llm' extra).",
    )
    return parser


def _resolve_task(args: argparse.Namespace, stdin: object) -> str:
    """Return the task text from positional args or stdin.

    Args:
        args: Parsed CLI arguments.
        stdin: The standard input stream.

    Returns:
        The task text, possibly empty.
    """
    if args.task:
        return " ".join(args.task)
    read = getattr(stdin, "read", None)
    return read().strip() if callable(read) else ""


def _decide(task: str, use_llm: bool) -> Decision:
    """Route ``task``, wiring up the LLM tie-breaker only when requested."""
    llm = None
    if use_llm:
        from .llm import HaikuClassifier

        llm = HaikuClassifier()
    return route(task, RouterConfig(), llm)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector for testing; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``2`` if no task was provided.
    """
    args = _build_parser().parse_args(argv)
    task = _resolve_task(args, sys.stdin)
    if not task:
        print("error: no task provided (pass as arguments or via stdin)", file=sys.stderr)
        return 2

    decision = _decide(task, args.llm)

    if args.json:
        print(
            json.dumps(
                {
                    "model": decision.model.value,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "method": decision.method,
                }
            )
        )
    else:
        print(decision.model.value)
        if args.explain:
            print(f"reason: {decision.reason} ({decision.method})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
