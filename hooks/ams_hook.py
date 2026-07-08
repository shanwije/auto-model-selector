#!/usr/bin/env python3
"""Plugin launcher for the ``UserPromptSubmit`` hook.

Claude Code invokes this file directly, with the hook event JSON on stdin. It
adds the bundled package source to ``sys.path`` and delegates to
:func:`auto_model_selector.hook.main`, so the plugin runs with **no install
step** — only a Python 3.11+ interpreter is required for the heuristic path
(the optional LLM tie-breaker additionally needs the ``anthropic`` package and
``AMS_HOOK_USE_LLM=1``).

The delegate fails open on any error and always exits 0, so a routing problem
can never block a prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

try:
    from auto_model_selector.hook import main
except Exception:  # noqa: BLE001 -- fail open: a broken import must never block a prompt
    # If the bundled source can't be imported for any reason, do nothing.
    sys.exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
