# auto-model-selector

Task-aware model routing for [Claude Code](https://claude.com/claude-code). Inspect a task, pick the cheapest capable model — Haiku, Sonnet, or Opus.

Ships two things that share one classifier:

- **`route` CLI** — prints a model alias for launch-time selection. This is where routing *actually* takes effect.
- **`ams-hook`** — a `UserPromptSubmit` hook that *advises* which model to switch to, mid-session. Strictly advisory, never blocks.

## The one constraint you must know

**Claude Code hooks cannot change the active model.** It is a hard limit of the hook API — model selection is user-controlled (`/model`, `--model`, or `settings.json` before launch). So this project splits the job:

| Where | What it does | Effect |
|-------|--------------|--------|
| `route` CLI, at launch | `claude --model "$(route '...')"` | **Chooses** the model |
| `ams-hook`, mid-session | injects a one-line recommendation | **Advises**; you switch with `/model` |

The hook is **fail-open and non-blocking by design**: it never emits `decision: "block"`, always exits `0`, and swallows any internal error. It is safe under autonomous / auto-accept mode.

## Install as a Claude Code plugin (recommended)

This repo is also a Claude Code plugin marketplace. From inside Claude Code:

```
/plugin marketplace add shanwije/auto-model-selector
/plugin install auto-model-selector@auto-model-selector
```

That wires up the advisory `UserPromptSubmit` hook automatically. The hook runs
**stdlib-only** from the bundled source — no `pip`/`uv` step, just a Python 3.11+
interpreter on `PATH`. (Set `AMS_HOOK_USE_LLM=1` to enable the optional Haiku
tie-breaker, which additionally needs the `anthropic` package.)

## Install as a Python package (for the `route` CLI)

```bash
uv tool install auto-model-selector          # or: pipx install auto-model-selector
uv tool install "auto-model-selector[llm]"   # + optional Haiku tie-breaker
```

## Use the CLI

```bash
claude --model "$(route 'refactor the scheduler to fix a race condition')"   # -> opus
route "rename foo to bar"                                                     # -> haiku
route --json "add a GET endpoint for the current user"                        # full decision
route --explain "optimize the query planner"                                  # reason on stderr
echo "fix the flaky test" | route                                            # reads stdin
```

Enable the LLM tie-breaker for ambiguous tasks (needs the `llm` extra and `ANTHROPIC_API_KEY`):

```bash
route --llm "make the importer idempotent"
```

## Use the hook

Add to `.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "ams-hook" } ] }
    ]
  }
}
```

The hook runs on heuristics only (instant, network-free). To enable the LLM tie-breaker inside the hook, set `AMS_HOOK_USE_LLM=1`.

## How it routes

A hybrid classifier:

1. **Heuristics** (pure, fast, deterministic) score keyword and length signals. Clear hard-task signals → Opus; clear trivial signals → Haiku; everything else → Sonnet.
2. **LLM tie-breaker** (optional) — only for prompts the heuristics find ambiguous, one small Haiku call. If it is unavailable or fails, the heuristic result stands.

Tune it via `RouterConfig`:

```python
from auto_model_selector import RouterConfig, route

config = RouterConfig(long_prompt_words=120, confidence_threshold=0.7)
decision = route("port the auth service to async", config)
print(decision.model.value, decision.reason)
```

## Develop

```bash
make install   # uv sync with dev + llm extras
make check     # lint + type check + tests (the full local gate)
make fix       # auto-fix lint and format
make help      # list all targets
```

Or drive the tools directly:

```bash
uv sync --extra dev --extra llm
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
