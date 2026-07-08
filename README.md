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

## Install

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
uv venv && uv pip install -e ".[dev,llm]"
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
