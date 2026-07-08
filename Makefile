.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install sync lint format typecheck test check fix build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install dev + llm extras
	$(UV) sync --extra dev --extra llm

sync: install ## Alias for install

lint: ## Run ruff lint and format checks
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-format the code
	$(UV) run ruff format .

typecheck: ## Run mypy in strict mode
	$(UV) run mypy

test: ## Run the test suite with coverage
	$(UV) run pytest

check: lint typecheck test ## Run the full local gate (lint + types + tests)

fix: ## Auto-fix lint issues and format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

build: ## Build the wheel and sdist
	$(UV) build

clean: ## Remove build and tooling caches
	rm -rf dist build *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
