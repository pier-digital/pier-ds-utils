
.PHONY: init
init:
	uv sync

.PHONY: tests
tests:
	uv run pytest --cov-report=term-missing:skip-covered --cov=pier_ds_utils tests/ | tee pytest-coverage.txt

.PHONY: check-formatting
check-formatting:
	uv run ruff check .

.PHONY: formatting
formatting:
	uv run ruff format .
	uv run ruff check .
