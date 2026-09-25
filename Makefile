COMPLEXITY_MAX ?= 12
RUFF_ARGS ?=
COMPLEXIPY_ARGS ?=
JSCPD_ARGS ?=
JSCPD_THRESHOLD ?= 3

.PHONY: init
init:
	uv sync

.PHONY: tests
tests:
	uv run pytest --cov-report=term-missing:skip-covered --cov=pier_ds_utils tests/ | tee pytest-coverage.txt

.PHONY: lint
lint:
	uv run ruff check . $(RUFF_ARGS)
	uv run ruff format --check .

.PHONY: duplication
duplication:
	npx --yes jscpd@latest pier_ds_utils --config .jscpd.json --threshold $(JSCPD_THRESHOLD) $(JSCPD_ARGS)

.PHONY: cc
cc:
	uvx complexipy . --max-complexity-allowed $(COMPLEXITY_MAX) $(COMPLEXIPY_ARGS)

.PHONY: check
check: lint duplication cc

.PHONY: formatting
formatting:
	uv run ruff format .
	uv run ruff check .
