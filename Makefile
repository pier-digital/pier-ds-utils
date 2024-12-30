
.PHONY: init
init:
	poetry install -n

.PHONY: tests
tests:
	poetry run pytest --cov-report=term-missing:skip-covered --cov=pier_ds_utils tests/ | tee pytest-coverage.txt

.PHONY: check-formatting
check-formatting:
	poetry run ruff check .

.PHONY: formatting
formatting:
	poetry run ruff format .