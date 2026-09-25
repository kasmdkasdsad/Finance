PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install api ui migrate test lint format typecheck check docker-up docker-down clean

install:  ## Create the virtualenv and install everything (pinned to tested versions)
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[frontend,dev]" -c constraints.txt

api:  ## Run the FastAPI server on :8000
	$(PY) -m uvicorn --factory quantpulse.api.app:app_factory --host 127.0.0.1 --port 8000

ui:  ## Run the Streamlit terminal on :8501
	.venv/bin/streamlit run frontend/app.py

migrate:  ## Apply database migrations
	.venv/bin/quantpulse-migrate

test:  ## Run the full test suite
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src frontend tests
	.venv/bin/ruff format --check src frontend tests

format:
	.venv/bin/ruff format src frontend tests
	.venv/bin/ruff check --fix src frontend tests

typecheck:
	.venv/bin/mypy

check: lint typecheck test  ## Everything CI runs

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
