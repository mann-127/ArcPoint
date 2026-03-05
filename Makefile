.PHONY: help install install-dev test test-cov lint format clean run-all generate train route serve dashboard

help:
	@echo "ArcPoint: Intelligent Request Router"
	@echo ""
	@echo "Available targets:"
	@echo "  make install         - Install dependencies with uv (recommended)"
	@echo "  make install-dev     - Install dev dependencies (testing, linting)"
	@echo "  make test            - Run all 145 tests"
	@echo "  make test-cov        - Run tests with coverage report"
	@echo "  make lint            - Run linters (ruff, mypy)"
	@echo "  make format          - Format code with black, ruff"
	@echo "  make clean           - Remove generated files and caches"
	@echo "  make generate        - Generate synthetic training data"
	@echo "  make train           - Train latency prediction model"
	@echo "  make route           - Run router simulation"
	@echo "  make serve           - Start Context Service (FastAPI on :8000)"
	@echo "  make dashboard       - Start Streamlit dashboard (requires service running)"
	@echo "  make run-all         - Full pipeline: data → train → route"
	@echo ""

install:
	uv sync

install-dev:
	uv sync --with dev

test:
	uv run --with pytest --with pytest-cov pytest tests/ -v

test-cov:
	uv run --with pytest --with pytest-cov pytest tests/ --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

lint:
	uv run --with dev python -m ruff check arcpoint/ data/ tests/ || true
	uv run --with dev python -m mypy arcpoint/ || true

format:
	uv run --with dev python -m black arcpoint/ data/ tests/
	uv run --with dev python -m ruff check --fix arcpoint/ data/ tests/ || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name 'htmlcov' -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
	@echo "Cleaned cache and temp files"

generate:
	uv run python data/generate.py

train:
	uv run python -m arcpoint.routing.model

route:
	uv run python -m arcpoint.routing.engine

serve:
	uv run --with fastapi uvicorn arcpoint.context.api:app --reload

dashboard:
	uv run --with streamlit streamlit run arcpoint/observability/dashboard.py

run-all: clean generate train route
	@echo ""
	@echo "✓ Pipeline complete! (data generation, model training, routing simulation)"
	@echo ""
	@echo "Next steps:"
	@echo "  Terminal 1: make serve        # Start Context Service"
	@echo "  Terminal 2: make dashboard    # Start dashboard (after service is up)"
	@echo "  Terminal 3: make test         # Run test suite"
	@echo ""

.DEFAULT_GOAL := help
