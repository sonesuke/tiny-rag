.PHONY: help install test lint clean build

help:
	@echo "Available commands:"
	@echo "  make install       Install the package"
	@echo "  make test          Run tests"
	@echo "  make lint          Format and check code"
	@echo "  make clean         Clean up cache and build files"
	@echo "  make build         Build the package"

install:
	uv pip install -e .

test:
	uv run pytest tests/ -v

lint:
	uv run ruff format src/
	uv run ruff check --fix src/
	uv run pyright

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/

build:
	uv build