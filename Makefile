# NeuralForge-X — developer shortcuts.
# Assumes the project virtualenv is active (so `python` is the venv interpreter).
PY    ?= python
CARGO ?= cargo

.DEFAULT_GOAL := help

.PHONY: help setup build dev test test-rust test-py bench lint fmt fmt-fix docs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install Python dev deps and build the extension
	$(PY) -m pip install -e ".[dev]"

build: ## Build the Rust workspace (release)
	$(CARGO) build --release --workspace

dev: ## Build + install the extension into the venv (maturin)
	maturin develop --release

test: test-rust test-py ## Run all tests

test-rust: ## Rust unit + integration tests
	$(CARGO) test --workspace

test-py: ## Python tests
	$(PY) -m pytest

bench: ## Run criterion benchmarks
	$(CARGO) bench -p neuralforge_core

lint: ## Clippy + ruff + mypy
	$(CARGO) clippy --all-targets -- -D warnings
	$(PY) -m ruff check .
	$(PY) -m mypy

fmt: ## Check formatting (Rust + Python)
	$(CARGO) fmt --all --check
	$(PY) -m ruff format --check .

fmt-fix: ## Apply formatting
	$(CARGO) fmt --all
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

docs: ## Build rustdoc
	$(CARGO) doc -p neuralforge_core --no-deps

clean: ## Remove build artifacts
	$(CARGO) clean
	-$(PY) -c "import shutil,glob,os; [shutil.rmtree(p,True) for p in glob.glob('**/__pycache__',recursive=True)]"
