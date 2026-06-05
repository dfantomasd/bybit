# =============================================================================
# Bybit AI Trader — Makefile
# =============================================================================
# Usage: make <target>
# Run `make help` to list all available targets.
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-unit test-integration \
        docker-build docker-up docker-down migrate clean security

PYTHON := python3
UV := uv
DOCKER_COMPOSE := docker compose
SRC := src/trader
TESTS := tests

# -----------------------------------------------------------------------------
help: ## Show this help message
	@echo "Bybit AI Trader — Available Make Targets"
	@echo "========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# Development environment
# -----------------------------------------------------------------------------
install: ## Install all dependencies (production + dev) via uv
	@echo "Installing dependencies..."
	$(UV) pip install --system ".[dev]"
	@echo "Done. Run 'make test' to verify the install."

install-prod: ## Install production dependencies only
	$(UV) pip install --system "."

# -----------------------------------------------------------------------------
# Code quality
# -----------------------------------------------------------------------------
lint: ## Run ruff linter (check mode, no auto-fix)
	ruff check $(SRC)/ $(TESTS)/

lint-fix: ## Run ruff linter with auto-fix
	ruff check --fix $(SRC)/ $(TESTS)/

format: ## Run ruff formatter (auto-format)
	ruff format $(SRC)/ $(TESTS)/

format-check: ## Check formatting without modifying files
	ruff format --check $(SRC)/ $(TESTS)/

typecheck: ## Run mypy type checker
	mypy $(SRC)/

check: lint format-check typecheck ## Run all checks (lint + format + types)

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
test: ## Run all tests with coverage
	pytest $(TESTS)/ \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v

test-unit: ## Run unit tests only (fast, no external services)
	pytest $(TESTS)/unit/ \
		--cov=$(SRC) \
		--cov-report=term-missing \
		-v \
		--tb=short

test-integration: ## Run integration tests (requires running services)
	pytest $(TESTS)/integration/ \
		--cov=$(SRC) \
		--cov-report=term-missing \
		-v \
		--tb=short

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw $(TESTS)/unit/ -- -v --tb=short

coverage: ## Generate HTML coverage report and open in browser
	pytest $(TESTS)/unit/ --cov=$(SRC) --cov-report=html:htmlcov -q
	@echo "Coverage report: htmlcov/index.html"

# -----------------------------------------------------------------------------
# Security
# -----------------------------------------------------------------------------
security: ## Run all security scans
	@echo "Running bandit SAST..."
	bandit -r $(SRC)/ -c pyproject.toml --severity-level medium
	@echo ""
	@echo "Running pip-audit..."
	pip-audit
	@echo ""
	@echo "Security scan complete."

bandit: ## Run bandit SAST scan
	bandit -r $(SRC)/ -c pyproject.toml --severity-level medium --confidence-level medium

audit: ## Run pip-audit dependency vulnerability check
	pip-audit

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
docker-build: ## Build Docker images
	$(DOCKER_COMPOSE) build

docker-build-no-cache: ## Build Docker images (no layer cache)
	$(DOCKER_COMPOSE) build --no-cache

docker-up: ## Start all services (production config)
	$(DOCKER_COMPOSE) up -d

docker-up-dev: ## Start all services (development config with hot-reload)
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up

docker-down: ## Stop all services
	$(DOCKER_COMPOSE) down

docker-down-volumes: ## Stop all services and remove volumes (DESTRUCTIVE)
	$(DOCKER_COMPOSE) down -v

docker-logs: ## Tail logs from all services
	$(DOCKER_COMPOSE) logs -f

docker-logs-trader: ## Tail logs from trader-core only
	$(DOCKER_COMPOSE) logs -f trader-core

docker-ps: ## List running containers
	$(DOCKER_COMPOSE) ps

docker-shell: ## Open a shell in the trader-core container
	$(DOCKER_COMPOSE) exec trader-core bash

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
migrate: ## Run Alembic database migrations
	alembic upgrade head

migrate-dry: ## Show pending migrations without applying them
	alembic upgrade head --sql

migrate-rollback: ## Roll back the last migration
	alembic downgrade -1

migrate-history: ## Show migration history
	alembic history --verbose

migrate-new: ## Create a new migration (usage: make migrate-new MSG="description")
	alembic revision --autogenerate -m "$(MSG)"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean: ## Remove build artifacts and cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "coverage.xml" -delete
	find . -name ".coverage" -delete
	@echo "Clean complete."

# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------
bootstrap: ## Full first-time setup (install + env + migrate)
	bash scripts/bootstrap.sh
