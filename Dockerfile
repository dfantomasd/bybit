# =============================================================================
# Multi-stage Dockerfile for Bybit AI Trader
# =============================================================================
# Stage 1: builder  — installs uv, resolves and installs dependencies
# Stage 2: runtime  — minimal image with only runtime artifacts
# =============================================================================

# ---- Stage 1: Builder -------------------------------------------------------
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy dependency specification first for layer caching
COPY pyproject.toml ./
COPY README.md ./

# Install all production dependencies
RUN uv pip install --system ".[dev]"

# Copy application source
COPY src/ ./src/
COPY config/ ./config/
COPY migrations/ ./migrations/
COPY alembic.ini ./

# Install the package itself in editable mode
RUN uv pip install --system -e .

# ---- Stage 2: Runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Install minimal runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 trader && \
    useradd --uid 1000 --gid trader --shell /bin/bash --create-home trader

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/trader /usr/local/bin/trader
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /usr/local/bin/alembic /usr/local/bin/alembic

# Copy application artifacts
COPY --from=builder /build/src ./src
COPY --from=builder /build/config ./config
COPY --from=builder /build/migrations ./migrations
COPY --from=builder /build/alembic.ini ./

# Set ownership
RUN chown -R trader:trader /app

# Switch to non-root user
USER trader:trader

# Expose observability ports (not exchange-facing)
EXPOSE 8080 9090

# Health check via the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f -s -H "X-API-Key: ${INTERNAL_API_KEY:-}" http://localhost:8080/health || exit 1

CMD ["trader"]
