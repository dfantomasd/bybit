#!/usr/bin/env bash
# =============================================================================
# Bybit AI Trader — Bootstrap Script
# =============================================================================
# Sets up the development environment from scratch.
# Run this once after cloning the repository.
# =============================================================================
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Step 1: Check Python version
# ---------------------------------------------------------------------------
info "Checking Python version..."
REQUIRED_MAJOR=3
REQUIRED_MINOR=11

if ! command -v python3 &>/dev/null; then
    error "Python 3 is not installed. Install Python 3.11+ and retry."
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [[ "$PYTHON_MAJOR" -lt "$REQUIRED_MAJOR" ]] || \
   [[ "$PYTHON_MAJOR" -eq "$REQUIRED_MAJOR" && "$PYTHON_MINOR" -lt "$REQUIRED_MINOR" ]]; then
    error "Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ is required. Found $PYTHON_VERSION."
fi

success "Python $PYTHON_VERSION found."

# ---------------------------------------------------------------------------
# Step 2: Install uv
# ---------------------------------------------------------------------------
info "Checking for uv package manager..."
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        error "uv installation failed. Check https://docs.astral.sh/uv/getting-started/installation/"
    fi
    success "uv installed."
else
    UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
    success "uv found: $UV_VERSION"
fi

# ---------------------------------------------------------------------------
# Step 3: Create virtual environment
# ---------------------------------------------------------------------------
VENV_DIR=".venv"
info "Setting up virtual environment in $VENV_DIR ..."
if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at $VENV_DIR — skipping creation."
else
    uv venv "$VENV_DIR" --python "python3"
    success "Virtual environment created."
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
success "Virtual environment activated."

# ---------------------------------------------------------------------------
# Step 4: Install dependencies
# ---------------------------------------------------------------------------
info "Installing project dependencies (this may take a few minutes)..."
uv pip install ".[dev]"
success "Dependencies installed."

# ---------------------------------------------------------------------------
# Step 5: Create .env from .env.example if missing
# ---------------------------------------------------------------------------
info "Checking .env file..."
if [[ -f ".env" ]]; then
    warn ".env already exists — skipping creation. Review it manually."
else
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        success ".env created from .env.example"
        warn "IMPORTANT: Open .env and replace all CHANGE_ME placeholders with real values."
        warn "NEVER commit .env to git."
    else
        warn ".env.example not found — cannot create .env automatically."
    fi
fi

# ---------------------------------------------------------------------------
# Step 6: Run DB migrations (if PostgreSQL is reachable)
# ---------------------------------------------------------------------------
info "Attempting to run database migrations..."
if [[ -f ".env" ]]; then
    # Source .env to get POSTGRES_DSN
    set -a
    # shellcheck disable=SC1091
    source .env 2>/dev/null || true
    set +a
fi

if [[ -n "${POSTGRES_DSN:-}" ]] && [[ "$POSTGRES_DSN" != *"CHANGE_ME"* ]]; then
    if alembic upgrade head 2>/dev/null; then
        success "Database migrations applied."
    else
        warn "Migration failed — check that PostgreSQL is running and POSTGRES_DSN is correct."
        warn "Run 'make migrate' manually once PostgreSQL is available."
    fi
else
    warn "POSTGRES_DSN is not set or not configured — skipping migrations."
    warn "Run 'make migrate' after configuring your database."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
success "Bootstrap complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit .env and fill in all CHANGE_ME values."
echo "  2. Start services: docker compose up -d postgres redis"
echo "  3. Apply migrations: make migrate"
echo "  4. Run tests: make test-unit"
echo "  5. Start the trader: make docker-up"
echo ""
echo "  IMPORTANT: The system starts in TESTNET mode by default."
echo "  Read RUNBOOK.md before attempting LIVE trading."
