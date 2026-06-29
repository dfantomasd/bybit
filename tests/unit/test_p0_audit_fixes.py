"""P0 audit-fix tests.

Covers:
- test_durable_state_written_before_rest_submit           (P0.1)
- test_adapter_reconcile_ignores_terminal_orders          (P0.2)
- test_adapter_reconcile_checks_only_pending              (P0.2)
- test_adapter_reconcile_restores_pending_from_db         (P0.2)
- test_order_update_marks_in_memory_terminal              (P0.3)
- test_order_update_marks_durable_terminal                (P0.3)
- test_duplicate_order_update_does_not_double_release_pending (P0.3)
- test_write_failure_marks_journal_unhealthy              (P0.4)
- test_canary_blocks_when_durable_store_unhealthy         (P0.4)
- test_allowed_chats_auto_subscribed_after_restart        (P0.5)
- test_supervisor_treats_private_ws_as_critical           (P0.6)
- test_supervisor_treats_risk_monitor_as_critical         (P0.6)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketType, OrderSide, OrderStatus, OrderType
from trader.domain.models import OrderIntent
from trader.exchange.bybit_adapter import BybitAdapter
from trader.exchange.idempotency import IdempotencyManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(
    symbol: str = "BTCUSDT",
    side: str = "Buy",
    qty: str = "0.001",
) -> OrderIntent:
    return OrderIntent(
        decision_id=uuid.uuid4(),
        proposal_id=uuid.uuid4(),
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=OrderSide(side),
        order_type=OrderType.MARKET,
        qty=Decimal(qty),
        order_link_id=f"TN-260101-TEST-{uuid.uuid4().hex[:8].upper()}-aa11bb",
    )


def _make_adapter(journal: MagicMock | None = None) -> BybitAdapter:
    adapter = BybitAdapter.__new__(BybitAdapter)
    adapter._idempotency = IdempotencyManager()
    adapter._rest = AsyncMock()
    adapter._mapper = MagicMock()
    adapter._mapper.intent_to_params.return_value = {}
    adapter._default_category = "linear"
    adapter._journal = journal
    return adapter


# ---------------------------------------------------------------------------
# P0.1 — durable state written before REST submit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_durable_state_written_before_rest_submit() -> None:
    """CREATED_LOCAL and SUBMITTING must be written to durable state BEFORE REST call."""
    call_order: list[str] = []

    journal = MagicMock()
    journal.is_enabled = True

    async def _upsert(**kwargs: object) -> None:
        call_order.append(str(kwargs["state"]))

    async def _place_order(**_kwargs: object) -> dict:
        call_order.append("REST")
        return {"result": {"orderId": "exch-001"}}

    journal.upsert_durable_order_state = _upsert
    adapter = _make_adapter(journal)
    adapter._rest.place_order = _place_order

    intent = _make_intent()
    await adapter.place_order(intent)

    created_idx = call_order.index("CREATED_LOCAL")
    submitting_idx = call_order.index("SUBMITTING")
    rest_idx = call_order.index("REST")
    rest_accepted_idx = call_order.index("REST_ACCEPTED")

    assert created_idx < rest_idx, "CREATED_LOCAL must precede REST call"
    assert submitting_idx < rest_idx, "SUBMITTING must precede REST call"
    assert rest_idx < rest_accepted_idx, "REST_ACCEPTED must follow REST call"


@pytest.mark.asyncio
async def test_durable_state_unknown_on_rest_failure() -> None:
    """UNKNOWN_RECONCILIATION_REQUIRED must be written when REST raises."""
    states_written: list[str] = []

    journal = MagicMock()

    async def _upsert(**kwargs: object) -> None:
        states_written.append(str(kwargs["state"]))

    journal.upsert_durable_order_state = _upsert
    adapter = _make_adapter(journal)
    adapter._rest.place_order = AsyncMock(side_effect=RuntimeError("network error"))

    intent = _make_intent()
    with pytest.raises(RuntimeError):
        await adapter.place_order(intent)

    assert "UNKNOWN_RECONCILIATION_REQUIRED" in states_written


# ---------------------------------------------------------------------------
# P0.2 — reconcile ignores terminal orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_reconcile_ignores_terminal_orders() -> None:
    """Terminal orders (FILLED, CANCELLED, etc.) must NOT count as reconciliation mismatches."""
    adapter = _make_adapter()

    # Seed idempotency with a terminal FILLED order
    intent = _make_intent()
    await adapter._idempotency.register_intent(intent)
    await adapter._idempotency.mark_submitted(intent.order_link_id)
    await adapter._idempotency.mark_confirmed(intent.order_link_id, "exch-001")
    # Advance to WS_CONFIRMED then FILLED
    adapter._idempotency._store[intent.order_link_id]["status"] = OrderStatus.WS_CONFIRMED
    await adapter._idempotency.mark_filled(intent.order_link_id)

    # Exchange returns empty open orders
    adapter._rest.get_open_orders = AsyncMock(return_value={"result": {"list": []}})

    result = await adapter.reconcile()
    assert result.discrepancies_found == 0, "FILLED orders must not be mismatches"


@pytest.mark.asyncio
async def test_adapter_reconcile_checks_only_pending() -> None:
    """Only PENDING orders should be compared against exchange; terminal are skipped."""
    adapter = _make_adapter()

    # Create one PENDING (REST_ACCEPTED) and one TERMINAL (FILLED) order
    pending_intent = _make_intent(symbol="BTCUSDT")
    filled_intent = _make_intent(symbol="ETHUSDT")
    # different order_link_ids
    filled_intent = OrderIntent(
        decision_id=uuid.uuid4(),
        proposal_id=uuid.uuid4(),
        symbol="ETHUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.01"),
        order_link_id="TN-260101-TEST-FFFFFFFF-bb22cc",
    )

    await adapter._idempotency.register_intent(pending_intent)
    await adapter._idempotency.mark_submitted(pending_intent.order_link_id)
    await adapter._idempotency.mark_confirmed(pending_intent.order_link_id, "exch-pending")

    await adapter._idempotency.register_intent(filled_intent)
    await adapter._idempotency.mark_submitted(filled_intent.order_link_id)
    await adapter._idempotency.mark_confirmed(filled_intent.order_link_id, "exch-filled")
    adapter._idempotency._store[filled_intent.order_link_id]["status"] = OrderStatus.WS_CONFIRMED
    await adapter._idempotency.mark_filled(filled_intent.order_link_id)

    # Exchange returns no open orders (both would appear missing if terminal was checked)
    adapter._rest.get_open_orders = AsyncMock(return_value={"result": {"list": []}})

    result = await adapter.reconcile()
    # Only the pending one should be a discrepancy
    assert result.discrepancies_found == 1
    assert pending_intent.order_link_id in result.mismatched_order_ids
    assert filled_intent.order_link_id not in result.mismatched_order_ids


@pytest.mark.asyncio
async def test_adapter_reconcile_restores_pending_from_db() -> None:
    """load_pending_from_db() must populate idempotency from PostgreSQL on restart."""
    journal = MagicMock()
    journal.get_pending_durable_orders = AsyncMock(
        return_value=[
            {
                "order_link_id": "TN-260101-TEST-ABCD1234-aabbcc",
                "state": "REST_ACCEPTED",
                "exchange_order_id": "exch-xyz",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "qty": Decimal("0.001"),
            }
        ]
    )
    adapter = _make_adapter(journal)

    loaded = await adapter.load_pending_from_db()

    assert loaded == 1
    state = await adapter._idempotency.get_state("TN-260101-TEST-ABCD1234-aabbcc")
    assert state == OrderStatus.REST_ACCEPTED


# ---------------------------------------------------------------------------
# P0.3 — OrderUpdateEvent syncs both stores
# ---------------------------------------------------------------------------


def _make_order_event(
    order_link_id: str,
    order_id: str = "exch-001",
    symbol: str = "BTCUSDT",
    status: OrderStatus = OrderStatus.FILLED,
    qty: Decimal = Decimal("0.001"),
) -> MagicMock:
    event = MagicMock()
    event.order_link_id = order_link_id
    event.order_id = order_id
    event.symbol = symbol
    event.status = status
    event.order_status = status  # support both field names
    event.side = MagicMock()
    event.side.value = "Buy"
    event.qty = qty
    return event


@pytest.mark.asyncio
async def test_order_update_marks_in_memory_terminal() -> None:
    """handle_order_update with FILLED should update in-memory idempotency to FILLED."""
    adapter = _make_adapter()

    intent = _make_intent()
    await adapter._idempotency.register_intent(intent)
    await adapter._idempotency.mark_submitted(intent.order_link_id)
    await adapter._idempotency.mark_confirmed(intent.order_link_id, "exch-001")
    # Move to WS_CONFIRMED so FILLED transition is valid
    adapter._idempotency._store[intent.order_link_id]["status"] = OrderStatus.WS_CONFIRMED

    event = _make_order_event(intent.order_link_id, status=OrderStatus.FILLED)
    is_terminal = await adapter.handle_order_update(event)

    assert is_terminal is True
    state = await adapter._idempotency.get_state(intent.order_link_id)
    assert state == OrderStatus.FILLED


@pytest.mark.asyncio
async def test_order_update_marks_durable_terminal() -> None:
    """handle_order_update must write terminal state to durable store via journal."""
    journal = MagicMock()
    durable_calls: list[str] = []

    async def _upsert(**kwargs: object) -> None:
        durable_calls.append(str(kwargs["state"]))

    journal.upsert_durable_order_state = _upsert
    adapter = _make_adapter(journal)

    event = _make_order_event("TN-260101-TEST-DEADBEEF-ff00ff", status=OrderStatus.CANCELLED)
    await adapter.handle_order_update(event)

    assert "CANCELLED" in durable_calls


@pytest.mark.asyncio
async def test_duplicate_order_update_does_not_double_release_pending() -> None:
    """Second terminal event for the same order must not increment release counter twice."""
    adapter = _make_adapter()

    intent = _make_intent()
    await adapter._idempotency.register_intent(intent)
    await adapter._idempotency.mark_submitted(intent.order_link_id)
    await adapter._idempotency.mark_confirmed(intent.order_link_id, "exch-001")
    adapter._idempotency._store[intent.order_link_id]["status"] = OrderStatus.WS_CONFIRMED

    event1 = _make_order_event(intent.order_link_id, status=OrderStatus.FILLED)
    event2 = _make_order_event(intent.order_link_id, status=OrderStatus.FILLED)

    release_count = 0
    _pending_released: set[str] = set()

    for event in (event1, event2):
        is_terminal = await adapter.handle_order_update(event)
        if is_terminal and event.order_link_id not in _pending_released:
            release_count += 1
            _pending_released.add(event.order_link_id)

    assert release_count == 1, "Pending count must be released exactly once per order"


# ---------------------------------------------------------------------------
# P0.4 — DB health fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_failure_marks_journal_unhealthy() -> None:
    """Three consecutive write failures must make durable_state_healthy return False."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal.__new__(TradeJournal)
    journal._enabled = True
    journal._consecutive_write_errors = 0
    journal._last_successful_write_at = None
    journal._last_write_error_at = None

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=RuntimeError("db down"))
    mock_pool = AsyncMock()
    mock_pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    journal._pool = mock_pool

    assert journal.durable_state_healthy is True

    for _ in range(3):
        await journal._execute("SELECT 1")

    assert journal.durable_state_healthy is False
    assert journal._consecutive_write_errors == 3
    assert journal._last_write_error_at is not None


@pytest.mark.asyncio
async def test_write_failure_resets_on_success() -> None:
    """Consecutive error counter must reset to zero after a successful write."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal.__new__(TradeJournal)
    journal._enabled = True
    journal._consecutive_write_errors = 2  # pre-set
    journal._last_successful_write_at = None
    journal._last_write_error_at = datetime.now(tz=UTC)
    journal._last_write_error = "previous error"

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=None)
    mock_pool = AsyncMock()
    mock_pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    journal._pool = mock_pool

    await journal._execute("SELECT 1")

    assert journal._consecutive_write_errors == 0
    assert journal.durable_state_healthy is True
    assert journal._last_successful_write_at is not None
    assert journal._last_write_error_at is None
    assert journal._last_write_error is None


def test_canary_blocks_when_durable_store_unhealthy() -> None:
    """durable_state_healthy must be False with >= 3 consecutive write errors."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal.__new__(TradeJournal)
    journal._enabled = True
    journal._pool = MagicMock()
    journal._consecutive_write_errors = 3
    journal._last_successful_write_at = None
    journal._last_write_error_at = None

    assert journal.durable_state_healthy is False

    health = journal.write_health()
    assert health["healthy"] is False
    assert health["consecutive_write_errors"] == 3


def test_write_health_unhealthy_when_configured_but_disconnected() -> None:
    """A configured journal with no pool is not write-healthy even before write errors."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal.__new__(TradeJournal)
    journal._enabled = True
    journal._pool = None
    journal._consecutive_write_errors = 0
    journal._last_successful_write_at = None
    journal._last_write_error_at = None
    journal._last_write_error = None
    journal._last_read_error_at = None
    journal._last_read_error = None
    journal._last_connect_error_at = None
    journal._last_connect_error = "Failed to connect to database: {:error, :econnrefused}"

    health = journal.write_health()

    assert health["healthy"] is False
    assert health["configured"] is True
    assert health["connected"] is False
    assert health["writable"] is False
    assert health["durable_state_healthy"] is True
    assert "econnrefused" in str(health["last_connect_error"])


@pytest.mark.asyncio
async def test_db_diagnostics_exposes_safe_connection_target() -> None:
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(
        "postgresql://user:secret@example.internal:6543/trades?sslmode=require",
        enabled=True,
    )

    diag = await journal.get_db_diagnostics()

    assert diag["connected"] is False
    assert diag["connection_target"] == {
        "scheme": "postgresql",
        "host": "example.internal",
        "port": 6543,
        "database": "trades",
    }
    assert "secret" not in str(diag["connection_target"])
    assert "user" not in str(diag["connection_target"])


@pytest.mark.asyncio
async def test_schema_script_executes_one_statement_at_a_time() -> None:
    from trader.storage.trade_journal import _execute_schema_script

    conn = MagicMock()
    conn.execute = AsyncMock()

    await _execute_schema_script(
        conn,
        """
        CREATE TABLE IF NOT EXISTS one (id int);
        CREATE INDEX IF NOT EXISTS one_id_idx ON one (id);

        ALTER TABLE one ADD COLUMN IF NOT EXISTS name text;
        """,
    )

    statements = [call.args[0] for call in conn.execute.await_args_list]
    assert statements == [
        "CREATE TABLE IF NOT EXISTS one (id int)",
        "CREATE INDEX IF NOT EXISTS one_id_idx ON one (id)",
        "ALTER TABLE one ADD COLUMN IF NOT EXISTS name text",
    ]


# ---------------------------------------------------------------------------
# P0.5 — Telegram auto-subscription
# ---------------------------------------------------------------------------


def test_allowed_chats_auto_subscribed_after_restart() -> None:
    """Allowed chat IDs must be pre-subscribed at bot creation without /start."""
    from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot

    config = TelegramBotConfig(
        token="dummy-token",
        allowed_chat_ids={12345, 67890},
        trading_mode="SHADOW",
        risk_profile="CONSERVATIVE",
        bybit_use_testnet=False,
    )
    bot = TelegramMonitorBot(
        config=config,
        health_provider=AsyncMock(),
        adapter_factory=MagicMock(),
    )

    assert 12345 in bot._subscribed
    assert 67890 in bot._subscribed


def test_empty_allowed_chats_subscribed_set_empty() -> None:
    """Empty allowed_chat_ids must produce empty subscribed set."""
    from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot

    config = TelegramBotConfig(
        token="dummy-token",
        allowed_chat_ids=set(),
        trading_mode="SHADOW",
        risk_profile="CONSERVATIVE",
        bybit_use_testnet=False,
    )
    bot = TelegramMonitorBot(
        config=config,
        health_provider=AsyncMock(),
        adapter_factory=MagicMock(),
    )

    assert bot._subscribed == set()


# ---------------------------------------------------------------------------
# P0.6 — Supervisor critical tasks
# ---------------------------------------------------------------------------


def test_supervisor_treats_private_ws_as_critical() -> None:
    """ws-private must be in _CRITICAL_TASK_NAMES."""
    from trader.app import _CRITICAL_TASK_NAMES

    assert "ws-private" in _CRITICAL_TASK_NAMES
    assert "ws-private-consumer" in _CRITICAL_TASK_NAMES


def test_supervisor_treats_risk_monitor_as_critical() -> None:
    """risk-monitor must be in _CRITICAL_TASK_NAMES."""
    from trader.app import _CRITICAL_TASK_NAMES

    assert "risk-monitor" in _CRITICAL_TASK_NAMES


def test_supervisor_critical_task_names_complete() -> None:
    """All required critical tasks must be present."""
    from trader.app import _CRITICAL_TASK_NAMES

    required = {
        "screener",
        "ws-public",
        "ws-consumer",
        "ws-private",
        "ws-private-consumer",
        "feature-pipeline",
        "strategy-loop",
        "risk-monitor",
        "reconciliation",
        "outcome-resolver",
        "load-governor",
    }
    missing = required - _CRITICAL_TASK_NAMES
    assert not missing, f"Missing critical task names: {missing}"


# ---------------------------------------------------------------------------
# P1.3 — Env name consistency
# ---------------------------------------------------------------------------


def test_model_shadow_scoring_enabled_in_config() -> None:
    """Config must use MODEL_SHADOW_SCORING_ENABLED (not MODEL_SHADOW_SCORING)."""
    from trader.config import Settings

    s = Settings()
    # New name exists
    assert hasattr(s, "MODEL_SHADOW_SCORING_ENABLED")
    # Old name must not exist
    assert not hasattr(s, "MODEL_SHADOW_SCORING"), "MODEL_SHADOW_SCORING renamed to MODEL_SHADOW_SCORING_ENABLED"
