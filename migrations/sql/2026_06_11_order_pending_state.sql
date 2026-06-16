-- Persistent pending-entry resolution state.
--
-- Purpose: the in-memory pending-entry set in ExecutionEngine is lost on
-- restart. If an order reached a terminal status (FILLED/CANCELLED/...) just
-- before a crash, the restored pending slot would block new entries for that
-- symbol forever. This table records each pending registration and its
-- resolution timestamp so restarts only restore genuinely unresolved orders.
--
-- NOTE: this DDL is also applied automatically by TradeJournal._ensure_schema()
-- at startup; this file exists for manual/offline application and audit.

CREATE TABLE IF NOT EXISTS order_pending_state (
    order_link_id text PRIMARY KEY,
    symbol text,
    created_at timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_order_pending_state_unresolved
    ON order_pending_state (created_at DESC) WHERE resolved_at IS NULL;

-- Hybrid ML mode: record when the model replaced the rule-based decision
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS model_decision jsonb;
