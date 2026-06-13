"""Initial schema — all tables and columns from _ensure_schema().

Revision ID: 0001
Revises:
Create Date: 2026-06-13 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS durable_order_state (
            order_link_id text PRIMARY KEY,
            proposal_id uuid,
            decision_id uuid,
            symbol text NOT NULL,
            side text NOT NULL,
            qty numeric NOT NULL,
            state text NOT NULL,
            exchange_order_id text,
            payload_hash text,
            retry_count integer NOT NULL DEFAULT 0,
            last_error text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_durable_order_state_state
            ON durable_order_state (state, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_durable_order_state_symbol
            ON durable_order_state (symbol, created_at DESC);

        CREATE TABLE IF NOT EXISTS trade_signals (
            proposal_id uuid PRIMARY KEY,
            created_at timestamptz NOT NULL,
            strategy_id text NOT NULL,
            symbol text NOT NULL,
            side text NOT NULL,
            confidence double precision NOT NULL,
            entry_price numeric,
            take_profit numeric,
            stop_loss numeric,
            requested_qty numeric NOT NULL,
            requested_notional_usd numeric,
            regime text,
            rationale text,
            features jsonb,
            model_decision jsonb,
            blocked_reason text
        );
        CREATE INDEX IF NOT EXISTS idx_trade_signals_symbol_created
            ON trade_signals (symbol, created_at DESC);

        CREATE TABLE IF NOT EXISTS risk_decisions (
            decision_id uuid PRIMARY KEY,
            proposal_id uuid NOT NULL,
            created_at timestamptz NOT NULL,
            symbol text NOT NULL,
            status text NOT NULL,
            approved_qty numeric,
            approved_notional_usd numeric,
            reason text,
            triggered_rules jsonb NOT NULL,
            portfolio_heat double precision,
            current_drawdown_pct double precision,
            open_positions_count integer
        );
        CREATE INDEX IF NOT EXISTS idx_risk_decisions_symbol_created
            ON risk_decisions (symbol, created_at DESC);

        CREATE TABLE IF NOT EXISTS order_events (
            order_link_id text PRIMARY KEY,
            proposal_id uuid NOT NULL,
            decision_id uuid NOT NULL,
            created_at timestamptz NOT NULL,
            symbol text NOT NULL,
            side text NOT NULL,
            qty numeric NOT NULL,
            status text NOT NULL,
            exchange_order_id text,
            error text
        );
        CREATE INDEX IF NOT EXISTS idx_order_events_symbol_created
            ON order_events (symbol, created_at DESC);

        CREATE TABLE IF NOT EXISTS closed_pnl (
            closed_pnl_id text PRIMARY KEY,
            created_at timestamptz NOT NULL,
            symbol text NOT NULL,
            side text,
            qty numeric,
            avg_entry_price numeric,
            avg_exit_price numeric,
            closed_pnl numeric NOT NULL,
            raw jsonb NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_closed_pnl_symbol_created
            ON closed_pnl (symbol, created_at DESC);

        CREATE TABLE IF NOT EXISTS market_candles (
            symbol text NOT NULL,
            interval text NOT NULL,
            open_time timestamptz NOT NULL,
            close_time timestamptz NOT NULL,
            open numeric NOT NULL,
            high numeric NOT NULL,
            low numeric NOT NULL,
            close numeric NOT NULL,
            volume numeric NOT NULL,
            turnover numeric NOT NULL,
            confirmed boolean NOT NULL DEFAULT false,
            source text NOT NULL DEFAULT 'ws',
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, interval, open_time)
        );
        CREATE INDEX IF NOT EXISTS idx_market_candles_interval_time
            ON market_candles (interval, open_time DESC);
        CREATE INDEX IF NOT EXISTS idx_market_candles_symbol_interval_time
            ON market_candles (symbol, interval, open_time DESC);

        CREATE TABLE IF NOT EXISTS feature_snapshots (
            snapshot_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at timestamptz NOT NULL DEFAULT now(),
            symbol text NOT NULL,
            interval text NOT NULL,
            candle_open_time timestamptz NOT NULL,
            feature_schema_hash text NOT NULL,
            feature_names jsonb NOT NULL,
            feature_values jsonb NOT NULL,
            training_eligible boolean NOT NULL DEFAULT true,
            invalid_reason text,
            invalidated_at timestamptz
        );
        CREATE INDEX IF NOT EXISTS idx_feature_snapshots_symbol_time
            ON feature_snapshots (symbol, created_at DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_snapshots_unique_eligible
            ON feature_snapshots (symbol, interval, candle_open_time, feature_schema_hash)
            WHERE training_eligible = true;

        CREATE TABLE IF NOT EXISTS prediction_events (
            prediction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at timestamptz NOT NULL DEFAULT now(),
            symbol text NOT NULL,
            interval text NOT NULL,
            model_version text NOT NULL,
            feature_snapshot_id uuid,
            score double precision NOT NULL,
            strategy_signal text,
            decision text,
            metadata jsonb
        );
        CREATE INDEX IF NOT EXISTS idx_prediction_events_symbol_time
            ON prediction_events (symbol, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_prediction_events_model_time
            ON prediction_events (model_version, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_prediction_events_model_decision_time
            ON prediction_events (model_version, decision, created_at DESC);

        CREATE TABLE IF NOT EXISTS prediction_outcomes (
            prediction_id uuid NOT NULL REFERENCES prediction_events(prediction_id),
            horizon_minutes integer NOT NULL,
            net_return_bps double precision,
            max_favorable_excursion_bps double precision,
            max_adverse_excursion_bps double precision,
            label integer,
            resolved_at timestamptz,
            label_schema_version text DEFAULT 'directional_net_v1',
            gross_return_bps double precision,
            cost_bps double precision,
            label_threshold_bps double precision,
            online_learned_at timestamptz,
            PRIMARY KEY (prediction_id, horizon_minutes)
        );
        CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_label_schema
            ON prediction_outcomes (label_schema_version);
        CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_horizon_schema
            ON prediction_outcomes (horizon_minutes, label_schema_version)
            WHERE label IS NOT NULL;

        CREATE TABLE IF NOT EXISTS model_versions (
            model_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            version text NOT NULL UNIQUE,
            status text NOT NULL DEFAULT 'SHADOW_CHALLENGER',
            training_started_at timestamptz,
            training_finished_at timestamptz,
            training_samples integer,
            feature_schema_hash text,
            artifact bytea,
            metrics jsonb,
            created_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_model_versions_status
            ON model_versions (status, created_at DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_model_versions_one_champion
            ON model_versions ((status))
            WHERE status = 'CHAMPION';

        CREATE TABLE IF NOT EXISTS model_promotion_log (
            promotion_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            event_type text NOT NULL,
            decision text,
            challenger_version text,
            champion_version text,
            new_champion_version text,
            from_version text,
            to_version text,
            reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
            metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
            metrics_snapshot jsonb,
            decided_at timestamptz NOT NULL DEFAULT now(),
            created_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_model_promotion_log_created
            ON model_promotion_log (created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_model_promotion_log_versions
            ON model_promotion_log (from_version, to_version, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_model_promotion_log_decided_at
            ON model_promotion_log (decided_at DESC);

        CREATE TABLE IF NOT EXISTS training_runs (
            run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            model_version text,
            mode text NOT NULL DEFAULT 'offline',
            started_at timestamptz NOT NULL DEFAULT now(),
            finished_at timestamptz,
            status text NOT NULL DEFAULT 'RUNNING',
            sample_count integer,
            error text,
            metrics jsonb
        );

        CREATE TABLE IF NOT EXISTS execution_events (
            exec_id text PRIMARY KEY,
            order_link_id text,
            exchange_order_id text,
            symbol text NOT NULL,
            side text NOT NULL,
            exec_price numeric NOT NULL,
            exec_qty numeric NOT NULL,
            exec_fee numeric,
            exec_value numeric,
            is_maker boolean,
            closed_size numeric,
            proposal_id uuid,
            decision_id uuid,
            created_at timestamptz NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_execution_events_symbol
            ON execution_events (symbol, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_execution_events_order_link
            ON execution_events (order_link_id);

        CREATE TABLE IF NOT EXISTS account_transaction_events (
            id bigserial PRIMARY KEY,
            transaction_time timestamptz NOT NULL,
            symbol text,
            side text,
            funding numeric,
            fee numeric,
            fee_rate numeric,
            cash_flow numeric,
            change numeric,
            cash_balance numeric,
            trade_price numeric,
            trade_id text,
            order_id text,
            order_link_id text,
            transaction_type text,
            category text,
            created_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS account_transaction_events_trade_id_idx
            ON account_transaction_events (trade_id) WHERE trade_id IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS account_transaction_events_hash_idx
            ON account_transaction_events (transaction_time, symbol, COALESCE(trade_id,''), COALESCE(fee::text,''))
            WHERE trade_id IS NULL;

        CREATE TABLE IF NOT EXISTS order_pending_state (
            order_link_id text PRIMARY KEY,
            symbol text,
            created_at timestamptz NOT NULL DEFAULT now(),
            resolved_at timestamptz
        );
        CREATE INDEX IF NOT EXISTS idx_order_pending_state_unresolved
            ON order_pending_state (created_at DESC) WHERE resolved_at IS NULL;
        CREATE INDEX IF NOT EXISTS idx_order_pending_state_symbol_unresolved
            ON order_pending_state (symbol, created_at DESC) WHERE resolved_at IS NULL;

        CREATE TABLE IF NOT EXISTS telegram_subscriptions (
            chat_id bigint PRIMARY KEY,
            subscribed_at timestamptz NOT NULL DEFAULT now()
        );
    """)


def downgrade() -> None:
    op.execute("""
        DROP TABLE IF EXISTS telegram_subscriptions;
        DROP TABLE IF EXISTS order_pending_state;
        DROP TABLE IF EXISTS account_transaction_events;
        DROP TABLE IF EXISTS execution_events;
        DROP TABLE IF EXISTS training_runs;
        DROP TABLE IF EXISTS model_promotion_log;
        DROP TABLE IF EXISTS model_versions;
        DROP TABLE IF EXISTS prediction_outcomes;
        DROP TABLE IF EXISTS prediction_events;
        DROP TABLE IF EXISTS feature_snapshots;
        DROP TABLE IF EXISTS market_candles;
        DROP TABLE IF EXISTS closed_pnl;
        DROP TABLE IF EXISTS order_events;
        DROP TABLE IF EXISTS risk_decisions;
        DROP TABLE IF EXISTS trade_signals;
        DROP TABLE IF EXISTS durable_order_state;
    """)
