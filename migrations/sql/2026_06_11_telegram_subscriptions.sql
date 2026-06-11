-- Telegram push-notification subscriptions.
--
-- Purpose: subscriptions used to live only in memory and were lost on every
-- restart/redeploy. This table persists them so /subscribe survives restarts.
--
-- NOTE: this DDL is also applied automatically by TradeJournal._ensure_schema()
-- at startup; this file exists for manual/offline application and audit.

CREATE TABLE IF NOT EXISTS telegram_subscriptions (
    chat_id bigint PRIMARY KEY,
    subscribed_at timestamptz NOT NULL DEFAULT now()
);
