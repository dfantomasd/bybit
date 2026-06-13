CREATE TABLE IF NOT EXISTS model_promotion_log (
    promotion_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type text NOT NULL,
    from_version text,
    to_version text,
    reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_model_promotion_log_created
    ON model_promotion_log (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_promotion_log_versions
    ON model_promotion_log (from_version, to_version, created_at DESC);
