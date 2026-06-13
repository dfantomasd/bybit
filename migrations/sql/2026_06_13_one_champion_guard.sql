WITH ranked AS (
    SELECT model_id,
           row_number() OVER (
               ORDER BY
                   CASE WHEN COALESCE(
                       NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                       NULLIF(metrics->>'wf_mean_bps', ''),
                       NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                   )::double precision > 0 THEN 0 ELSE 1 END,
                   COALESCE(
                       NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                       NULLIF(metrics->>'wf_mean_bps', ''),
                       NULLIF(metrics->>'best_threshold_avg_net_return_bps', ''),
                       '-1000000'
                   )::double precision DESC,
                   COALESCE(NULLIF(metrics->>'lift_bps', ''), '0')::double precision DESC,
                   training_finished_at DESC NULLS LAST,
                   created_at DESC
           ) AS rn
    FROM model_versions
    WHERE status = 'CHAMPION'
)
UPDATE model_versions mv
SET status = 'ARCHIVED'
FROM ranked
WHERE mv.model_id = ranked.model_id
  AND ranked.rn > 1;

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_versions_one_champion
    ON model_versions ((status))
    WHERE status = 'CHAMPION';
