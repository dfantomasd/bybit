# Operations Runbook

This runbook covers operational procedures for the Bybit AI Trader in production.

> **WARNING**: Read this document completely before operating the system in any mode other than TESTNET.

---

## Starting the Bot

### Prerequisites
1. `.env` is configured with valid API keys and database credentials.
2. PostgreSQL and Redis are running and reachable.
3. Migrations have been applied: `make migrate`.
4. You are intentionally in the correct mode (confirm `TRADING_MODE` in `.env`).

### Procedure
```bash
# Start all services
docker compose up -d

# Verify health (wait ~60 seconds for startup)
curl http://localhost:8080/health

# Watch logs for preflight output
docker compose logs -f trader-core
```

### Expected startup sequence
```
[INFO] settings_loaded trading_mode=TESTNET risk_profile=CONSERVATIVE
[INFO] observability_configured
[INFO] preflight_check_passed check=postgres
[INFO] preflight_check_passed check=redis
[INFO] preflight_check_passed check=bybit_connectivity
[INFO] preflight_passed
[INFO] http_server_starting port=8080
[INFO] trading_system_running trading_mode=TESTNET
```

If any preflight check fails, the system exits with code 1.

---

## Stopping Safely

### Graceful stop (preferred)
```bash
# Sends SIGTERM — triggers graceful shutdown
docker compose stop trader-core

# Allow up to 30 seconds for in-flight operations to complete
docker compose down
```

### What happens during graceful shutdown
1. `STOPPING` status is set.
2. Open order cancellations are requested (Phase 2).
3. WebSocket connections are closed cleanly.
4. Database connections are released.
5. `STOPPED` status is logged.

### Verify clean stop
```bash
docker compose ps
# trader-core should show 'exited (0)'

# Check for any orphaned open orders via Bybit dashboard
```

---

## Emergency Kill Switch

Use when the system must stop immediately (e.g., runaway losses, exchange connectivity issues, suspected compromise).

### Level 1: Pause new entries
```bash
# Environment variable override — restart required
TRADING_MODE=SHADOW docker compose up -d trader-core
```

### Level 2: Cancel all open orders
- Via Bybit web dashboard: Account → Orders → Cancel All
- Or use Bybit app: same path

### Level 3: Full stop + close all positions
```bash
# Hard stop the bot
docker compose kill trader-core

# Manually close all positions via Bybit dashboard or app
# Account → Positions → Close All
```

### After an emergency stop
1. Review logs: `docker compose logs trader-core > incident-$(date +%Y%m%d).log`
2. Check Bybit dashboard for any unfilled orders or unexpected positions.
3. Run reconciliation manually after restarting.
4. Document the incident.

---

## Investigating Reconciliation Mismatches

Reconciliation mismatches occur when local order state diverges from exchange state.

### Signs of a mismatch
- Log message: `reconciliation_discrepancies_found discrepancies=N`
- Prometheus metric: `trader_reconciliation_discrepancies_total > 0`
- `OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED` in the database

### Investigation steps
```bash
# 1. View recent reconciliation logs
docker compose logs trader-core | grep reconciliation

# 2. Query the database for unknown-state orders
psql $POSTGRES_DSN -c "
  SELECT order_link_id, status, created_at, updated_at
  FROM orders
  WHERE status = 'UNKNOWN_RECONCILIATION_REQUIRED'
  ORDER BY created_at DESC
  LIMIT 20;
"

# 3. Cross-check with Bybit dashboard
# Bybit: Account → Order History → filter by date range

# 4. If order is filled on exchange but local shows unknown:
#    Update local status manually (use audit-logged admin command — Phase 2)
```

### Common causes
- Network timeout during order submission (order may or may not have reached exchange)
- WebSocket reconnect during order lifecycle
- Clock skew between server and exchange

---

## Handling Model Drift

Model drift occurs when the live data distribution diverges from the training distribution, degrading model performance.

### Detection
- Prometheus metric: `trader_model_drift_score{model_id="..."} > 0.3`
- Evidently reports: weekly drift analysis in `/app/mlflow/drift-reports/`
- Performance degradation: declining Sharpe ratio in the reporting worker output

### Response procedure

**Minor drift (score 0.2–0.4)**:
1. Review the drift report to identify which features are drifting.
2. Trigger a model retrain with updated data: `docker compose exec trainer-worker python -m trader.workers.trainer --retrain`
3. Run the retrained model in shadow mode for 24–48 hours.
4. Compare shadow performance vs. live model.
5. If shadow model is better, promote it.

**Major drift (score > 0.4)**:
1. Switch trading mode to SHADOW immediately.
2. Notify team.
3. Investigate root cause (regime change, market microstructure change, data pipeline issue).
4. Retrain with appropriate data.
5. Validate thoroughly before returning to live.

---

## Database Backup and Restore

### Backup
```bash
# Manual backup
docker compose exec postgres pg_dump -U trader trader | \
  gzip > trader-backup-$(date +%Y%m%d-%H%M%S).sql.gz

# The backup should be stored off-server (S3, etc.)
# Automate with cron or a backup service
```

### Restore
```bash
# STOP THE TRADER FIRST
docker compose stop trader-core

# Restore from backup
gunzip -c trader-backup-YYYYMMDD.sql.gz | \
  docker compose exec -T postgres psql -U trader trader

# Verify restoration
docker compose exec postgres psql -U trader -c "\dt" trader

# Restart the trader
docker compose start trader-core
```

---

## Adding New Symbols

1. **Verify the symbol exists on Bybit** and meets minimum liquidity requirements.
2. **Update `config/symbols.yaml`**:
   ```yaml
   whitelist:
     - BTCUSDT
     - ETHUSDT
     - SOLUSDT   # ← new symbol
   ```
3. **Retrain or fine-tune** the model on the new symbol's historical data (Phase 2).
4. **Test in shadow mode** for at least 48 hours before enabling live trading.
5. **Restart the trader**: `docker compose restart trader-core`

---

## Rotating API Keys

Rotate every 90 days or immediately upon suspected compromise.

### Procedure
1. **Generate new API key** on Bybit dashboard.
   - Copy the new key and secret immediately (shown only once).
   - Apply the same IP whitelist and permission set as the old key.

2. **Update `.env`**:
   ```bash
   nano .env
   # Update BYBIT_API_KEY and BYBIT_API_SECRET
   ```

3. **Restart the trader**:
   ```bash
   docker compose restart trader-core
   ```

4. **Verify connectivity**:
   ```bash
   curl http://localhost:8080/health
   # bybit_rest should be true
   ```

5. **Revoke the old key** on the Bybit dashboard.

---

## Upgrading Dependencies

### Procedure
```bash
# Check for outdated packages
uv pip list --outdated

# Run pip-audit to check for vulnerabilities
pip-audit

# Update specific package
uv pip install --system "package>=new.version"

# Run full test suite
make test

# Review CHANGELOG of updated packages for breaking changes

# Rebuild Docker image
make docker-build

# Test in dev environment
make docker-up-dev

# If all good, deploy to production
make docker-up
```

### After security vulnerability patches
Follow the same procedure but treat it as urgent (same-day if CRITICAL severity).

---

## Monitoring and Alerting

### Grafana dashboards
Access at: http://localhost:3000 (admin / see `secrets/grafana_password.txt`)

Key dashboards (Phase 2 — to be created):
- System Overview: status, mode, health
- Trading Performance: PnL, Sharpe, drawdown
- Execution Quality: slippage, fill rates
- Risk Monitor: heat, drawdown, circuit breakers

### Prometheus alerts (Phase 2)
Configure alerting rules for:
- `trader_daily_drawdown_pct > max_daily_drawdown_pct`
- `trader_component_healthy{component="postgres"} == 0`
- `trader_ws_connection_status == 0` for > 60 seconds
- `trader_model_drift_score > 0.3`
