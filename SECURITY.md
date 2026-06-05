# Security Policy

## Reporting Vulnerabilities

This is a private, personal-use system. If you discover a security vulnerability, please follow responsible disclosure:

1. Do **not** open a public GitHub issue for security vulnerabilities.
2. Contact the maintainer privately (use the email in your git config).
3. Include a description, reproduction steps, and impact assessment.
4. Allow 30 days for remediation before any public disclosure.

---

## API Key Management

### Generation
- Generate a **dedicated** Bybit API key for this bot. Never reuse keys shared with other applications.
- Enable only the minimum required permissions: **read account info**, **create/cancel orders**.
- Enable IP whitelist restriction to the server's IP address.
- Disable withdrawal permissions entirely.

### Storage
- API keys are stored as environment variables or Docker secrets only.
- Keys are never written to disk in plaintext outside of the `.env` file.
- `.env` must be in `.gitignore` (it is). Verify with `git status` before every commit.
- In production, use Docker secrets (`/run/secrets/`) instead of `.env`.
- Never pass API keys as command-line arguments (they appear in `ps` output).

### Rotation
- Rotate API keys every 90 days or immediately upon suspected compromise.
- See RUNBOOK.md for the rotation procedure.

### Revocation
- If a key is compromised, revoke it immediately via the Bybit web dashboard.
- The exchange API key management page is: https://www.bybit.com/app/user/api-management

---

## Secret Handling in Code

All secret fields in the codebase use `pydantic.SecretStr`:
- `BYBIT_API_KEY`
- `BYBIT_API_SECRET`
- `POSTGRES_DSN` (contains password)
- `REDIS_URL` (may contain password)
- `TELEGRAM_BOT_TOKEN`

`SecretStr` ensures:
- `str(secret)` returns `'**********'`
- `repr(secret)` returns `SecretStr('**********')`
- The actual value is only accessible via `.get_secret_value()`

The structlog configuration includes a secret redaction processor that additionally scans log records for field names matching known patterns (`api_key`, `secret`, `token`, `password`, etc.) and replaces their values with `***REDACTED***`.

**Never call `.get_secret_value()` outside of the minimal scope that requires it.**

---

## Dependency Management

- Dependencies are pinned with minimum version bounds in `pyproject.toml`.
- `pip-audit` runs in CI on every push to check for known vulnerabilities.
- Trivy scans the Docker image for OS and library CVEs.
- Dependabot (or equivalent) should be configured to open PRs for security updates.
- Review `CHANGELOG` entries of all updated dependencies before merging.

---

## Container Security

- The runtime Docker image runs as non-root user `trader:1000`.
- `no-new-privileges:true` security option is set for all containers.
- The filesystem is read-only where possible.
- Only the Grafana port (3000) is exposed externally. All other services are on internal Docker networks.

---

## Network Security

### IP Whitelist
- Configure the Bybit API key to allow connections only from the trading server's static IP.
- Restrict inbound connections to the server to known management IPs.

### TLS
- All connections to Bybit (REST and WebSocket) use TLS.
- Never disable certificate verification.

### Firewall
- Expose only ports 3000 (Grafana) and your management SSH port externally.
- The FastAPI port (8080) should only be accessible from localhost or a VPN.

---

## Threat Model

See [THREAT_MODEL.md](THREAT_MODEL.md) for a full analysis of known attack vectors and mitigations.

---

## Audit Trail

Every significant action is logged as an `AuditEvent` record stored in PostgreSQL. The audit log is append-only and includes:
- Actor (system component)
- Action (what was done)
- Resource (what was acted upon)
- Outcome (success/failure)
- Timestamp and correlation ID

Do not delete or truncate the audit log table.
