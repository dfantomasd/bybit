# Render Free Deployment

This project can run on Render Free as a read-only monitoring service.

## What Runs

- One Render Free Web Service.
- External PostgreSQL via Supabase.
- No Redis by default.
- Telegram commands are read-only: `/status`, `/balance`, `/positions`, `/help`.

## Create The Service

1. In Render, create a new Blueprint from this repository.
2. Select the `bybit-monitor` service from `render.yaml`.
3. Keep the plan as `Free`.
4. Fill the secret environment variables:

```env
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
POSTGRES_DSN=postgresql+asyncpg://...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_CHAT_IDS=-1003976706688
```

Use the Supabase pooler URI, but replace `postgresql://` with `postgresql+asyncpg://`.

## Keep It Awake

Render Free web services spin down after 15 minutes without inbound traffic.
Create a free UptimeRobot monitor that pings:

```text
https://YOUR_RENDER_SERVICE.onrender.com/livez
```

Use a 10-minute interval. One always-on Free web service consumes about 720-744
instance hours per month, within Render's 750-hour monthly free allowance.

## Safety

Keep these values unless you intentionally change modes:

```env
LIVE_MODE=false
SHADOW_MODE=true
```

The current runtime does not expose Telegram trading commands.
