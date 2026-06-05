# Architecture Decision Records

This document records the key architectural decisions made during the design of the Bybit AI Trader, including the rationale for each choice.

---

## ADR-001: Why pybit over ccxt for execution

**Decision**: Use the official `pybit` SDK for order submission and WebSocket connectivity.

**Rationale**:
- `pybit` is the officially maintained Bybit SDK. It receives API updates first and covers v5 endpoints comprehensively.
- `ccxt` provides a unified interface across exchanges, but abstractions introduce latency (extra serialisation/deserialisation layers) and lag behind exchange-specific features.
- For a single-exchange system, the portability benefit of `ccxt` does not outweigh the latency and feature-lag costs.
- `pybit` supports async WebSocket natively, which is critical for the hot path.
- Bybit's v5 unified account API is fully supported in `pybit >= 5.8.0`.

**Trade-offs**:
- System is locked to Bybit. Migrating to another exchange would require significant rework of the execution layer.

---

## ADR-002: Why asyncio over threading

**Decision**: The entire trading system is built on Python's `asyncio` event loop with a single-process async architecture.

**Rationale**:
- WebSocket feeds, REST calls, database queries, and Redis operations are all I/O-bound. Asyncio excels here with minimal overhead.
- Avoiding threads eliminates most race conditions in shared state management. The single-threaded event loop model makes reasoning about order of operations straightforward.
- CPU-bound work (model inference, feature computation) is offloaded to separate worker processes or `asyncio.to_thread()`, not OS threads.
- `asyncpg`, `aiohttp`, `redis[asyncio]`, and `asyncio` WebSocket clients all support the async model natively.

**Trade-offs**:
- A blocking call anywhere on the event loop can stall the hot path. All blocking calls must be audited to use `run_in_executor` or equivalent.
- Debugging async stack traces is harder than synchronous code.

---

## ADR-003: Why PostgreSQL as primary store

**Decision**: PostgreSQL 16 via asyncpg and SQLAlchemy (async) as the primary persistent store.

**Rationale**:
- Audit trail, order history, trade fills, and reconciliation records require ACID guarantees. NoSQL databases cannot provide the consistency needed for financial data.
- PostgreSQL's JSONB columns allow flexible payload storage for events without sacrificing query capability.
- TimescaleDB (a PostgreSQL extension) can be added later for high-frequency market data without changing the ORM layer.
- `asyncpg` is the fastest async PostgreSQL driver available for Python.
- PostgreSQL supports row-level locking, which is important for the reconciliation loop.

**Trade-offs**:
- PostgreSQL requires more operational overhead than SQLite or Redis-only solutions.
- Schema migrations (Alembic) add complexity to deployment.

---

## ADR-004: Why PPO first, SAC as challenger

**Decision**: Proximal Policy Optimisation (PPO) is the primary RL algorithm. Soft Actor-Critic (SAC) is the challenger model.

**Rationale**:
- PPO is stable, sample-efficient for discrete/continuous action spaces, and has well-understood hyperparameter sensitivity. It is the most widely validated algorithm in financial RL literature.
- PPO's clipped objective prevents catastrophic policy updates — a critical safety property for live trading.
- SAC's entropy maximisation makes it more exploratory and often achieves better asymptotic performance, but requires more careful tuning.
- Running SAC in shadow mode as a challenger allows continuous A/B testing without production risk.
- `stable-baselines3` provides production-grade PPO/SAC implementations with Gymnasium compatibility.

**Trade-offs**:
- PPO requires on-policy data, making it less sample-efficient than SAC in some regimes.
- Neither algorithm has guarantees of profitability.

---

## ADR-005: Why hot/slow path separation

**Decision**: The system separates a "hot path" (latency-sensitive) from a "slow path" (throughput-sensitive) with distinct processing budgets.

**Hot path** (target: <5ms end-to-end from WS message to order intent):
- WebSocket message receipt
- Feature vector computation
- Model inference
- Risk check
- Order submission

**Slow path** (relaxed latency, run as background tasks):
- Reconciliation
- Model retraining
- Telegram notifications
- Database writes (async, non-blocking)
- Report generation

**Rationale**:
- Crypto markets move fast. Any unnecessary work on the critical path increases slippage.
- Background tasks that block the event loop can cause missed market data or late order submissions.
- Separating concerns makes it easier to profile and optimise each path independently.

**Trade-offs**:
- More complex code structure.
- Requires careful task priority management in asyncio.

---

## ADR-006: Why LLM cannot trade directly

**Decision**: The LLM (if enabled) is limited to providing textual analysis/commentary. It has no ability to submit orders, modify risk parameters, or bypass the Risk Manager.

**Rationale**:
- LLMs are probabilistic and can hallucinate. Allowing an LLM to directly trigger orders creates an unacceptable tail risk of adversarial or nonsensical trades.
- Prompt injection via news feeds is a documented attack vector. An LLM with direct trading capability would be a high-value target.
- The LLM's output feeds into the human-review or regime-classification pipeline only. The RL model's policy, not the LLM, drives trading decisions.
- This satisfies the principle of minimal authority: each component has only the permissions it needs.

**Trade-offs**:
- Limits the system's ability to react to unstructured qualitative signals (e.g., regulatory news).
- LLM value is confined to advisory/monitoring functions.

---

## ADR-007: Why the Risk Manager is the final authority

**Decision**: Every order intent, regardless of source (RL model, strategy, operator command), must pass through the Risk Manager before submission. The Risk Manager's decision is final and cannot be overridden programmatically.

**Rationale**:
- Financial loss is irreversible. The Risk Manager is the last line of defence against the model generating harmful trades during distribution shift, edge cases, or system faults.
- A single authoritative risk gate prevents multiple code paths from bypassing limits independently.
- The Risk Manager enforces portfolio-level invariants (max heat, drawdown limits) that no individual strategy can see holistically.
- This design mirrors institutional risk architecture (pre-trade risk checks).

**Trade-offs**:
- The Risk Manager becomes a bottleneck. Its code must be exceptionally well-tested and efficient.
- Operator intervention (emergency override) requires a separate kill-switch path that is itself audited.

---

## ADR-008: Redis for in-process state cache

**Decision**: Redis is used for short-lived shared state (latest prices, feature vectors, regime context) between the trader-core process and background workers.

**Rationale**:
- Redis pub/sub enables loose coupling between the core process and workers without a message broker.
- Feature vectors and prices have a short TTL (seconds to minutes). Redis TTL handles expiry automatically.
- Redis is significantly faster than PostgreSQL for frequent reads of small objects.
- Redis Streams could be adopted for the event bus if message persistence is needed in Phase 2.

**Trade-offs**:
- Redis is not ACID. It cannot replace PostgreSQL for audit-critical data.
- Redis data is lost on restart unless persistence (AOF/RDB) is configured.
