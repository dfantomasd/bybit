# Bybit Trading Bot - Signal Generation & Label Flow Analysis

## Executive Summary

The Bybit trading bot has **CRITICAL ARCHITECTURAL ISSUES** that explain the weak model quality and negative walk-forward performance (-14.92 bps):

1. **Signal-to-Trade Disconnect**: Trading signals are labeled as if they were executed, regardless of whether actual trades occurred
2. **Counterfactual Training Data**: The model trains on "what would have happened" rather than "what actually happened"
3. **Unrealistic Label Threshold**: A 2 bps threshold in a 26 bps cost environment creates almost exclusively negative labels
4. **RULE_BASELINE_V1 Only**: Training only uses baseline rule-based signals, not actual live execution outcomes

---

## Part 1: Signal Generation Flow

### ScalpMicroStrategy (`src/trader/strategies/scalp_micro.py`)

**Entry Logic:**
- **EMA Cross Detection**: Fresh cross of EMA9 above/below EMA21 on confirmed bar (or trend continuation in shadow mode)
- **RSI Filter**: RSI14 < 70 (overbought) for buys, RSI14 > 30 (oversold) for sells
- **Volume Impulse**: Last candle volume > 1.5x SMA(volume, 20)
- **Spread Filter**: Fail-closed if spread > 3 bps or unknown
- **Price Bounce**: Current price must bounce from 5-candle low within ATR * 0.35 zone
- **Orderbook Imbalance**: Buy requires imbalance ≥ 0.15, Sell requires ≤ -0.15
- **Net Edge Check**: Gross edge (ATR * 1.0) minus costs (taker_fee + spread + slippage) must exceed 0.05%
- **ADX Filter**: ADX14 ≥ 20 (avoid flat markets) - relaxed to 0.15 in shadow mode

**Exits (Fixed):**
- TP = Entry ± ATR * 1.0
- SL = Entry ∓ ATR * 0.5
- Reward:Risk ≈ 2:1

**Position Sizing:**
```
qty_usd = available_balance_usd * 0.01 / (atr_pct * 0.5)
capped at min(100 USD, 30% of balance)
```

**Rate Limits:**
- Per-symbol cooldown: 60 seconds
- Global cap: 10 signals/minute

### ShadowProbeStrategy (`src/trader/strategies/shadow_probe.py`)

**Paper-Only Entry (SHADOW mode only):**
- **Orderbook Imbalance**: |imbalance| ≥ 0.05
- **EMA Confirmation**: EMA9 > EMA21 for buy (RSI < 68), EMA9 < EMA21 for sell (RSI > 32)
- **Minimal Quality Gate**: feature.quality_score ≥ 0.45
- **Broader TP/SL**: TP = ATR * 1.4, SL = ATR * 0.8
- **Cost-Aware Gate**: Requires min_net_return_pct ≥ 0.05%
- **Burst Limiter**: Max 3 signals/300s, then 600s cooldown
- **Position Cap**: Max 2-4 open positions

**Purpose:**
- Generate training labels for SHADOW outcomes
- Broad market coverage (not alpha-focused like scalp_micro)
- Collect data on TP/SL mechanics

---

## Part 2: Signal Recording & Prediction Events

### Flow: Trading Loop → Trade Journal (`src/trader/modules/trading_loop.py`)

```
TradeProposal (from strategy.evaluate)
    ↓
record_feature_snapshot() 
    → feature_snapshots table with feature vector
    ↓
record_prediction_event(model_version="RULE_BASELINE_V1")
    → prediction_events table
       - symbol, interval, model_version
       - score = proposal.confidence
       - strategy_signal = "Buy" or "Sell"
       - decision = "SHADOW_BASELINE" (for rule-based) or "GATE_PASS"/"GATE_BLOCK" (for model gate)
       - feature_snapshot_id (links to features)
    ↓
ExecutionEngine.submit()
    → RiskManager evaluation
    → Order placement or shadow log
    ↓
record_signal()
    → trade_signals table (proposal metadata)
```

**Key Insight:** `prediction_events` are created for **EVERY** signal, regardless of:
- Whether RiskManager approved or rejected it
- Whether an actual order was submitted
- Whether the trade was executed or remained pending

This means REJECTED signals still create prediction_events and get trained on.

---

## Part 3: Outcome Resolution (The Core Problem)

### Outcome Resolution Process (`src/trader/storage/directional_trade_journal.py`)

**Function:** `resolve_outcomes_from_candles()`

**Query Logic:**
```sql
SELECT prediction_id, symbol, strategy_signal, entry_time, feature_names, feature_values
FROM prediction_events pe
JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
LEFT JOIN prediction_outcomes po ON ...
WHERE po.prediction_id IS NULL  -- Not yet resolved
  AND pe.created_at < now() - (horizon_minutes * '1 minute')
  AND pe.strategy_signal IN ('Buy', 'Sell')
  AND (
    pe.decision IN ('GATE_PASS', 'GATE_BLOCK')
    OR fs.training_eligible = true
  )
```

**For each unresolved prediction_event:**

1. **Fetch entry candle** (at signal's creation time)
   - Use market_candles table
   - Entry price = candle close

2. **Fetch horizon path** (all candles for next N minutes)
   - Query market_candles for [entry_time, entry_time + horizon]
   - Require exactly horizon_minutes confirmations (fail if gap)

3. **Calculate outcome:**
```python
gross_return_bps = (exit_price - entry_price) / entry_price * 10,000
mfe_bps = max favorable excursion (high for Buy, low for Sell)
mae_bps = max adverse excursion (low for Buy, high for Sell)

# With TP/SL enabled (MODEL_LABEL_USE_TPSL_EXIT=true):
exit_price = first touch of TP or SL, else horizon_close
```

4. **Apply cost model:**
```
cost_bps = 5.5 + 5.5 + 4.0 + 3.0 + 3.0 + 1.0 + 5.0 = 26.5 bps
net_return_bps = gross_return_bps - cost_bps

label = 1 if net_return_bps > label_threshold_bps (default 2.0), else 0
```

5. **Persist outcome:**
```sql
INSERT INTO prediction_outcomes (
    prediction_id, horizon_minutes,
    gross_return_bps, cost_bps, net_return_bps,
    mfe_bps, mae_bps, label, label_schema_version
)
```

---

## Part 4: THE CRITICAL GAP - Signals vs Actual Execution

### What the system SHOULD do:

```
Signal → Risk Evaluation → Approved/Rejected
  ↓ (only if approved)
Actual Trade Executed
  ↓
Outcome from actual position close (TP hit, SL hit, forced close, etc)
  ↓
Label based on REALIZED outcome
```

### What the system ACTUALLY does:

```
Signal → prediction_event created
  ↓ (regardless of execution)
Risk Evaluation → Approved/Rejected (may never happen)
  ↓ (outcome resolved either way)
Outcome from market candles (counterfactual path)
  ↓
Label based on "what would have been" if trade was executed
```

### Concrete Example:

**Scenario 1: Signal Rejected by Risk Manager**
- Strategy generates BUY signal for BTCUSDT at $50,000
- RiskManager rejects due to exposure cap
- **No trade is placed**
- **Outcome is still resolved** from market candles
- If price goes to $50,100, label = 1 (positive)
- But the trade was **never placed**, so no actual PnL occurred

**Scenario 2: Shadow Mode Signal**
- ShadowProbeStrategy generates paper signal
- Feature snapshot + prediction_event recorded
- **No live order submitted**
- Outcome resolved from candles as if it were executed
- Training data includes **counterfactual** outcomes

**Scenario 3: Pending Order That Gets Stuck**
- Signal approved, order submitted to Bybit
- Order sits in order book for 5 minutes (maker-first mode)
- Never fills before candle closes
- **Outcome resolved for unfilled order** 
- Path may show positive return, but actual PnL = 0

---

## Part 5: Training Data Corruption

### Current Training Query (`src/trader/training/train.py`):

```sql
SELECT feature_names, feature_values, net_return_bps, label
FROM feature_snapshots fs
JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
WHERE pe.model_version = 'RULE_BASELINE_V1'
  AND po.label IS NOT NULL
  AND po.label_threshold_bps = 2.0
  AND fs.training_eligible = true
  AND pe.strategy_signal IN ('Buy', 'Sell')
```

**This trains on:**
- ✓ All baseline rule signals (scalp_micro, shadow_probe, trend_ema, etc.)
- ✗ Regardless of execution status
- ✗ Regardless of risk decision (approved/rejected)
- ✗ Counterfactual market paths, not actual trading outcomes
- ✗ Only RULE_BASELINE_V1 (not actual live model decisions)

**No filtering for:**
- `pe.decision` = "GATE_PASS" ← would at least limit to gated signals
- Actual order_events with FILLED status
- Realized PnL from closed_pnl table

---

## Part 6: Label Threshold Problem

### Cost Breakdown (Documented in `directional_trade_journal.py`):

```python
DEFAULT_TAKER_FEE_BPS = 5.5 * 2 = 11 bps  (round-trip)
DEFAULT_SPREAD_BPS = 4.0 bps               (scalp max spread)
DEFAULT_SLIPPAGE_PER_SIDE_BPS = 3.0 * 2 = 6 bps
DEFAULT_FUNDING_BPS = 1.0 bps
DEFAULT_SAFETY_MARGIN_BPS = 5.0 bps       (mirrors net_edge_safety_margin_pct)
────────────────────────────────────────────
Total = 27 bps
```

### Label Distribution:

**For scalp_micro with ATR-based TP/SL:**
- Expected gross return: TP distance = ATR * 1.0
- ATR for high-volatility alts: 0.5% - 1.0% → 50-100 bps gross
- For low-volatility pairs (BTC, ETH): 0.1% - 0.3% → 10-30 bps gross

**Example (typical ETHUSDT scalp):**
```
Entry: 2500.00
TP: 2502.50 (ATR * 1.0 = 100 bps)
Gross return: 100 bps
Costs: 27 bps
Net return: 73 bps
Label (threshold 2 bps): 1 ✓

But if realized cost is higher (5 bps spread instead of 4 bps):
Net return: 68 bps → still 1

However, if only 15 bps gross (smaller ATR):
Gross: 15 bps
Net: 15 - 27 = -12 bps
Label: 0 ✗ (negative outcome)
```

**The Problem:**
- Most micro-scalp targets: 10-20 bps gross
- Many signals fail to achieve TP distance before SL triggered
- Label distribution is heavily skewed toward 0 (negative)
- Precision of baseline rules: ~35% (example from logs)
- Model cannot beat 65% baseline of always predicting class 0

---

## Part 7: Why Walk-Forward is Negative (-14.92 bps)

### Walk-Forward Validation Loop:

```python
folds = [
    (train=[0:1000], val=[1000:1250]),    # train on old data, test on newer
    (train=[0:1250], val=[1250:1500]),
    (train=[0:1500], val=[1500:1750]),
    (train=[0:1750], val=[1750:2000]),
    (train=[0:2000], val=[2000:2250]),
]

for train_idx, val_idx in folds:
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[val_idx])
    avg_net_return_bps = mean(returns_bps[val_idx][preds == 1])
    fold_metrics.append(avg_net_return_bps)

expectancy_bps = mean(fold_metrics)  # -14.92 bps
```

**Why Negative:**

1. **Counterfactual labels decay over time**
   - Early signals marked positive because counterfactual path was favorable
   - But real execution never happened, so no actual edge
   - Later validation folds see learned "edges" that don't exist

2. **Overfitting to market regime**
   - Training includes all market regimes mixed together
   - Model learns: "in low-volatility sideways markets, signals fail"
   - But labels never reflect actual risk decision (approved/rejected)
   - Learns spurious: "low volatility + signal → negative" (true for counterfactual, false for execution)

3. **Baseline is too strong**
   - RULE_BASELINE_V1 is already well-optimized (net_edge_safety_margin_pct applied)
   - Adding a model on top just adds complexity
   - Without actual execution data, model cannot improve the baseline

4. **Label schema mismatch**
   - TP/SL exit prices (MODEL_LABEL_USE_TPSL_EXIT) assume perfect execution
   - Real market fills occur at worse prices during slippage
   - Model learns TP/SL path, but gets penalized on validation with actual spread/slip

---

## Part 8: The Complete Data Flow with Issues Marked

```
Strategy Layer
├── scalp_micro.evaluate()        → TradeProposal(confidence, entry, tp, sl)
│   └── Issues:
│       - Net edge threshold (0.05%) is threshold for signal, not for training label
│       - Accepted signals may still get rejected downstream
│
└── shadow_probe.evaluate()       → TradeProposal(confidence, entry, tp, sl)
    └── Issues:
        - Only in SHADOW mode (paper trading)
        - Outcomes resolved but trades never placed
        - Creates training data from counterfactuals

        ↓
    
Risk Decision Layer
├── RiskManager.evaluate()        → RiskDecision(APPROVED/REJECTED/RESIZED)
│   └── Issues:
│       - Rejection happens AFTER signal recorded
│       - Rejected signals still get outcomes resolved
│       - No way to know downstream if trade was approved
│
└── ExecutionEngine.submit()      → OrderIntent (or shadow log)
    └── Issues:
        - Pending orders may not fill before horizon
        - No connection to prediction_events table
        - No way to know if outcome corresponds to executed trade

        ↓

Trade Journal Layer
├── record_signal()               → trade_signals table
│   └── Metadata only (proposal info)
│
├── record_feature_snapshot()     → feature_snapshots table
│   └── Feature vector at signal time
│
├── record_prediction_event()     → prediction_events table
│   ├── model_version='RULE_BASELINE_V1'
│   ├── decision='SHADOW_BASELINE'
│   ├── strategy_signal='Buy'/'Sell'
│   └── Issues:
│       - No link to actual execution
│       - No link to RiskDecision
│       - No link to OrderIntent
│       - Created for ALL signals, approved or not
│
└── record_order_event()          → order_events table
    └── Issues:
        - order_link_id does NOT appear in prediction_events
        - No way to join outcomes back to actual orders
        - Paper orders recorded same as real orders

        ↓

Outcome Resolution Layer
└── resolve_outcomes_from_candles()
    ├── Uses market_candles[entry_time : entry_time + horizon]
    ├── Calculates gross_return_bps (COUNTERFACTUAL)
    ├── Applies cost_model (27 bps)
    ├── Creates label (threshold 2 bps)
    └── Issues:
        - No verification trade was actually placed
        - No verification trade was actually filled
        - No verification outcome matches reality
        - Labels are "what would have been" not "what was"

        ↓

Training Layer
└── train.py
    ├── Query joins prediction_events → prediction_outcomes
    ├── Filter: pe.model_version='RULE_BASELINE_V1'
    ├── No filter: pe.decision or execution status
    ├── Trains on counterfactual labels
    └── Result: WEAK quality, -14.92 bps walk-forward

        ↓

Model Deployment
└── Canary/Live Gate
    ├── Model trained on counterfactual data
    ├── Applied to live signals (which also have issues)
    ├── Cannot improve on baseline
    └── Result: Negative lift, unsafe to use
```

---

## Part 9: Summary of Root Causes

### Critical Issues (Prevent Model Improvement):

1. **No Execution Verification**
   - Signals are labeled regardless of execution status
   - Rejected/shadow signals get same treatment as executed trades
   - No link between prediction_events and actual fills

2. **Counterfactual Training Data**
   - Labels computed from market path, not actual PnL
   - No distinction between "signal would have been profitable" and "signal actually made money"
   - RULE_BASELINE is paper, not live execution

3. **Training Data Filtering**
   - Only uses `pe.model_version='RULE_BASELINE_V1'`
   - No filter for `pe.decision IN ('GATE_PASS', 'GATE_BLOCK')`
   - No filter for actual order_events with FILLED status
   - Includes all signals regardless of risk approval

4. **Label Threshold Mismatch**
   - 2 bps threshold in 27 bps cost environment
   - Results in heavily skewed labels (mostly negative)
   - Baseline precision (35%) already near ceiling of what's learnable

### Secondary Issues (Reduce Quality):

5. **TP/SL Exit Assumption**
   - Labels use ATR-based TP/SL path
   - Actual fills occur at market (worse prices)
   - Model learns idealized exit, validated on realistic exit

6. **Regime Bucketing Not Used**
   - Metadata includes regime context
   - Training doesn't filter by regime
   - Model conflates bull/bear/sideways markets

7. **Strategy Mixing**
   - Multiple strategies trained together
   - Scalp_micro and shadow_probe have different costs/expectations
   - Model doesn't specialize

---

## Part 10: How to Fix It

### Phase 1: Fix Training Data (Immediate - 2-3 hours)

**Change prediction_outcomes query to require execution:**

```python
# In training/train.py, add filter to eligible_samples:
AND (
    pe.decision IN ('GATE_PASS', 'GATE_BLOCK')
    OR fs.training_eligible = true AND pe.decision = 'SHADOW_BASELINE'
)
AND EXISTS (
    SELECT 1 FROM order_events oe
    WHERE oe.order_link_id = ANY(
        SELECT order_link_id FROM durable_order_state
        WHERE status IN ('FILLED', 'PARTIALLY_FILLED')
    )
)
```

Actually, the simpler fix:

```python
# Use GATE_PASS only (signals that passed model gate):
AND pe.decision = 'GATE_PASS'
AND po.label IS NOT NULL
```

Or focus on closed_pnl (actual realized):
```python
# Join to closed_pnl for realized outcomes
AND EXISTS (
    SELECT 1 FROM closed_pnl cp
    WHERE cp.symbol = fs.symbol
    AND cp.created_at BETWEEN 
        pe.created_at AND pe.created_at + (horizon * interval '1 minute')
)
```

### Phase 2: Link Execution to Outcomes (Medium - 1-2 days)

**Add order_link_id to prediction_events:**
```sql
ALTER TABLE prediction_events
ADD COLUMN order_link_id text,
ADD COLUMN execution_status text;

CREATE INDEX idx_prediction_events_order_link
ON prediction_events (order_link_id);
```

**Wire execution_engine to record prediction_event link:**
```python
# In execution_engine.submit()
prediction_id = await self._trade_journal.record_prediction_event(...)
await self._trade_journal.link_order_to_prediction(order_link_id, prediction_id)
```

**Verify fills before resolving outcomes:**
```python
# In resolve_outcomes_from_candles()
# Check order_events for actual FILLED status before creating label
for pe in unresolved_events:
    fills = await journal.fetch("SELECT * FROM order_events WHERE ... AND status='FILLED'")
    if not fills:
        continue  # Skip this signal, it never executed
```

### Phase 3: Fix Label Threshold (Quick - 30 minutes)

**Raise threshold to match actual costs:**
```python
# In config.py or .env:
MODEL_AUTO_TRAIN_LABEL_BPS = 15.0  # Instead of 2.0

# This makes labels = 1 only when net_return > 15 bps
# Which is realistic after 27 bps costs
```

### Phase 4: Separate Training by Strategy (Medium - 4-6 hours)

**Train separate models for scalp_micro vs shadow_probe:**
```python
# In train.py:
strategy_models = {}
for strategy_id in ['scalp_micro_v1', 'shadow_probe_hv_v2', ...]:
    strategy_models[strategy_id] = train_on_strategy(strategy_id)
```

### Phase 5: Live Execution Feedback Loop (Advanced - 1-2 weeks)

**Track actual closed positions and compute real outcomes:**
```python
# New table: realized_outcomes
# Triggered when closed_pnl recorded

# Query actual closed_pnl instead of counterfactual candle paths:
SELECT
    pe.prediction_id,
    cp.closed_pnl,
    cp.avg_entry_price,
    cp.avg_exit_price,
    ...
FROM prediction_events pe
JOIN closed_pnl cp ON 
    cp.symbol = pe.symbol AND
    cp.created_at > pe.created_at AND
    cp.created_at < pe.created_at + interval '1 hour'
WHERE pe.order_link_id = cp.order_link_id  -- or similar join
```

Then train on realized_outcomes, not counterfactual candle-based outcomes.

---

## Conclusion

The Bybit bot's weak model quality is **not due to weak signals or poor risk management**, but rather:

**Signals → Outcomes mismatch**: The system trains models on "what would have been profitable if the trade was executed" rather than "what was actually profitable when the trade was executed."

This is a **fundamental data pipeline issue**, not a model architecture issue. The walk-forward negative performance (-14.92 bps) is the model correctly identifying that it cannot improve over the baseline when trained on counterfactual data.

The fix is to **connect trading signals to their actual execution outcomes**, then retrain on realized PnL, not simulated candle paths.

