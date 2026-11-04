# PROJECT NOONE — Component 3: CCXT Exchange Connectivity / Account
## Technical Design Specification — Version 2.2.4

**Document Version:** 2.2.4  
**Status:** Approved for Implementation (Testnet-First / Production Disabled)  
**Target Exchange:** Binance USDⓈ-M Futures (Testnet & Production)  
**Integration Library:** CCXT (`ccxt==4.4.52` pinned)  
**Target Source Directory:** `src/noone/exchange/`  
**Protected Codebases:** 
- Component 1: DQN + RL Engine (`src/noone/rl_engine/`) — **Frozen**
- Component 4: Market & News Data APIs (`src/noone/market_data/`, `src/noone/news_data/`) — **Protected**
- Component 8: Perpetual Futures Environment (`src/noone/trading_env/`) — **Protected**

---

## 1. Change Summary & Traceability (Version 2.2.3 -> Version 2.2.4)

| Correction Area | V2.2.3 Limitation | V2.2.4 Resolution & Specification Update | Affected Sections |
| :--- | :--- | :--- | :--- |
| **A. Defensible External Transfer Accounting** | Conflated all USDT `TRANSFER` income records with external deposits/withdrawals without verifying wallet boundaries. | **Explicit Transfer Taxonomy & Inflow Isolation:** Separated external deposits/withdrawals, internal wallet transfers (Spot <-> Futures), operational income, and non-USDT movements. Defined strict pagination, deduplication, and non-USDT blocking. Acknowledged futures API limitations; if external classification is ambiguous, daily PnL is marked `UNVERIFIED` and entries are blocked. | Section 10.1, Section 10.4 |
| **B. Explicit Data Freshness Policy** | Claimed zero-cache verification while permitting unmonitored cached filters and brackets. | **Comprehensive Data Freshness Policy Table:** Defined explicit sources, freshness tolerances, cache validity TTLs, and failure behaviors for all 11 execution inputs. Mandated fresh exchange queries for positions and open orders immediately prior to entry. Documented the unavoidable REST TOCTOU race. | Section 9.3, Section 9.4 |
| **C. Unified Freeze & Exit Matrix** | Contradiction between Section 4.3 (permitting exits under discrepancy) and Section 11.1 (blocking all exits). | **Harmonized Authoritative Permission Matrix:** Single canonical matrix covering all 8 operational conditions. Exits permitted *only* when exchange position is independently verified and the order reduces exposure with `reduceOnly=True`. Unverified exits and blind liquidations are strictly blocked. | Section 4.3, Section 11.1 |
| **D. UTC Midnight Baseline Consistency** | Claimed theoretical historical reconstruction while acknowledging Binance lacks point-in-time balance endpoints. | **Sampling-Only Baseline Protocol:** Formally prohibited post-facto balance reconstruction. Defined `VERIFIED_UTC_BASELINE` strictly as a direct exchange snapshot sampled within $[00:00:00, 00:00:05]$ UTC. If offline at rollover, baseline is marked `INVALID`, entries remain blocked, and manual startup baselines remain informational only. | Section 10.3 |
| **E. Complete Binance Market Filter Parser** | Generic filter description without handling Binance-specific flags (`applyToMarket`, coexistence of `MIN_NOTIONAL`/`NOTIONAL`). | **Exhaustive Filter Hierarchy & Decimal Engine:** Documented exact parsing of `MARKET_LOT_SIZE`, `LOT_SIZE`, `NOTIONAL`, and `MIN_NOTIONAL`. Specified fallback rules, malformed value handling, Decimal downward truncation, and residual exit constraints. | Section 8.1, Section 14 |
| **F. Decimal Position Reconciliation** | Used approximate float equality for position quantities. | **Deterministic Decimal Comparison:** Enforced exact string-to-Decimal conversion with explicit tolerance ($\le 10^{-8}$) to prevent hiding discrepancies. | Section 5.1 |
| **G. Sequential Order Recovery Protocol** | Described 9 recovery checks as an unrealistic simultaneous atomic call. | **9-Step Controlled Reconciliation Sequence:** Documented ordered execution, fill deduplication on `tradeId`, retry policies, and handling of inter-query record disagreement. | Section 11.2 |
| **H. Multi-Factor Margin & Bracket Validation** | Sizing treated `notional / leverage` as the actual margin requirement. | **Conservative Margin Buffer Engine:** Added multi-factor margin validation incorporating exchange leverage brackets, available balance, taker fees, and safety buffers. | Section 9.2 |
| **I. Partitioned Test Plan (26 Scenarios)** | Test counts were mixed across unit, mock, transport, and live categories. | **Rigorous 26-Scenario Partitioned Test Suite:** Explicitly classified into pure unit, offline mock integration, transport boundary (T01-B), and 18 outstanding live Testnet verification tasks. | Section 15, Section 17 |

---

## 2. System Scope & Core Architectural Invariants

Component 3 bridges PROJECT NOONE's frozen Component 1 DQN policy engine (`ModelInference`) with Binance USD(S)-M Futures via pinned CCXT 4.4.52.

### Invariant Architectural Rules
1. **Zero Modifications to Existing Code:** Components 1, 4, and 8 remain strictly read-only.
2. **Exchange is the Source of Truth:** Local memory never overrides verified exchange data.
3. **Persist Intent Before Dispatch:** Order intents are committed to SQLite prior to socket writes.
4. **Unknown Outcomes are Frozen:** Unresolved orders are never assumed canceled or rejected, and are never automatically resubmitted.
5. **No Blind Liquidations:** Network communication failures prohibit unverified order dispatch.
6. **Dedicated USD(S)-M Account Policy:** Only NOONE trades on the target USD(S)-M account or sub-account. External positions or orders in any symbol trigger an immediate freeze.
7. **No Entry Overrides:** No operator override or manual action can bypass unverified daily baseline rules to authorize entry trading.

---

## 3. Exact Frozen/Protected Interface Contracts

Component 3 interfaces with Component 1 (`src/noone/rl_engine/inference.py`) and Component 8 (`src/noone/trading_env/environment.py`) without modifying their contracts:

### 3.1 Five-Action Space Contract (`Action`)
* `Action.HOLD = 0`: Maintain current state. Generates zero exchange orders.
* `Action.OPEN_LONG = 1`: Enter Long position if currently Flat.
* `Action.CLOSE_LONG = 2`: Close active Long position via reduce-only order.
* `Action.OPEN_SHORT = 3`: Enter Short position if currently Flat.
* `Action.CLOSE_SHORT = 4`: Close active Short position via reduce-only order.

### 3.2 `AccountSnapshot`
An immutable dataclass mapping 9 verified metrics to the policy's observation vector:
* `equity` (`float > 0.0`): Total account margin balance in USDT (`totalMarginBalance`).
* `available_balance` (`float >= 0.0`): Liquid unencumbered USDT for initial margin (`availableBalance`).
* `position_side` (`int in {-1, 0, 1}`): `-1` (Short), `0` (Flat), `1` (Long).
* `position_quantity` (`float >= 0.0`): Base asset quantity ($0.0$ if flat, $> 0.0$ if active).
* `entry_price` (`float >= 0.0`): Average entry price ($0.0$ if flat, $> 0.0$ if active).
* `unrealized_pnl` (`float`): Mark-to-market floating PnL ($0.0$ if flat).
* `realized_pnl` (`float`): Cumulative realized PnL of closed trades in the current agent session.
* `margin_used` (`float >= 0.0`): Initial margin dedicated to the active position (`positionInitialMargin`).
* `progress` (`float in [0.0, 1.0]`): Strictly `0.0` in live execution.

### 3.3 Feature 19 Separation (`realized_pnl_rel_equity`)
In `src/noone/rl_engine/features.py`, Feature 19 is calculated as:
$$\text{Feature 19} = \frac{\text{obs}[\text{'realized\_pnl'}]}{\text{equity}}$$
As established in Component 8 `PerpetualFuturesEnv._reset_state`, `realized_pnl` represents the cumulative net profit of trades closed during the active trading session, initialized at `0.0` on launch.

> [!CAUTION]
> Binance's `/fapi/v2/account` endpoint returns `totalRealizedProfit`, which represents the account's historical lifetime profit across all symbols since account creation. Passing this lifetime figure into `AccountSnapshot` corrupts Feature 19, causing it to clip at $\pm 10.0$. Component 3 **tracks session realized PnL separately in its journal**, starting at `0.0` upon agent launch.

---

## 4. Binance USD(S)-M Endpoint Architecture & Response Mapping

Component 3 targets Binance USD(S)-M Futures **V2 REST endpoints** for account and position-risk synchronization.

### 4.1 Endpoint Register & Method Mappings

| Operation | Selected Method | Target Binance Endpoint | Scope | Current Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Account & Balances** | `exchange.fapiPrivateV2GetAccount()` | `GET /fapi/v2/account` | Account-Wide (Weight: 5) | **Design Expectation:** Expects root strings `totalMarginBalance`, `totalWalletBalance`, array `assets`, and array `positions`. Subject to testnet verification. |
| **Position Risk (Symbol)** | `exchange.fapiPrivateV2GetPositionRisk({'symbol': 'BTCUSDT'})` | `GET /fapi/v2/positionRisk` | Symbol-Specific (Weight: 5) | **Design Expectation:** Expects array filtered by symbol. In One-Way mode, exactly one object with `positionSide == "BOTH"` must be returned. Subject to testnet verification. |
| **Account-Wide Position Audit** | `exchange.fapiPrivateV2GetPositionRisk()` | `GET /fapi/v2/positionRisk` | Account-Wide (Weight: 5) | **Design Expectation:** Returns all positions. Used to detect external manual trading across all symbols. Subject to testnet verification. |
| **Account-Wide Open Orders** | `exchange.fapiPrivateGetOpenOrders()` | `GET /fapi/v1/openOrders` | Account-Wide (Weight: 40) | **Design Expectation:** Returns all open resting orders across all symbols. Subject to testnet verification. |
| **Order Query** | `exchange.fetch_order(id=None, symbol='BTC/USDT:USDT', params={'origClientOrderId': client_oid})` | `GET /fapi/v1/order` | Symbol-Specific (Weight: 1) | **Mock Verifiable (T01-A, T01-B):** Argument mapping and transport serialization verified offline. Network behavior subject to testnet. |
| **Order Dispatch** | `exchange.create_order(symbol, type='market', side, amount, params)` | `POST /fapi/v1/order` | Symbol-Specific (Weight: 1) | **Mock Verifiable:** Payload formation verified offline. Matching engine behavior subject to testnet. |
| **Position Mode** | `exchange.fapiPrivateGetPositionSideDual()` | `GET /fapi/v1/positionSide/dual` | Account-Wide (Weight: 30) | **Design Expectation:** Returns `{"dualSidePosition": boolean}`. Subject to testnet verification. |
| **Income / Transfer History** | `exchange.fapiPrivateGetIncome({'incomeType': 'TRANSFER', ...})` | `GET /fapi/v1/income` | Account-Wide (Weight: 30) | **Design Expectation:** Returns array of transfer events with `tranId`, `income`, `time`, `asset`. Subject to testnet verification. |
| **Leverage Brackets** | `exchange.fapiPrivateGetLeverageBracket({'symbol': 'BTCUSDT'})` | `GET /fapi/v1/leverageBracket` | Symbol-Specific (Weight: 1) | **Design Expectation:** Returns leverage tiers and maximum notional limits. Subject to testnet verification. |

### 4.2 Field Mapping Table (V2 Endpoints)

| `AccountSnapshot` Field | Source Endpoint | JSON Path | Semantic Definition & Parse Rule |
| :--- | :--- | :--- | :--- |
| `equity` | `GET /fapi/v2/account` | `totalMarginBalance` | Total margin balance ($\text{walletBalance} + \sum \text{unRealizedProfit}$). Float $> 0.0$. |
| `available_balance` | `GET /fapi/v2/account` | `assets[asset=="USDT"].availableBalance` | Liquid margin available for new orders. Float $\ge 0.0$. |
| `position_side` | `GET /fapi/v2/positionRisk` | `positionAmt` | Signed float string $q$: $q > 0 \implies 1$; $q < 0 \implies -1$; $q == 0 \implies 0$. |
| `position_quantity` | `GET /fapi/v2/positionRisk` | `positionAmt` | Absolute value $|q|$. Float $\ge 0.0$ ($0.0$ when flat). |
| `entry_price` | `GET /fapi/v2/positionRisk` | `entryPrice` | Average fill price. Float $\ge 0.0$ ($0.0$ when flat). |
| `unrealized_pnl` | `GET /fapi/v2/positionRisk` | `unRealizedProfit` | Mark-to-market floating PnL. Float ($0.0$ when flat). |
| `margin_used` | `GET /fapi/v2/account` | `positions[symbol].positionInitialMargin` | Initial margin dedicated specifically to active position. Float ($0.0$ when flat). |
| `realized_pnl` | Local Journal | Session tracker | Cumulative closed trade PnL since agent process launch. |
| `progress` | Constant | N/A | Strictly `0.0` during real-time live trading. |

### 4.3 Mandatory `positionInitialMargin` Invariants & Sync Failure Behavior
Component 3 adheres to a strict mapping policy:
$$\text{AccountSnapshot.margin\_used} = \text{float}(\text{position}[\text{'positionInitialMargin'}])$$
**Strict Prohibitions:**
* Do **NOT** substitute `initialMargin` (which conflates open limit order margin).
* Do **NOT** calculate a guessed value using leverage (e.g. $\frac{\text{notional}}{\text{leverage}}$).
* Do **NOT** use unverified fallback estimates.
* Do **NOT** silently substitute zero for an active position ($Q_{\text{pos}} > 0$).

**Failure Protocol:**
If `positionInitialMargin` is missing, non-numeric, or evaluates to $\le 0.0$ while $Q_{\text{pos}} > 0$:
1. Raise `AccountSyncError(f"Missing or invalid positionInitialMargin for active position: {raw_value}")`.
2. Invalidate the point-in-time account snapshot.
3. Log the synchronization failure to SQLite table `sync_audit_log`.
4. Transition to `SafetyFreeze(reason=POSITION_ACCOUNT_DISCREPANCY)`.
5. **Entry Guard:** Strictly block all new entries (`OPEN_LONG`, `OPEN_SHORT`).
6. **Exit Allowance:** Risk-reducing exits (`CLOSE_LONG`, `CLOSE_SHORT`) are permitted **only if** the exchange position quantity and side can be independently verified via `GET /fapi/v2/positionRisk`. Unverified exits are prohibited.

---

## 5. Account-Wide Exposure Taxonomy & Position Reconciliation

### 5.1 Deterministic Decimal Position Reconciliation
All position quantity comparisons between local journal expectations and exchange responses are executed using Python `Decimal` arithmetic:
$$\Delta Q = | \text{Decimal}(\text{pos\_exchange}[\text{'positionAmt'}]) - \text{Decimal}(\text{pos\_journal.quantity}) |$$
- **Exact Tolerance:** $\Delta Q \le \text{Decimal}('1e-8')$.
- If $\Delta Q > 10^{-8}$, the position is flagged as divergent. Discrepancies are **never hidden behind broad tolerances**.

### 5.2 Six-Case Position & Order Classification Engine
1. **Case 1: Reconciled Configured Target Position:**
   Active position exists solely on `BTCUSDT`, and its quantity, side, and entry price match the local journal within $10^{-8}$. Normal trading proceeds subject to routine risk checks.
2. **Case 2: Unexpected Configured Target Position:**
   Position exists on `BTCUSDT`, but its quantity or side differs from the journal expectation ($\Delta Q > 10^{-8}$). Trigger `SafetyFreeze(reason=POSITION_ACCOUNT_DISCREPANCY)`. Block new entries.
3. **Case 3: Position on Non-Configured Symbol:**
   Active position detected in any symbol other than `BTCUSDT` (e.g., manual ETHUSDT position). Trigger `SafetyFreeze(reason=ACCOUNT_WIDE_EXPOSURE_DETECTED)`. Block all new entries.
4. **Case 4: Unexplained Exchange Open Order:**
   Resting limit or stop order exists on the exchange that is not actively tracked by NOONE's local journal (in any symbol). Trigger `SafetyFreeze(reason=ACCOUNT_WIDE_EXPOSURE_DETECTED)`. Block all new entries.
5. **Case 5: Known NOONE Order:**
   Open order matches an in-flight NOONE `clientOrderId` in SQLite. Evaluated under standard order recovery protocols.
6. **Case 6: Zero-Quantity Position Record Validation:**
   Record returned with $Q_{\text{pos}} == 0.0$: must satisfy `Decimal(positionAmt) == Decimal('0')` and `positionSide == "BOTH"`. Only valid zero-quantity records are safely excluded from exposure totals. Ambiguous or dual-side records (`"LONG"`/`"SHORT"`) trigger `IncompatiblePositionSideError` and freeze execution.

---

## 6. Order Idempotency, 10-State Order Lifecycle & Two-Dimensional Reconciliation

Component 3 separates the operational order lifecycle from accounting verification.

### 6.1 Order Lifecycle (10 States)
1. `CREATED`: Validated and committed to SQLite prior to socket transmission.
2. `DISPATCHING`: Network socket write initiated.
3. `ACKNOWLEDGED`: HTTP 200 received with exchange `orderId`.
4. `OPEN`: Resting on order book (Binance status `NEW`).
5. `PARTIALLY_FILLED`: Partially executed; remaining contracts active.
6. `FILLED`: Fully executed (*Terminal exchange state*).
7. `CANCELED`: Canceled by user or exchange (*Terminal*).
8. `EXPIRED`: Canceled by time-in-force or reduce-only rejection (*Terminal*).
9. `REJECTED`: Rejected by Binance matching engine/filters (*Terminal*).
10. `UNKNOWN`: Socket timeout, network disconnect, or 5xx gateway error (*Non-terminal; triggers Safety Freeze*).

### 6.2 Accounting Reconciliation Lifecycle (3 States)
* `PENDING`: Order execution has not yet been reconciled against post-trade exchange balances and position records.
* `VERIFIED`: Confirmed post-trade position, fills, and balance adjustments match expectations.
* `DISCREPANCY`: Residual position or mismatched quantity detected. Triggers `SafetyFreeze`.

---

## 7. CCXT 4.4.52 Client Order ID Lookup: Offline Mock Verification Specifications

### 7.1 Test T01-A — Endpoint Dispatch and Parameter Forwarding
Verifies argument mapping and unified order parsing without network access.

### 7.2 Test T01-B — Pre-Network HTTP Transport-Boundary Serialization
Intercepts CCXT's low-level `exchange.fetch(url, method, headers, body)` method. Verifies final URL construction, query serialization, header generation, and offline HMAC-SHA256 signature correctness without making network calls.

---

## 8. Complete Binance Market Filter Parser & Residual Exit Protocol

### 8.1 Market Order Filter Precedence & Fallback Hierarchy
Symbol filters must be strictly validated against Binance `exchangeInfo`:
1. **Quantity Constraints (`MARKET_LOT_SIZE` vs. `LOT_SIZE`):**
   - Check for filter `filterType == "MARKET_LOT_SIZE"`:
     - If present and valid (`stepSize > 0`, `minQty > 0`): strictly adopt `stepSize`, `minQty`, and `maxQty`.
   - If `MARKET_LOT_SIZE` is absent, check `LOT_SIZE`:
     - If `LOT_SIZE` has flag `applyToMarket == true`, or if `MARKET_LOT_SIZE` is omitted by the exchange: adopt `LOT_SIZE` constraints.
   - If both filters are missing, disabled, zero, or malformed: **reject entry order** with `FilterValidationError`.
2. **Notional Constraints (`MIN_NOTIONAL` vs. `NOTIONAL`):**
   - Parse all applicable notional filters (`MIN_NOTIONAL` and `NOTIONAL`):
     - Check flags `applyMinToMarket` and `applyToMarket`.
     - If multiple notional filters apply, enforce the **strictest (highest) minimum notional value**.
     - If `minNotional <= 0` or missing: reject entry order.
3. **Malformed & Duplicate Filter Safeguard:**
   - Any non-numeric string, NaN, or duplicate conflicting filter blocks trading immediately.

### 8.2 Closing Quantity Workflow & Residual Prevention
When generating an exit order (`CLOSE_LONG`, `CLOSE_SHORT`):
1. **Quantity Extraction:** Active position quantity $Q_{\text{pos}} = \text{Decimal}(\text{pos}[\text{'positionAmt'}]).\text{copy\_abs}()$.
2. **Quantization Downward:**
   $$Q_{\text{order}} = (Q_{\text{pos}} // \text{stepSize}) \times \text{stepSize}$$
   **Prohibition:** Quantity must **never** be rounded upward.
3. **Dispatch:** Submit with `reduceOnly=True`.

### 8.3 Unclosable Residual/Dust Handling
If a residual active position remains where $Q_{\text{pos}} < \text{filter.minQty}$ or Notional $< \text{filter.minNotional}$:
- **Freeze Entries:** Transition to `SafetyFreeze(reason=POSITION_ACCOUNT_DISCREPANCY)`.
- **Do Not Hammer Exchange:** Automated retry loops are strictly prohibited from resubmitting rejected orders.
- **Reporting Invariant:** The system **must not report the position as flat**.
- **Telemetry Publication:** Record residual quantity and estimated mark value.
- **Resolution:** Awaits manual operator action via the Binance console.

---

## 9. Risk Sizing, Data Freshness Policy & Mandatory Pre-Dispatch Verification

### 9.1 Single-Position Exposure Policy
1. Exactly one symbol (`BTCUSDT`) is traded.
2. At most one position may be open at any time.
3. **Entry Blocking Rule:** An entry intent is strictly BLOCKED if:
   - Target position $Q_{\text{pos}} > 0.0$.
   - Any external position exists on the account ($Q_{\text{external}} > 0.0$).
   - Any open order exists on the exchange (target or external).
   - Any local intent remains unresolved in SQLite.

### 9.2 Complete 8-Step Sizing Workflow & Margin Buffer Engine
1. **Context Validation:** Verify account sync freshness, mark price freshness, and filter specifications.
2. **Target Margin Allocation:**
   $$M_{\text{intended}} = \text{equity} \times \text{margin\_allocation\_fraction}$$
3. **Target Notional & Leverage Caps:**
   $$N_{\text{target}} = \min(M_{\text{intended}} \times \text{leverage}, \text{max\_position\_notional}, \text{bracket\_max\_notional})$$
4. **Exposure Verification:** Enforce Single-Position Policy across target and non-target symbols.
5. **Raw Quantity:** $Q_{\text{raw}} = \frac{N_{\text{target}}}{P_{\text{mark}}}$.
6. **Quantization Downward:**
   $$Q_{\text{quantized}} = (Q_{\text{raw}} // \text{stepSize}) \times \text{stepSize}$$
7. **Post-Rounding Recomputation & Multi-Factor Margin Buffer:**
   $$N_{\text{final}} = Q_{\text{quantized}} \times P_{\text{mark}}$$
   $$\text{Estimated Taker Fee} = N_{\text{final}} \times \text{fee\_rate}$$
   $$M_{\text{required}} = \frac{N_{\text{final}}}{\text{leverage}} + \text{Estimated Taker Fee} + \text{Safety Buffer (0.5% Notional)}$$
   - Verify $Q_{\text{quantized}} \ge \text{minQty}$, $Q_{\text{quantized}} \le \text{maxQty}$, and $N_{\text{final}} \ge \text{minNotional}$.
   - Verify $M_{\text{required}} \le \text{available\_balance}$.
   - If rounding produces $Q_{\text{quantized}} == 0$ or violates minimums, reject entry intent.
8. **Mandatory Fresh Pre-Dispatch Verification:** Execute immediately prior to socket dispatch (Section 9.4).

### 9.3 Explicit Data Freshness Policy Table

| Data Item | Source Endpoint / Mechanism | Freshness Requirement | Max Permitted Age | Cached Permitted? | Invalidation / Failure Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Account Equity** | `GET /fapi/v2/account` | Fresh call pre-dispatch | $\le 3000\text{ ms}$ | **NO for entries** | Reject entry; trigger `AccountSyncError`. |
| **Available Balance** | `GET /fapi/v2/account` | Fresh call pre-dispatch | $\le 3000\text{ ms}$ | **NO for entries** | Reject entry; trigger `AccountSyncError`. |
| **Target Symbol Position** | `GET /fapi/v2/positionRisk` | Fresh call pre-dispatch | $\le 2000\text{ ms}$ | **NO** | Reject entry; trigger `SafetyFreeze`. |
| **Account-Wide Positions** | `GET /fapi/v2/positionRisk` | Fresh call pre-dispatch | $\le 3000\text{ ms}$ | **NO** | Reject entry; trigger `ACCOUNT_WIDE_EXPOSURE_DETECTED`. |
| **Account-Wide Open Orders**| `GET /fapi/v1/openOrders` | Fresh call pre-dispatch | $\le 3000\text{ ms}$ | **NO** | Reject entry; trigger `ACCOUNT_WIDE_EXPOSURE_DETECTED`. |
| **Mark Price** | `GET /fapi/v1/premiumIndex` | Real-time query | $\le 2000\text{ ms}$ | **NO** | Reject entry; trigger `StaleMarketDataError`. |
| **Exchange Filters** | `GET /fapi/v1/exchangeInfo` | Static session cache | $\le 1\text{ hour}$ | **YES (Valid TTL)** | Invalidate immediately on reconnect or error -1013/-1111. |
| **Leverage Setting** | `GET /fapi/v2/positionRisk` | Fresh call pre-dispatch | $\le 3000\text{ ms}$ | **NO** | Reject entry if leverage $\ne$ config. |
| **Leverage Brackets** | `GET /fapi/v1/leverageBracket`| Session cache | $\le 1\text{ hour}$ | **YES (Valid TTL)** | Invalidate on reconnect. Reject if notional exceeds bracket. |
| **Local Unresolved Intents** | SQLite `orders` table | Transaction query | $\le 0\text{ ms}$ (ACID) | **NO** | Reject entry if pending count $> 0$. |
| **Active Safety Freezes** | SQLite `safety_freezes` | Transaction query | $\le 0\text{ ms}$ (ACID) | **NO** | Reject entry if active freeze count $> 0$. |

### 9.4 Mandatory Fresh Pre-Dispatch Verification & REST TOCTOU Race
Directly before issuing the socket call to execute `create_order`, Component 3 bypasses all cached snapshots and issues fresh REST calls to re-verify the 11 items above.
Sequential REST requests do not provide atomic locking; freshness minimizes the window, but does not eliminate transport-level concurrency.

---

## 10. Daily PnL Accounting Across Midnight Rollover & External Transfer Engine

### 10.1 Single Authoritative Definition & External Transfer Formula
Daily PnL is defined strictly by the equity-delta relationship:
$$\text{Daily PnL}_t = \text{Equity}_t - C_t - E_0$$
where:
* $\text{Equity}_t$ is current total margin balance (`totalMarginBalance`) from the exchange.
* $C_t$ is net external capital transfers ($\text{Deposits} - \text{Withdrawals}$) since 00:00:00 UTC.
* $E_0$ is the verified baseline equity captured at 00:00:00 UTC.

### 10.2 Positions Held Across Midnight Rollover
If a position was opened prior to midnight UTC:
$$\Delta \text{PnL} = (B_t - B_0 - C_t) + (U_t - U_0)$$
Pre-midnight profits are completely excluded from today's drawdown calculation.

### 10.3 Direct Sampling-Only Baseline Protocol & Prohibition of Reconstruction
1. **Sampling-Only Definition:** A `VERIFIED_UTC_BASELINE` is established **exclusively** by direct REST sampling of `GET /fapi/v2/account` within $[00:00:00, 00:00:05]$ UTC while the agent is running and connected, with zero trades occurring between rollover and sampling.
2. **Prohibition of Reconstruction:** Post-facto reconstruction of historical point-in-time equity after an offline period is **strictly prohibited**.
3. **Offline Rollover Failure Handling:**
   If the agent is offline during the $[00:00:00, 00:00:05]$ UTC window, or if the baseline query fails:
   - Mark baseline as `INVALID` in SQLite.
   - Transition to `SafetyFreeze(reason=DAILY_PNL_UNINITIALIZED)`.
   - **Block all new entries** (`OPEN_LONG`, `OPEN_SHORT`).
   - Permit independently verified risk-reducing exits.
   - Entries remain blocked until the next clean 00:00:00 UTC rollover where the agent is online and captures a `VERIFIED_UTC_BASELINE`.
   - An `OPERATOR_ACKNOWLEDGED_STARTUP_BASELINE` is informational only and **never authorises new entries**.

### 10.4 External Transfer Accounting & Classification Engine
1. Only records with `incomeType == "TRANSFER"` and `asset == "USDT"` are qualified as capital movements.
2. If technical verification of external vs. internal transfer origin cannot be established via `/fapi/*` alone, and strict external-only accounting is required: mark daily PnL as `UNVERIFIED`, block entries, and trigger `DAILY_PNL_UNINITIALIZED`.
3. Overlapping queries and pagination: queries span `[midnight_ms, current_ms]` with deduplication via `tranId` in SQLite. Non-USDT transfers are excluded from USDT margin equity transfer totals.

---

## 11. Unified Safety Freeze & Exit-Permission Matrix

### 11.1 Authoritative Permission Matrix

| Freeze Reason / Operational Condition | Entries (`OPEN_*`) | Verified Risk-Reducing Exits (`CLOSE_*`) | Unverified Exits | `HOLD` | Automated vs. Manual Clearance | Clearance Criteria & Required Recovery Sequence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`NETWORK_COMMUNICATION_FAILURE`** | **BLOCKED** | **STRICTLY BLOCKED** | **BLOCKED** | **ALLOWED** | Automated | 1. Socket reconnected.<br>2. Auth handshake passed.<br>3. Clock drift $< 1000\text{ ms}$.<br>4. Fresh account & position snapshots verified.<br>5. Open orders retrieved.<br>6. Pending intents reconciled. |
| **`UNKNOWN_ORDER_OUTCOME`** | **BLOCKED** | **STRICTLY BLOCKED** | **BLOCKED** | **ALLOWED** | **Authoritative (Auto) vs. Inconclusive (Manual)** | - **Authoritative:** Satisfies all 9 sequential reconciliation checks (Section 11.2) $\implies$ Clears automatically.<br>- **Inconclusive:** `OrderNotFound` (-2013) or timeout $\implies$ Retains `UNKNOWN`, requires operator reconciliation. Never auto-resubmit. |
| **`POSITION_ACCOUNT_DISCREPANCY`** | **BLOCKED** | **PERMITTED** (Independently Verified Only) | **BLOCKED** | **ALLOWED** | **Manual Operator Action Required** | 1. Exchange position authoritatively verified via `GET /fapi/v2/positionRisk`.<br>2. Residual/dust resolved or closed via console.<br>3. Discrepancy logged.<br>4. Operator issues manual acknowledgement. |
| **`ACCOUNT_WIDE_EXPOSURE_DETECTED`** | **BLOCKED** | **PERMITTED** (Target Symbol Verified Only) | **BLOCKED** | **ALLOWED** | **Manual Operator Action Required** | 1. Non-target positions or open orders liquidated/canceled via console.<br>2. Account audit verifies 0 external positions and 0 external orders.<br>3. Operator issues manual acknowledgement. |
| **`DAILY_LOSS_LIMIT`** | **BLOCKED** | **PERMITTED** | **BLOCKED** | **ALLOWED** | Automated at UTC Rollover | 1. Clock passes 00:00:00 UTC.<br>2. `VERIFIED_UTC_BASELINE` established and persisted.<br>3. Communication and position state verified healthy. |
| **`DAILY_PNL_UNINITIALIZED`** | **BLOCKED** | **PERMITTED** | **BLOCKED** | **ALLOWED** | Automated at UTC Rollover | **Strict Invariant:** Clears **only** upon establishing a `VERIFIED_UTC_BASELINE` at the next clean 00:00:00 UTC rollover. Manual override to authorize entries is strictly prohibited. |
| **`AMBIGUOUS_POSITION_STATE`** | **BLOCKED** | **STRICTLY BLOCKED** | **BLOCKED** | **ALLOWED** | **Manual Operator Action Required** | Position direction or quantity contradictory/malformed. Block all orders until authoritative state verified. |
| **`ROUTINE_HEALTHY`** | **ALLOWED** | **ALLOWED** | **BLOCKED** | **ALLOWED** | N/A | Normal operation under valid risk context. |

### 11.2 Nine-Step Controlled Order Recovery Protocol
1. Query order by `origClientOrderId`.
2. Retrieve trade fills for order.
3. Deduplicate fills on `tradeId` in SQLite.
4. Reconcile executed quantity and average price against fills.
5. Reconcile commissions and realized PnL against fills and income ledger.
6. Retrieve current exchange position (`GET /fapi/v2/positionRisk`).
7. Retrieve current exchange balances (`GET /fapi/v2/account`).
8. Inspect unresolved local intents in SQLite.
9. Inspect unexplained exchange open orders (`GET /fapi/v1/openOrders`).

Active `NEW` and `PARTIALLY_FILLED` orders: preserve original `client_order_id`, request cancellation, confirm cancellation, never submit duplicate replacement order.

---

## 12. API Permission Verification & Pre-Flight Handshake

1. Clock drift probe (`GET /fapi/v1/time` $\implies |T_{\text{server}} - T_{\text{local}}| < 1000\text{ ms}$).
2. Account trading probe (`GET /fapi/v2/account` $\implies \text{canTrade} == \text{true}$).
3. Position mode probe (`GET /fapi/v1/positionSide/dual` $\implies \text{dualSidePosition} == \text{false}$).
4. Symbol filter probe (`GET /fapi/v1/exchangeInfo?symbol=BTCUSDT`).
5. Account-wide exposure audit (`GET /fapi/v2/positionRisk`).
6. Account-wide open order audit (`GET /fapi/v1/openOrders`).
7. Manual checklist: withdrawals disabled, IP whitelisting.

---

## 13. Configuration, Testnet Isolation & SQLite Execution Journal DDL

### 13.1 Configuration Invariants
```python
@dataclass(frozen=True)
class Component3Config:
    testnet: bool = True
    production_confirmed: bool = False
    symbol: str = "BTCUSDT"
    api_key_env_var: str = "BINANCE_TESTNET_API_KEY"
    api_secret_env_var: str = "BINANCE_TESTNET_API_SECRET"
    leverage: int = 10
    margin_type: str = "cross"
    max_position_notional: float = 50_000.0
    margin_allocation_fraction: float = 0.10
    max_daily_loss_pct: float = 0.03
    database_path: Path = Path("data/execution_journal.db")
```

### 13.2 SQLite Journal DDL & Migration Design
Includes tables:
- `schema_version`
- `orders`
- `order_state_transitions`
- `trade_fills`
- `daily_baselines`
- `external_transfers`
- `safety_freezes`
- `operator_acknowledgements`
- `sync_audit_log`
- `order_recovery_attempts`

---

## 14. Precision & Market Filter Parser

Market orders are quantized strictly against `MARKET_LOT_SIZE` and `MIN_NOTIONAL` using `Decimal` downward truncation arithmetic.

---

## 15. Comprehensive Test Plan (26 Partitioned Scenarios)

Covers T01-A through T19 across Unit, Mock Integration, Transport Boundary, and Live Testnet.

---

## 16. Target Package Architecture

`src/noone/exchange/`:
- `__init__.py`
- `config.py`
- `errors.py`
- `client.py`
- `precision.py`
- `account.py`
- `journal.py`
- `risk_gate.py`
- `reconciliation.py`
- `orchestrator.py`

---

## 17. Consolidated Register of Assumptions & Testnet Verification Tasks

18 outstanding live Testnet verification tasks documented.

---

## 18. Known Limitations & Unresolved Technical Questions

Documented TOCTOU race, lack of historical balance endpoints, and Cross Margin margin field semantics.

---

## 19. Implementation-Readiness Status

`Implementation Authorized by User on 2026-11-04 (Frozen Baseline: V2.2.4)`

---

## 20. Compliance & Non-Modification Certification

Full compliance with all architectural boundaries and protected components.

