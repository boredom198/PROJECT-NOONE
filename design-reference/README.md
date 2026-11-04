# PROJECT NOONE — UI Dashboard Design Reference (Component 6)

This directory contains the user interface design references, wireframes, and visual style assets for **Component 6 (UI/UX)** in PROJECT NOONE.

---

## 1. Visual Wireframe Reference

The original hand-drawn architectural layout wireframe is archived as:
* File: [`ui_dashboard_wireframe.jpg`](ui_dashboard_wireframe.jpg)

```
+--------------------------------------------------------------------------------------------------------------------+
|  PROJECT NOONE      [Dashboard]  [Testnet]  [Trade Logs]  [Relay Buffer]                 [Connect Account]         |
+--------------------------------------------------------------------------------------------------------------------+
|  +-----------------------+     +-------------------------------+     +-----------------------------------------+   |
|  | Today's PnL           |     | Active Order                  |     | Account Balance                         |   |
|  | $ xxx     [~~~/\~~~]  |     | xx   (x Long / y Short)       |     | $ xxx           [~~~~/\~~]              |   |
|  | PnL               ↗   |     | History                   ↗   |     | Portfolio                           ↗   |   |
|  +-----------------------+     +-------------------------------+     +-----------------------------------------+   |
+--------------------------------------------------------------------------------------------------------------------+
|  +---------------------+  +-----------------------------------------------------+  +-----------------------+       |
|  | Token Detail        |  |                                                     |  | Rules                 |       |
|  | • RSI, MACD         |  |                                                     |  | • Leverage            |       |
|  | • Volatility        |  |             TradingView Chart                       |  | • Max Loss Budget     |       |
|  | • Volume            |  |         (Component 5 Integration)                   |  | • Stop Loss %         |       |
|  | • Slippage          |  |                                                     |  | • Margin Mode         |       |
|  |---------------------|  |                                                     |  | • Daily Loss Limit    |       |
|  | Engine Status       |  |                                                     |  |-----------------------|       |
|  | [Active / Idle]     |  |                                                     |  |        [Apply]        |       |
|  | [ Start / Stop ]    |  |                                                     |  +-----------------------+       |
|  +---------------------+  +-----------------------------------------------------+                                  |
+--------------------------------------------------------------------------------------------------------------------+
|  +--------------------+  +--------------------+  +--------------------+  +------------------+  +-----------------+ |
|  | BTCUSDT  ↓ 7.5%    |  | ETHUSDT  ↑ 1.5%    |  | PEPEUSDT ↑ 10%     |  |                  |  |                 | |
|  | $ 78,400  [~~\__]  |  | $ 2,600   [__/~]   |  | $ 0.000459 [__/\~] |  |   + Add Token    |  |   + Add Token   | |
|  +--------------------+  +--------------------+  +--------------------+  +------------------+  +-----------------+ |
+--------------------------------------------------------------------------------------------------------------------+
|                                              Trade at Your Own Risk                                                |
+--------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Layout Breakdown & Annotations

### A. Navigation Bar (Top Header)
* **Title / Logo:** `PROJECT NOONE` ("FOR EVERYONE").
* **Mode & Screen Tabs:**
  * **`Dashboard`** *(Annotated: "Dashboard by default mainnet")*: Default view displaying live Mainnet trading status, active positions, and high-level portfolio metrics.
  * **`Testnet`** *(Annotated: "Try Testnet")*: Switches execution environment to Binance Futures Testnet (powered by Component 3).
  * **`Trade Logs`** *(Annotated: "Trade logs")*: Detailed order execution history, fills, timestamps, and fees queried from `ExecutionJournal` / `MainnetDemoJournal`.
  * **`Relay Buffer`** *(Annotated: "View mistakes / buffer logs")*: System safety logs, state reconciliation telemetry, drift monitoring, and safety freeze events.
* **Header Action Button:**
  * **`Connect Account`** *(Annotated: "View portfolio")*: Opens the modal to view portfolio details or input/manage exchange API credentials securely via **Component 7 (Windows DPAPI)**.

---

### B. Top Metric Cards (KPI Bar)
* **Card 1: `Today's PnL`**
  * Displays today's realized and unrealized dollar profit/loss (`$ xxx`).
  * Features a mini sparkline trendline showing daily PnL trajectory.
  * Drill-down link/arrow `↗` annotated with *"View from on which trade more PnL"*.
* **Card 2: `Active Order`**
  * Displays count of active positions/orders (`xx`).
  * Breakdown of directional exposure (e.g., `7 Long / 4 Short`).
  * Drill-down link/arrow `↗` annotated with *"View Active Order and history"*.
* **Card 3: `Account Balance`**
  * Displays total account equity (`$ xxx`) and margin utilization.
  * Features a sparkline showing equity growth over time.
  * Drill-down link/arrow `↗` annotated with *"View portfolio"*.

---

### C. Main Content Grid (Three Columns)

#### 1. Left Column: Market Technicals & Engine Control
* **`Token Detail` Section:**
  * Technical and execution metrics for the currently selected market:
    * `RSI` (Relative Strength Index)
    * `MACD` (Moving Average Convergence Divergence)
    * `Volatility` (ATR / Historical Volatility)
    * `Volume` (24h volume and volume delta)
    * `Slippage` (Estimated execution slippage)
* **`Engine Status` Section:**
  * Live status of the RL/DQN autonomous trading policy (e.g., *Idle*, *Evaluating*, *In Position*, *Safety Locked*).
* **Primary Toggle Button:**
  * **`[Start / Stop]`**: Arms/starts or disarms/stops the autonomous trade execution loop.

#### 2. Center Column: Main Interactive Candlestick Chart
* **`Chart` Container:**
  * Houses the pre-made **TradingView Advanced Chart widget** implemented in **Component 5** (`noone.chart.widget.generate_tradingview_embed_html`).
  * Automatically switches ticker to match the selected token (e.g., `BINANCE:BTCUSDT.P`).
  * Provides timeframe selector (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`), candlestick renderer, and technical volume series directly from TradingView's live data servers.

#### 3. Right Column: Risk & Trading Rules Configuration
* **`Rules` Panel:**
  * User-configurable risk parameters that feed directly into Component 3 / Mainnet Demo risk gates:
    * Maximum risk/loss budget ($ USDT)
    * Stop-loss percentage (%)
    * Margin allocation fraction (%)
    * Leverage setting (1x isolated for Mainnet Demo, up to 10x cross for Testnet)
    * Maximum daily loss percentage
* **Confirmation Button:**
  * **`[Apply]`**: Persists the updated risk constraints into configuration.

---

### D. Bottom Watchlist / Multi-Token Shelf
* Displays up to 5 user-selected perpetual futures markets:
  * **Tile 1:** `BTCUSDT` (e.g., Price `$ 78,400`, 24h change `↓ 7.5%`, sparkline)
  * **Tile 2:** `ETHUSDT` (e.g., Price `$ 2,600`, 24h change `↑ 1.5%`, sparkline)
  * **Tile 3:** `PEPEUSDT` (e.g., Price `$ 0.000459`, 24h change `↑ 10%`, sparkline)
  * **Tile 4 & 5:** `[+ Add Token]` buttons to append additional supported pairs.
* Clicking any tile synchronizes the entire dashboard: updates the Left Panel details, changes the Center TradingView Chart symbol, and loads token-specific rules.

---

### E. Footer
* **Compliance / Safety Notice:** `Trade at your own risk`.

---

## 3. Visual "Look & Feel" References (`ui_look_ref/`)

The high-fidelity UI visual references provided in `ui_look_ref.zip` have been extracted into [`ui_look_ref/`](ui_look_ref/):

### Reference Screenshots:
1. [`ui_look_ref/ref1_signal_ai_top_dashboard.png`](ui_look_ref/ref1_signal_ai_top_dashboard.png)
   * **Signal AI (Top Section):**
     * Dark obsidian aesthetic with warm red/orange ambient backlight glow.
     * Frosted glassmorphic card containers (`backdrop-filter: blur(12px)` with subtle border strokes).
     * Top pill-navigation bar with intuitive icon tabs.
     * KPI stat cards displaying `$ P&L`, active order count, win rates, and total balance with mini trendline sparklines and pill badges.
2. [`ui_look_ref/ref2_signal_ai_lower_sentiment.png`](ui_look_ref/ref2_signal_ai_lower_sentiment.png)
   * **Signal AI (Lower Section):**
     * Circular radial arc gauge for Market Sentiment / Greed-Fear index (Score `76`).
     * Real-time active trading signals list with live status indicators (`• RUNNING`, `BUY` / `SELL` badges, confidence %).
     * Recent trades and economic calendar timeline.
     * Floating prompt input container with quick suggestion chips.
3. [`ui_look_ref/ref3_voltrex_chart_order_panel.png`](ui_look_ref/ref3_voltrex_chart_order_panel.png)
   * **Voltrex Futures Dashboard (Upper Section):**
     * Clean, modern cryptocurrency candlestick chart layout with interval selector chips (`1m`, `5m`, `1h`, `8h`, `1D`, `1W`).
     * Right-hand side Rules / Order panel featuring:
       * `BUY` / `SELL` mode pill toggle.
       * Amount input with quick-select denomination chips (`1.00`, `2.00`, `5.00`, `10.00`, `15.00`).
       * Interactive Leverage slider (`1x` to `50x`) with a dynamic risk badge (`Low Risk`).
       * Stop Loss percentage selector.
4. [`ui_look_ref/ref4_voltrex_token_shelf_rules.png`](ui_look_ref/ref4_voltrex_token_shelf_rules.png)
   * **Voltrex Futures Dashboard (Lower Section):**
     * Bottom multi-token watchlist cards (`ETH/USD`, `SOL/USD`, `MATIC/USD`, `TUSD/USD`) featuring coin icons, current prices, 24h percentage badges, and smooth gradient bezier trendlines with endpoint dots.
     * Quick stop-loss percentage chips (`0%`, `-10%`, `-25%`, `-50%`, `-75%`) and take-profit chips (`25%`, `50%`, `100%`, `300%`).
     * Execution parameter breakdown: execution price, spread, slippage, notional value, borrow fee, and daily loss limit.

---

## 4. Synthesis: How Wireframe and Visual References Guide Component 6

* **Layout & Structure:** Strictly governed by the hand-drawn wireframe ([`ui_dashboard_wireframe.jpg`](ui_dashboard_wireframe.jpg)):
  * Header navigation $\rightarrow$ Top 3 KPI stat cards $\rightarrow$ 3-column main body (Left: Token detail & Start/Stop; Center: Component 5 TradingView chart; Right: Rules & Apply) $\rightarrow$ Bottom 5-token watchlist shelf.
* **Aesthetic & Styling:** Governed by the visual look references ([`ui_look_ref/`](ui_look_ref/)):
  * **Theme:** Deep obsidian/dark theme with subtle ambient glow and high-contrast typography.
  * **Card Design:** Glassmorphic rounded rectangles with soft semi-transparent dark backgrounds and 1px border highlights.
  * **Trendlines & Sparklines:** Smooth SVG/Canvas bezier curves embedded into the top KPI cards and bottom token cards.
  * **Controls:** Pill-shaped toggle chips for timeframe, leverage, and stop-loss percentages.

---

## 5. Mapping to NOONE Core Components

| UI Wireframe & Look Reference Section | Underlying NOONE Backend Component |
| :--- | :--- |
| `Connect Account` modal | **Component 7** (`noone.security` Windows DPAPI encrypted credential storage) |
| `Dashboard` & `Testnet` tabs | **Mainnet Demo Mode** (`noone.mainnet_demo`) & **Component 3** (`noone.exchange`) |
| `Trade Logs` & `Relay Buffer` | **Execution Journals** (`noone.exchange.journal.ExecutionJournal` & `MainnetDemoJournal`) |
| `Token Detail` (RSI, Vol, Slippage) | **Component 4** (`noone.market_data`) & **Component 1** Feature Builder (`noone.rl_engine.features`) |
| `Engine Status` & `[Start/Stop]` | **Component 1** RL Inference (`noone.rl_engine.inference.ModelInference`) |
| `Rules` Panel & `[Apply]` | **Component 3 Risk Gate** (`noone.exchange.risk_gate.RiskGate`) & `MainnetDemoConfig` |
| Central `Chart` Widget | **Component 5** (`noone.chart.widget` TradingView Advanced Chart embed) |
