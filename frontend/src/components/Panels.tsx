import { useEffect, useState, useRef } from "react";
import { fmt, fmtPct, fmtPrice } from "../api/client";
import type { Baseline, EngineStatus, Rules, TokenDetail } from "../api/types";

function StatRow({
  label,
  value,
  subvalue,
  tone,
}: {
  label: string;
  value: string;
  subvalue?: string;
  tone?: string;
}) {
  return (
    <div className="flex items-center justify-between py-[4px] border-b border-white/[0.04] last:border-0">
      <span className="text-[11px] text-slate-400">{label}</span>
      <div className="text-right">
        <span className={`num text-[11.5px] font-semibold ${tone || "text-slate-200"}`}>
          {value}
        </span>
        {subvalue && (
          <span className="text-[9.5px] text-slate-500 ml-1 num">{subvalue}</span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 1. Token Detail Panel (Technicals: RSI, MACD, Volatility, Volume, Slippage)
// ---------------------------------------------------------------------------

export function TokenDetailPanel({
  detail,
  loading,
}: {
  detail: TokenDetail | null;
  loading: boolean;
}) {
  const prevPriceRef = useRef<number | null>(null);
  const [flash, setFlash] = useState<"up" | "down" | null>(null);

  useEffect(() => {
    if (detail?.last_price !== null && detail?.last_price !== undefined) {
      if (prevPriceRef.current !== null && prevPriceRef.current !== detail.last_price) {
        setFlash(detail.last_price > prevPriceRef.current ? "up" : "down");
        const t = setTimeout(() => setFlash(null), 800);
        prevPriceRef.current = detail.last_price;
        return () => clearTimeout(t);
      }
      prevPriceRef.current = detail.last_price;
    }
  }, [detail?.last_price]);

  const rsi = detail?.rsi;
  const macd = detail?.macd;

  return (
    <section className="panel p-3 flex flex-col shrink-0">
      <div className="flex items-center justify-between mb-1.5">
        <span className="label text-[10px]">TOKEN DETAIL</span>
        <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.06] text-slate-400 font-semibold">
          USDⓈ-M PERPETUAL
        </span>
      </div>

      {loading && !detail && (
        <div className="py-6 text-center text-slate-500 text-xs">
          Loading market data…
        </div>
      )}

      {detail && (
        <div>
          {/* Symbol & Price with Live Flash */}
          <div className="flex items-baseline justify-between">
            <span className="text-[14px] font-bold text-white tracking-wide">
              {detail.symbol}
            </span>
            <span
              className={`text-[11px] font-bold num px-1.5 py-0.5 rounded ${
                (detail.price_change_percent ?? 0) >= 0
                  ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                  : "bg-rose-500/15 text-rose-400 border border-rose-500/30"
              }`}
            >
              {fmtPct(detail.price_change_percent)}
            </span>
          </div>

          <div
            className={`num text-[22px] font-bold tracking-tight text-white mt-0.5 transition-colors duration-200 ${
              flash === "up" ? "text-emerald-400 flash-up" : flash === "down" ? "text-rose-400 flash-down" : ""
            }`}
          >
            ${fmtPrice(detail.last_price)}
          </div>

          {/* Pricing & Volume Stats */}
          <div className="mt-2 divide-y divide-white/[0.04]">
            <StatRow label="Mark Price" value={`$${fmtPrice(detail.mark_price)}`} />
            <StatRow label="Index Price" value={`$${fmtPrice(detail.index_price)}`} />
            <StatRow
              label="24h High / Low"
              value={`$${fmtPrice(detail.high_price)} / $${fmtPrice(detail.low_price)}`}
            />
            <StatRow label="24h Volume" value={`${fmt(detail.volume_24h, 2)}`} />
            <StatRow label="24h Quote Vol" value={`$${fmt(detail.quote_volume_24h, 0)}`} />
          </div>

          {/* Technical Indicators */}
          <div className="mt-2 pt-2 border-t border-white/[0.06]">
            <div className="label text-[9.5px] text-slate-400 mb-1">
              TECHNICAL INDICATORS
            </div>

            {/* RSI */}
            <div className="flex items-center justify-between py-[3px]">
              <span className="text-[11px] text-slate-400">RSI (14)</span>
              <div className="flex items-center gap-1.5">
                <span
                  className={`num text-[11.5px] font-semibold ${
                    rsi === null || rsi === undefined
                      ? "text-slate-400"
                      : rsi > 70
                      ? "text-rose-400"
                      : rsi < 30
                      ? "text-emerald-400"
                      : "text-slate-200"
                  }`}
                >
                  {rsi !== null && rsi !== undefined ? rsi.toFixed(1) : "—"}
                </span>
                {rsi !== null && rsi !== undefined && (
                  <span className="text-[9px] text-slate-500 uppercase">
                    {rsi > 70 ? "Overbought" : rsi < 30 ? "Oversold" : "Neutral"}
                  </span>
                )}
              </div>
            </div>

            {/* MACD */}
            <StatRow
              label="MACD (12,26,9)"
              value={
                macd ? `${macd.macd > 0 ? "+" : ""}${macd.macd}` : "—"
              }
              subvalue={macd ? `Sig: ${macd.signal}` : undefined}
              tone={macd ? (macd.hist >= 0 ? "text-emerald-400" : "text-rose-400") : undefined}
            />

            {/* Volatility */}
            <StatRow
              label="Volatility (ATR)"
              value={detail.atr !== undefined && detail.atr !== null ? `$${fmt(detail.atr, 2)}` : "—"}
              subvalue={
                detail.volatility_pct !== undefined && detail.volatility_pct !== null
                  ? `(${detail.volatility_pct}%)`
                  : undefined
              }
            />

            {/* Slippage & Funding */}
            <StatRow
              label="Est. Slippage"
              value={
                detail.slippage_pct !== undefined && detail.slippage_pct !== null
                  ? `${detail.slippage_pct}%`
                  : "0.02%"
              }
            />
            <StatRow
              label="Funding Rate"
              value={
                detail.funding_rate !== null && detail.funding_rate !== undefined
                  ? `${(detail.funding_rate * 100).toFixed(4)}%`
                  : "—"
              }
            />
          </div>

          <p className="text-[9.5px] text-slate-500 mt-2 leading-tight">
            {detail.source}
          </p>
        </div>
      )}

      {!loading && !detail && (
        <div className="py-4 text-center text-rose-400 text-xs">
          Market data currently unavailable.
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// 2. Engine Panel (Status, Telemetry & Dedicated START/STOP Controls)
// ---------------------------------------------------------------------------

export function EnginePanel(props: {
  environment: string;
  engine: EngineStatus | null;
  baseline: Baseline | null;
  connected: boolean;
  busy: boolean;
  onStart: () => void;
  onStop: () => void;
  onPromptMainnetSwitch: () => void;
}) {
  const { engine, baseline, environment } = props;
  const state = engine?.state || "STOPPED";
  const isRunning = state === "RUNNING";
  const isStarting = state === "STARTING";
  const isStopping = state === "STOPPING";
  const isHeld = state === "HELD";
  const isError = state === "ERROR";

  const statusConfig = {
    RUNNING: {
      label: "RUNNING",
      dotClass: "bg-emerald-400 led-pulse-green",
      badgeClass: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    },
    STARTING: {
      label: "STARTING…",
      dotClass: "bg-amber-400 led-pulse-amber",
      badgeClass: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    },
    STOPPING: {
      label: "STOPPING…",
      dotClass: "bg-amber-400 led-pulse-amber",
      badgeClass: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    },
    HELD: {
      label: "SAFETY HELD",
      dotClass: "bg-amber-400 led-pulse-amber",
      badgeClass: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    },
    ERROR: {
      label: "ERROR",
      dotClass: "bg-rose-500",
      badgeClass: "bg-rose-500/15 text-rose-400 border-rose-500/30",
    },
    STOPPED: {
      label: "STOPPED",
      dotClass: "bg-slate-500",
      badgeClass: "bg-slate-500/15 text-slate-300 border-slate-500/30",
    },
  }[state] || {
    label: state,
    dotClass: "bg-slate-500",
    badgeClass: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  };

  const handleStart = () => {
    if (environment === "mainnet") {
      props.onPromptMainnetSwitch();
    } else {
      props.onStart();
    }
  };

  return (
    <section className="panel p-3 flex flex-col shrink-0">
      <div className="flex items-center justify-between mb-2">
        <span className="label text-[10px]">ENGINE CONTROLS</span>
        <div
          className={`flex items-center gap-1.5 px-2 py-0.5 rounded border text-[10px] font-bold tracking-wider ${statusConfig.badgeClass}`}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${statusConfig.dotClass}`} />
          {statusConfig.label}
        </div>
      </div>

      {/* Operational Metadata */}
      <div className="space-y-1 mt-1">
        <StatRow
          label="Execution Target"
          value={engine?.symbol ? engine.symbol : "BTCUSDT"}
          subvalue={environment === "testnet" ? "Testnet" : "Mainnet (Read-only)"}
        />
        <StatRow
          label="RL Model Policy"
          value="DQN Stage 2"
          subvalue={engine?.model_available ? "Loaded" : "Checking…"}
          tone={engine?.model_available ? "text-emerald-400" : "text-amber-400"}
        />
        <StatRow
          label="UTC Daily Baseline"
          value={baseline?.status || (environment === "testnet" ? "UNINITIALIZED" : "UNAVAILABLE")}
          subvalue={baseline?.baseline_equity ? `$${fmt(baseline.baseline_equity)}` : undefined}
          tone={baseline?.status === "VERIFIED_UTC_BASELINE" ? "text-emerald-400" : undefined}
        />
      </div>

      {/* Active Freezes & Errors */}
      {baseline?.active_freezes && baseline.active_freezes.length > 0 && (
        <div className="mt-2 p-2 rounded bg-amber-500/10 border border-amber-500/20 text-[10.5px] text-amber-300">
          <div className="font-semibold text-amber-400">SAFETY FREEZE ACTIVE</div>
          {baseline.active_freezes.map((f, i) => (
            <div key={i} className="text-slate-300 text-[10px] mt-0.5">
              {f.reason} {f.details ? `— ${f.details}` : ""}
            </div>
          ))}
        </div>
      )}

      {engine?.last_error && (
        <div className="mt-2 p-2 rounded bg-rose-500/10 border border-rose-500/20 text-[10.5px] text-rose-300 leading-tight">
          {engine.last_error}
        </div>
      )}

      {/* Primary Action Controls: START / STOP */}
      <div className="grid grid-cols-2 gap-2 mt-3">
        <button
          className="btn btn-success w-full font-bold"
          disabled={props.busy || isRunning || isStarting}
          onClick={handleStart}
        >
          <span>▶</span>
          <span>{isStarting ? "STARTING…" : "START"}</span>
        </button>

        <button
          className="btn btn-danger w-full font-bold"
          disabled={props.busy || (!isRunning && !isStarting && !isHeld)}
          onClick={props.onStop}
        >
          <span>⏹</span>
          <span>{isStopping ? "STOPPING…" : "STOP"}</span>
        </button>
      </div>

      {environment === "mainnet" && (
        <p className="text-[9.5px] text-slate-500 mt-1.5 text-center leading-relaxed">
          Mainnet is in safe monitor mode. Clicking Start will offer a switch to Testnet.
        </p>
      )}

      {environment === "testnet" && !props.connected && (
        <p className="text-[9.5px] text-amber-400/90 mt-1.5 text-center leading-relaxed">
          Provision Testnet credentials in Connect Account before starting.
        </p>
      )}

      {/* Live Event Stream */}
      {engine?.events && engine.events.length > 0 && (
        <div className="mt-2.5 pt-2 border-t border-white/[0.06]">
          <div className="label text-[9px] text-slate-500 mb-1">
            RECENT ENGINE EVENTS
          </div>
          <div className="max-h-[90px] overflow-auto space-y-1">
            {[...engine.events].reverse().slice(0, 6).map((e, i) => (
              <div key={i} className="text-[10px] flex items-baseline gap-1.5 leading-snug">
                <span className="text-slate-500 num shrink-0">
                  {e.ts ? e.ts.slice(11, 19) : ""}
                </span>
                <span
                  className={`truncate ${
                    e.kind === "held" || e.kind === "error"
                      ? "text-rose-400 font-semibold"
                      : e.kind === "prediction"
                      ? "text-blue-300"
                      : "text-slate-300"
                  }`}
                  title={e.message}
                >
                  {e.message}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// 3. Rules Panel (Right Column: Actual Backend Configuration & Risk Gates)
// ---------------------------------------------------------------------------

export function RulesPanel(props: {
  rules: Rules | null;
  loading: boolean;
  error?: string;
  onApply: (values: Record<string, number | string>) => Promise<void>;
}) {
  const r = props.rules;
  const [leverage, setLeverage] = useState("10");
  const [alloc, setAlloc] = useState("10");
  const [notional, setNotional] = useState("50000");
  const [dailyLoss, setDailyLoss] = useState("3");
  const [tp, setTp] = useState("2.0");
  const [sl, setSl] = useState("1.0");
  const [msg, setMsg] = useState<{ text: string; isError: boolean } | null>(null);
  const [applying, setApplying] = useState(false);

  useEffect(() => {
    if (r) {
      setLeverage(String(r.leverage ?? 10));
      setAlloc(String(((r.margin_allocation_fraction ?? 0.1) * 100).toFixed(0)));
      setNotional(String(r.max_position_notional ?? 50000));
      setDailyLoss(String(((r.max_daily_loss_pct ?? 0.03) * 100).toFixed(1)));
      setTp(String(r.take_profit_pct ?? 2.0));
      setSl(String(r.stop_loss_pct ?? 1.0));
    }
  }, [r]);

  const handleApply = async () => {
    setMsg(null);
    setApplying(true);
    try {
      await props.onApply({
        leverage: Number(leverage),
        margin_allocation_fraction: Number(alloc) / 100,
        max_position_notional: Number(notional),
        max_daily_loss_pct: Number(dailyLoss) / 100,
        take_profit_pct: Number(tp),
        stop_loss_pct: Number(sl),
      });
      setMsg({ text: "Saved. Takes effect at next engine start.", isError: false });
    } catch (e) {
      setMsg({
        text: e instanceof Error ? e.message : "Failed to apply rules",
        isError: true,
      });
    } finally {
      setApplying(false);
    }
  };

  return (
    <section className="panel p-3 h-full flex flex-col overflow-auto">
      <div className="flex items-center justify-between mb-2 shrink-0">
        <span className="label text-[10px]">TRADING RULES & RISK GATES</span>
        <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 font-semibold border border-blue-500/20">
          RISK CONTROLS
        </span>
      </div>

      {props.loading && !r && (
        <div className="py-6 text-center text-slate-500 text-xs">
          Loading trading rules…
        </div>
      )}

      {r && (
        <div className="flex-1 flex flex-col justify-between">
          <div className="space-y-2">
            {/* Mode & Constraint Badges */}
            <div className="flex gap-1.5">
              <span className="flex-1 text-center py-1 rounded bg-white/[0.04] border border-white/[0.08] text-[10px] font-semibold text-slate-300">
                {r.margin_mode?.toUpperCase() || "CROSS"} MARGIN
              </span>
              <span className="flex-1 text-center py-1 rounded bg-white/[0.04] border border-white/[0.08] text-[10px] font-semibold text-slate-300">
                {r.position_mode?.toUpperCase() || "ONE-WAY"}
              </span>
            </div>

            {/* Public Binance Filters */}
            <div className="py-1 border-t border-b border-white/[0.06] divide-y divide-white/[0.04]">
              <StatRow label="Active Pair" value={r.symbol} />
              <StatRow
                label="Min / Max Qty"
                value={`${r.filters?.min_qty ?? "0.001"} / ${r.filters?.max_qty ?? "1000"}`}
              />
              <StatRow label="Step Size" value={r.filters?.step_size ?? "0.001"} />
              <StatRow label="Min Notional" value={`$${r.filters?.min_notional ?? "5.0"}`} />
              <StatRow label="Tick Size" value={r.filters?.tick_size ?? "0.1"} />
            </div>

            {/* Configurable Risk Form */}
            <div className="space-y-2 pt-1">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="label text-[9.5px]">LEVERAGE</span>
                  <div className="flex gap-1">
                    {["2", "5", "10", "20"].map((lvl) => (
                      <button
                        key={lvl}
                        type="button"
                        disabled={r.read_only}
                        onClick={() => setLeverage(lvl)}
                        className={`px-1.5 py-0.5 text-[9px] font-semibold rounded ${
                          leverage === lvl
                            ? "bg-blue-600 text-white"
                            : "bg-white/[0.06] text-slate-400 hover:text-white"
                        }`}
                      >
                        {lvl}x
                      </button>
                    ))}
                  </div>
                </div>
                <input
                  className="input num"
                  disabled={r.read_only}
                  value={leverage}
                  onChange={(e) => setLeverage(e.target.value)}
                />
              </div>

              <div>
                <span className="label text-[9.5px] block mb-1">
                  MARGIN ALLOCATION (% OF EQUITY)
                </span>
                <input
                  className="input num"
                  disabled={r.read_only}
                  value={alloc}
                  onChange={(e) => setAlloc(e.target.value)}
                />
              </div>

              <div>
                <span className="label text-[9.5px] block mb-1">
                  MAX POSITION NOTIONAL (USDT)
                </span>
                <input
                  className="input num"
                  disabled={r.read_only}
                  value={notional}
                  onChange={(e) => setNotional(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <span className="label text-[9.5px] block mb-1">TAKE PROFIT (%)</span>
                  <input
                    className="input num"
                    disabled={r.read_only}
                    value={tp}
                    onChange={(e) => setTp(e.target.value)}
                  />
                </div>
                <div>
                  <span className="label text-[9.5px] block mb-1">STOP LOSS (%)</span>
                  <input
                    className="input num"
                    disabled={r.read_only}
                    value={sl}
                    onChange={(e) => setSl(e.target.value)}
                  />
                </div>
              </div>

              <div>
                <span className="label text-[9.5px] block mb-1">DAILY LOSS LIMIT (%)</span>
                <input
                  className="input num"
                  disabled={r.read_only}
                  value={dailyLoss}
                  onChange={(e) => setDailyLoss(e.target.value)}
                />
              </div>
            </div>
          </div>

          {/* Action & Feedback */}
          <div className="pt-3">
            <button
              className="btn btn-primary w-full font-bold"
              disabled={r.read_only || applying}
              onClick={handleApply}
            >
              {applying ? "SAVING…" : "APPLY RULES"}
            </button>

            {r.read_only && (
              <p className="text-[9.5px] text-slate-500 mt-2 text-center leading-relaxed">
                Mainnet is in safe observation mode. Rules reflect exchange limits.
              </p>
            )}

            {msg && (
              <p
                className={`text-[10.5px] mt-2 text-center ${
                  msg.isError ? "text-rose-400" : "text-emerald-400 font-medium"
                }`}
              >
                {msg.text}
              </p>
            )}
          </div>
        </div>
      )}

      {!props.loading && !r && (
        <div className="py-4 text-center text-rose-400 text-xs">
          {props.error || "Rules unavailable. Reconnecting exchange service…"}
        </div>
      )}
    </section>
  );
}
