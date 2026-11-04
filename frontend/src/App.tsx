import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api/client";
import type {
  Baseline,
  Bootstrap,
  EngineStatus,
  Environment,
  Rules,
  Summary,
  Ticker,
  TokenDetail,
  TradeLog,
  View,
} from "./api/types";
import ChartPanel from "./components/ChartPanel";
import MetricCards from "./components/MetricCards";
import {
  ConnectModal,
  MainnetSwitchPromptModal,
  TokenSearchModal,
} from "./components/Modals";
import { EnginePanel, RulesPanel, TokenDetailPanel } from "./components/Panels";
import Splash from "./components/Splash";
import TopNav from "./components/TopNav";
import Watchlist from "./components/Watchlist";
import {
  BalanceDetail,
  OrdersDetail,
  PnlDetail,
  RelayView,
  TradeLogsView,
} from "./components/Views";

const DEFAULT_CREDENTIALS: Bootstrap["credentials"] = {
  mainnet: { connected: false },
  testnet: { connected: false },
};

export default function App() {
  const [splash, setSplash] = useState(!window.location.search.includes("nosplash"));
  const [view, setView] = useState<View>("dashboard");
  const [environment, setEnvironment] = useState<Environment>("mainnet");
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setIntervalStr] = useState("15m");
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [credentials, setCredentials] = useState<Bootstrap["credentials"]>(DEFAULT_CREDENTIALS);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [detail, setDetail] = useState<TokenDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [rules, setRules] = useState<Rules | null>(null);
  const [rulesLoading, setRulesLoading] = useState(false);
  const [tickers, setTickers] = useState<Record<string, Ticker | null>>({});
  const [chartHtml, setChartHtml] = useState("");
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState("");
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [baseline, setBaseline] = useState<Baseline | null>(null);
  const [engineBusy, setEngineBusy] = useState(false);
  const [trades, setTrades] = useState<TradeLog[]>([]);
  const [tradesAvailable, setTradesAvailable] = useState(false);
  const [tradesReason, setTradesReason] = useState("");
  const [tradesLoading, setTradesLoading] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [showConnect, setShowConnect] = useState(false);
  const [showSwitchPrompt, setShowSwitchPrompt] = useState(false);
  const [banner, setBanner] = useState("");

  const envRef = useRef(environment);
  envRef.current = environment;
  const symbolRef = useRef(symbol);
  symbolRef.current = symbol;
  const watchlistRef = useRef(watchlist);
  watchlistRef.current = watchlist;

  useEffect(() => {
    const t = setTimeout(() => setSplash(false), 1400);
    return () => clearTimeout(t);
  }, []);

  const refreshSummary = useCallback(async () => {
    try {
      const s = await api.get<Summary>("/api/summary");
      if (envRef.current) setSummary(s);
    } catch {
      /* summary error state handled via backend payload */
    } finally {
      setSummaryLoading(false);
    }
  }, []);

  const refreshDetail = useCallback(async (sym: string) => {
    try {
      const d = await api.get<TokenDetail>(`/api/market/detail?symbol=${sym}`);
      setDetail(d);
    } catch {
      /* retain last valid snapshot */
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const refreshRules = useCallback(async (sym: string) => {
    try {
      const r = await api.get<Rules>(`/api/rules?symbol=${sym}`);
      setRules(r);
    } catch {
      /* retain last valid rules */
    } finally {
      setRulesLoading(false);
    }
  }, []);

  const refreshChart = useCallback(async (sym: string, iv: string) => {
    setChartLoading(true);
    setChartError("");
    try {
      const r = await api.get<{ html: string }>(`/api/chart?symbol=${sym}&interval=${iv}`);
      setChartHtml(r.html);
    } catch (e) {
      setChartError(e instanceof Error ? e.message : "Chart failed to load");
    } finally {
      setChartLoading(false);
    }
  }, []);

  const refreshTickers = useCallback(async (symbols: string[]) => {
    if (!symbols.length) return;
    await Promise.all(
      symbols.map(async (s) => {
        try {
          const t = await api.get<Ticker>(`/api/market/ticker?symbol=${s}`);
          setTickers((prev) => ({ ...prev, [s]: t }));
        } catch {
          /* keep previous */
        }
      }),
    );
  }, []);

  const refreshEngine = useCallback(async () => {
    try {
      setEngine(await api.get<EngineStatus>("/api/engine/status"));
    } catch {
      /* keep last */
    }
    try {
      setBaseline(await api.get<Baseline>("/api/baseline"));
    } catch {
      /* keep last */
    }
  }, []);

  const refreshTrades = useCallback(async () => {
    setTradesLoading(true);
    try {
      const r = await api.get<{ available: boolean; items: TradeLog[]; reason?: string }>("/api/tradelogs");
      setTradesAvailable(r.available);
      setTrades(r.items || []);
      setTradesReason(r.reason || "");
    } catch (e) {
      setTradesAvailable(false);
      setTradesReason(e instanceof Error ? e.message : "Unavailable");
    } finally {
      setTradesLoading(false);
    }
  }, []);

  // 1. Initial Bootstrap
  useEffect(() => {
    api
      .get<Bootstrap>("/api/bootstrap")
      .then((b) => {
        setEnvironment(b.environment);
        setSymbol(b.selected_symbol);
        setIntervalStr(b.interval);
        setWatchlist(b.watchlist);
        setCredentials(b.credentials);
        setEngine(b.engine);
        refreshDetail(b.selected_symbol);
        refreshRules(b.selected_symbol);
        refreshChart(b.selected_symbol, b.interval);
        refreshTickers(b.watchlist);
      })
      .catch((e) => setBanner(e instanceof Error ? e.message : "Backend unavailable"));
  }, [refreshDetail, refreshRules, refreshChart, refreshTickers]);

  // 2. On Symbol or Interval change: load immediately
  useEffect(() => {
    setDetailLoading(true);
    setRulesLoading(true);
    refreshDetail(symbol);
    refreshRules(symbol);
    refreshChart(symbol, interval);
  }, [symbol, interval, environment, refreshDetail, refreshRules, refreshChart]);

  // 3. CONTINUOUS LIVE MARKET DATA UPDATES
  // Selected token detail polls every 1500ms
  useEffect(() => {
    const t = setInterval(() => {
      refreshDetail(symbolRef.current);
    }, 1500);
    return () => clearInterval(t);
  }, [refreshDetail]);

  // Watchlist tokens poll every 2000ms
  useEffect(() => {
    if (!watchlist.length) return;
    refreshTickers(watchlist);
    const t = setInterval(() => {
      refreshTickers(watchlistRef.current);
    }, 2000);
    return () => clearInterval(t);
  }, [watchlist, refreshTickers]);

  // Account summary & Engine status poll every 4000ms
  useEffect(() => {
    refreshSummary();
    refreshEngine();
    const t = setInterval(() => {
      refreshSummary();
      refreshEngine();
    }, 4000);
    return () => clearInterval(t);
  }, [environment, symbol, refreshSummary, refreshEngine]);

  async function switchEnvironment(env: Environment) {
    if (env === environment) {
      if (env === "mainnet") setView("dashboard");
      return;
    }
    try {
      await api.post("/api/env", { environment: env });
      setEnvironment(env);
      setView("dashboard");
      setBanner("");
      refreshSummary();
      refreshRules(symbolRef.current);
      refreshEngine();
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Environment switch failed");
    }
  }

  async function selectSymbol(sym: string) {
    const next = sym.toUpperCase();
    try {
      const r = await api.post<{ watchlist: string[] }>("/api/select", { symbol: next });
      setWatchlist(r.watchlist);
    } catch {
      /* local selection still updates workspace */
    }
    setSymbol(next);
    setView("dashboard");
  }

  async function removeFromWatchlist(sym: string) {
    const next = watchlist.filter((s) => s !== sym);
    setWatchlist(next);
    try {
      await api.post("/api/watchlist", { symbols: next });
    } catch {
      /* ignore */
    }
  }

  async function startEngine() {
    setEngineBusy(true);
    try {
      setEngine(await api.post<EngineStatus>("/api/engine/start", {}));
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Engine start failed");
      refreshEngine();
    } finally {
      setEngineBusy(false);
    }
  }

  async function stopEngine() {
    setEngineBusy(true);
    try {
      setEngine(await api.post<EngineStatus>("/api/engine/stop", {}));
    } catch (e) {
      setBanner(e instanceof Error ? e.message : "Engine stop failed");
      refreshEngine();
    } finally {
      setEngineBusy(false);
    }
  }

  if (splash) return <Splash />;

  const connected = credentials[environment]?.connected ?? false;

  return (
    <div className="h-full flex flex-col bg-[#080B11] text-[#F1F5F9] overflow-hidden select-none">
      <TopNav
        environment={environment}
        view={view}
        connected={connected}
        onDashboard={() => switchEnvironment("mainnet")}
        onTestnet={() => switchEnvironment("testnet")}
        onTradeLogs={() => {
          setView("tradelogs");
          refreshTrades();
        }}
        onRelay={() => setView("relay")}
        onConnect={() => setShowConnect(true)}
      />

      {banner && (
        <div className="px-4 py-1.5 text-[11px] text-amber-300 bg-amber-950/70 border-b border-amber-800/50 flex justify-between items-center shrink-0">
          <span>⚠️ {banner}</span>
          <button
            className="hover:text-white px-1 leading-none text-sm"
            onClick={() => setBanner("")}
          >
            ×
          </button>
        </div>
      )}

      <main className="flex-1 min-h-0 flex flex-col gap-2.5 p-3">
        {/* KPI Bar */}
        <MetricCards
          summary={summary}
          loading={summaryLoading}
          onPnl={() => setView("pnl")}
          onOrders={() => setView("orders")}
          onBalance={() => setView("balance")}
        />

        {/* Dashboard View */}
        {view === "dashboard" && (
          <>
            <div className="flex gap-2.5 flex-1 min-h-0">
              {/* Left Column: Token Detail + Engine Controls */}
              <div className="w-[280px] shrink-0 flex flex-col gap-2.5 overflow-auto">
                <TokenDetailPanel detail={detail} loading={detailLoading} />
                <EnginePanel
                  environment={environment}
                  engine={engine}
                  baseline={baseline}
                  connected={connected}
                  busy={engineBusy}
                  onStart={startEngine}
                  onStop={stopEngine}
                  onPromptMainnetSwitch={() => setShowSwitchPrompt(true)}
                />
              </div>

              {/* Center Column: TradingView Advanced Chart */}
              <ChartPanel
                html={chartHtml}
                symbol={symbol}
                interval={interval}
                loading={chartLoading}
                error={chartError}
                onInterval={(iv) => {
                  setIntervalStr(iv);
                  api.post("/api/interval", { interval: iv }).catch(() => undefined);
                }}
              />

              {/* Right Column: Trading Rules & Risk Gates */}
              <div className="w-[300px] shrink-0">
                <RulesPanel
                  rules={rules}
                  loading={rulesLoading}
                  onApply={async (values) => {
                    await api.post("/api/rules/apply", values);
                    refreshRules(symbolRef.current);
                  }}
                />
              </div>
            </div>

            {/* Bottom Watchlist: 5-Token Shelf */}
            <Watchlist
              symbols={watchlist}
              tickers={tickers}
              selected={symbol}
              onSelect={selectSymbol}
              onAdd={() => setShowSearch(true)}
              onRemove={removeFromWatchlist}
            />
          </>
        )}

        {view === "tradelogs" && (
          <TradeLogsView
            available={tradesAvailable}
            reason={tradesReason}
            items={trades}
            loading={tradesLoading}
          />
        )}
        {view === "relay" && <RelayView environment={environment} />}
        {view === "pnl" && (
          <PnlDetail
            summary={summary}
            trades={trades}
            tradesAvailable={tradesAvailable}
          />
        )}
        {view === "orders" && (
          <OrdersDetail
            summary={summary}
            environment={environment}
            onCancel={async (order) => {
              await api.post("/api/orders/cancel", {
                order_id: order.id,
                client_order_id: order.client_order_id || "",
              });
              refreshSummary();
            }}
          />
        )}
        {view === "balance" && <BalanceDetail summary={summary} />}

        {view !== "dashboard" && (
          <button
            className="btn self-start mt-1 font-semibold"
            onClick={() => setView("dashboard")}
          >
            ← BACK TO DASHBOARD
          </button>
        )}
      </main>

      {/* Compliance / Safety Notice Footer */}
      <footer className="h-[28px] shrink-0 border-t border-white/[0.08] bg-[#0A0E17] flex items-center justify-between px-4 text-[10px] text-slate-400">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <span>Market: Binance USDⓈ-M Futures (public real-time) · TradingView Chart</span>
        </div>
        <span className="tracking-[0.14em] font-bold text-amber-400/90">
          ⚠️ TRADE AT YOUR OWN RISK
        </span>
        <div className="flex items-center gap-1.5 font-medium text-slate-400">
          <span>{environment === "mainnet" ? "Mainnet (Read-Only Safety Lock)" : "Testnet (Autonomous RL Execution)"}</span>
        </div>
      </footer>

      {/* Modals */}
      {showSearch && (
        <TokenSearchModal
          onClose={() => setShowSearch(false)}
          onPick={(sym) => {
            setShowSearch(false);
            selectSymbol(sym);
          }}
        />
      )}

      {showConnect && (
        <ConnectModal
          environment={environment}
          credentials={credentials}
          onClose={() => setShowConnect(false)}
          onConnect={async (env, key, secret) => {
            const r = await api.post<{ credentials: Bootstrap["credentials"] }>("/api/connect", {
              environment: env,
              api_key: key,
              api_secret: secret,
            });
            setCredentials(r.credentials);
            refreshSummary();
          }}
          onDisconnect={async (env) => {
            const r = await api.post<{ credentials: Bootstrap["credentials"] }>("/api/disconnect", {
              environment: env,
            });
            setCredentials(r.credentials);
            refreshSummary();
          }}
        />
      )}

      {showSwitchPrompt && (
        <MainnetSwitchPromptModal
          onClose={() => setShowSwitchPrompt(false)}
          onSwitchAndStart={async () => {
            await switchEnvironment("testnet");
            await startEngine();
          }}
        />
      )}
    </div>
  );
}
