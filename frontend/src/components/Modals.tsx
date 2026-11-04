import { useEffect, useState } from "react";
import { api, fmtPct, fmtPrice } from "../api/client";
import type { CredentialsStatus, Environment, Ticker } from "../api/types";

function Shell({
  title,
  subtitle,
  onClose,
  children,
}: {
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="panel w-[520px] max-w-full max-h-[85vh] flex flex-col p-5 bg-[#0D1321] border border-white/[0.12] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3 shrink-0 pb-2 border-b border-white/[0.06]">
          <div>
            <h3 className="text-[14px] font-bold tracking-wider text-white">
              {title}
            </h3>
            {subtitle && (
              <p className="text-[11px] text-slate-400 mt-0.5">{subtitle}</p>
            )}
          </div>
          <button
            className="w-7 h-7 rounded-md flex items-center justify-center text-slate-400 hover:text-white hover:bg-white/[0.08] transition-colors text-lg leading-none"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 1. Token Selection Modal
// ---------------------------------------------------------------------------

export function TokenSearchModal(props: {
  onClose: () => void;
  onPick: (symbol: string) => void;
}) {
  const [q, setQ] = useState("");
  const [items, setItems] = useState<{ symbol: string; base_asset: string | null }[]>([]);
  const [tickers, setTickers] = useState<Record<string, Ticker>>({});
  const [error, setError] = useState("");

  useEffect(() => {
    const t = setTimeout(() => {
      api
        .get<{ items: { symbol: string; base_asset: string | null }[] }>(
          `/api/market/universe?q=${encodeURIComponent(q)}`,
        )
        .then((r) => setItems(r.items))
        .catch((e) => setError(e instanceof Error ? e.message : "Failed to load tokens"));
    }, 150);
    return () => clearTimeout(t);
  }, [q]);

  useEffect(() => {
    items.slice(0, 15).forEach((item) => {
      if (!tickers[item.symbol]) {
        api
          .get<Ticker>(`/api/market/ticker?symbol=${item.symbol}`)
          .then((t) => setTickers((prev) => ({ ...prev, [item.symbol]: t })))
          .catch(() => undefined);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items]);

  return (
    <Shell
      title="SELECT FUTURES MARKET"
      subtitle="Binance USDⓈ-M Perpetual Futures"
      onClose={props.onClose}
    >
      <input
        className="input mb-3"
        placeholder="Type to search symbol (e.g. BTC, ETH, SOL, PEPE)…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        autoFocus
      />
      {error && <p className="text-rose-400 text-xs mb-2">{error}</p>}

      <div className="overflow-auto max-h-[360px] divide-y divide-white/[0.04]">
        {items.map((item) => {
          const t = tickers[item.symbol];
          const isUp = (t?.price_change_percent ?? 0) >= 0;
          return (
            <button
              key={item.symbol}
              className="w-full flex items-center justify-between py-2.5 px-2 text-left hover:bg-white/[0.05] rounded-md transition-colors"
              onClick={() => props.onPick(item.symbol)}
            >
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-bold text-white tracking-wide">
                  {item.symbol}
                </span>
                <span className="text-[10px] text-slate-500 uppercase">
                  {item.base_asset || "USDT"}
                </span>
              </div>

              <div className="text-right">
                <div className="num text-[12.5px] font-semibold text-white">
                  ${fmtPrice(t?.last_price)}
                </div>
                <div
                  className={`num text-[10.5px] font-bold ${
                    isUp ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {fmtPct(t?.price_change_percent)}
                </div>
              </div>
            </button>
          );
        })}

        {!items.length && !error && (
          <p className="text-slate-500 text-xs py-6 text-center">
            No matching perpetual futures markets found.
          </p>
        )}
      </div>
    </Shell>
  );
}

// ---------------------------------------------------------------------------
// 2. Connect Account Modal (Component 7 DPAPI Encrypted)
// ---------------------------------------------------------------------------

export function ConnectModal(props: {
  environment: Environment;
  credentials: CredentialsStatus;
  onClose: () => void;
  onConnect: (env: Environment, key: string, secret: string) => Promise<void>;
  onDisconnect: (env: Environment) => Promise<void>;
}) {
  const [env, setEnv] = useState<Environment>(props.environment);
  const [key, setKey] = useState("");
  const [secret, setSecret] = useState("");
  const [showSecret, setShowSecret] = useState(false);
  const [msg, setMsg] = useState<{ text: string; isError: boolean } | null>(null);
  const [busy, setBusy] = useState(false);

  const connected = props.credentials[env].connected;

  return (
    <Shell
      title="EXCHANGE CREDENTIALS"
      subtitle="Encrypted at rest with Windows DPAPI (Component 7)"
      onClose={props.onClose}
    >
      {/* Environment Selector Tabs */}
      <div className="flex items-center gap-2 mb-3 p-1 rounded-lg bg-black/40 border border-white/[0.06]">
        {(["mainnet", "testnet"] as Environment[]).map((e) => {
          const isConn = props.credentials[e].connected;
          return (
            <button
              key={e}
              className={`flex-1 py-1.5 px-3 rounded-md text-[11px] font-bold tracking-wider transition-all flex items-center justify-center gap-1.5 ${
                env === e
                  ? "bg-[#1E293B] text-white shadow-sm border border-white/[0.1]"
                  : "text-slate-400 hover:text-white"
              }`}
              onClick={() => {
                setEnv(e);
                setMsg(null);
              }}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isConn ? "bg-emerald-400" : "bg-slate-500"
                }`}
              />
              <span>{e.toUpperCase()}</span>
              <span className="text-[9.5px] font-normal text-slate-400">
                ({isConn ? "Connected" : "Disconnected"})
              </span>
            </button>
          );
        })}
      </div>

      <div className="p-3 rounded-lg bg-blue-500/[0.06] border border-blue-500/20 text-[11px] text-slate-300 leading-relaxed mb-3">
        <p>
          <strong className="text-blue-300">Security Guarantee:</strong> Your API
          key and secret are protected using Windows CryptProtectData (DPAPI) tied
          strictly to your local Windows user account. Secret keys are never
          logged, transmitted to browsers, or stored in plaintext.
        </p>
        {env === "mainnet" && (
          <p className="mt-1 text-amber-300/90 font-medium">
            Note: Mainnet credentials are used exclusively for read-only account and
            balance queries. Order dispatch is permanently disabled on Mainnet.
          </p>
        )}
      </div>

      {/* Form */}
      <div className="space-y-2.5">
        <div>
          <span className="label text-[10px] block mb-1">
            BINANCE {env.toUpperCase()} API KEY
          </span>
          <input
            className="input num"
            placeholder="Paste your API key here…"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            autoComplete="off"
            spellCheck="false"
          />
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="label text-[10px]">
              BINANCE {env.toUpperCase()} API SECRET
            </span>
            <button
              type="button"
              className="text-[10px] text-slate-400 hover:text-white"
              onClick={() => setShowSecret(!showSecret)}
            >
              {showSecret ? "Hide" : "Show"}
            </button>
          </div>
          <input
            className="input num"
            type={showSecret ? "text" : "password"}
            placeholder="Paste your API secret here…"
            value={secret}
            onChange={(e) => setSecret(e.target.value)}
            autoComplete="off"
            spellCheck="false"
          />
        </div>
      </div>

      {msg && (
        <p
          className={`text-[11.5px] mt-2.5 text-center font-medium ${
            msg.isError ? "text-rose-400" : "text-emerald-400"
          }`}
        >
          {msg.text}
        </p>
      )}

      {/* Actions */}
      <div className="flex gap-2 mt-4 pt-3 border-t border-white/[0.06]">
        <button
          className="btn btn-primary flex-1 font-bold"
          disabled={busy || !key.trim() || !secret.trim()}
          onClick={async () => {
            setBusy(true);
            setMsg(null);
            try {
              await props.onConnect(env, key.trim(), secret.trim());
              setKey("");
              setSecret("");
              setMsg({
                text: `${env.toUpperCase()} credentials encrypted and stored successfully.`,
                isError: false,
              });
            } catch (e) {
              setMsg({
                text: e instanceof Error ? e.message : "Failed to store credentials",
                isError: true,
              });
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "SAVING…" : "SAVE ENCRYPTED CREDENTIALS"}
        </button>

        <button
          className="btn btn-danger"
          disabled={busy || !connected}
          onClick={async () => {
            setBusy(true);
            try {
              await props.onDisconnect(env);
              setMsg({
                text: `${env.toUpperCase()} credentials removed.`,
                isError: false,
              });
            } catch (e) {
              setMsg({
                text: e instanceof Error ? e.message : "Failed to remove credentials",
                isError: true,
              });
            } finally {
              setBusy(false);
            }
          }}
        >
          DISCONNECT
        </button>
      </div>
    </Shell>
  );
}

// ---------------------------------------------------------------------------
// 3. Mainnet -> Testnet Switch Prompt Modal
// ---------------------------------------------------------------------------

export function MainnetSwitchPromptModal(props: {
  onClose: () => void;
  onSwitchAndStart: () => void;
}) {
  return (
    <Shell
      title="START AUTONOMOUS ENGINE"
      subtitle="Safety Restriction Enforcement"
      onClose={props.onClose}
    >
      <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-[12px] leading-relaxed mb-3">
        <p className="font-bold text-amber-400 mb-1">
          Mainnet Trading Is Strictly Read-Only
        </p>
        <p>
          PROJECT NOONE enforces a strict safety boundary: live Mainnet order
          dispatch is blocked. Autonomous RL/DQN trading execution is permitted
          only in the <strong>Binance Futures Testnet</strong> environment.
        </p>
      </div>

      <p className="text-[11.5px] text-slate-300 leading-relaxed mb-4">
        Would you like to switch to the <strong>Testnet</strong> environment and
        start the autonomous engine now?
      </p>

      <div className="flex gap-2">
        <button
          className="btn btn-success flex-1 font-bold"
          onClick={() => {
            props.onClose();
            props.onSwitchAndStart();
          }}
        >
          SWITCH TO TESTNET & START
        </button>
        <button className="btn" onClick={props.onClose}>
          CANCEL
        </button>
      </div>
    </Shell>
  );
}
