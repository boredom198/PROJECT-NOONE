import { fmt, fmtPrice } from "../api/client";
import type { OpenOrder, Position, Summary, TradeLog } from "../api/types";

function Unavailable({ title, reason }: { title: string; reason: string }) {
  return (
    <div className="panel flex-1 flex flex-col items-center justify-center text-center px-6 py-16">
      <div className="text-[13px] font-semibold tracking-[0.16em] text-muted">{title}</div>
      <p className="text-[12px] text-faint mt-2 max-w-[520px] leading-relaxed">{reason}</p>
      <div className="text-[11px] text-faint mt-3 border border-line rounded-sm2 px-3 py-1.5">
        No data will be shown until a real source is available.
      </div>
    </div>
  );
}

export function RelayView({ environment }: { environment: string }) {
  return (
    <Unavailable
      title="RELAY BUFFER — UNAVAILABLE"
      reason={`No relay-buffer data source exists in the current PROJECT NOONE system for the ${environment} environment. This section intentionally shows no activity, counters or transactions until a real relay component is implemented.`}
    />
  );
}

export function TradeLogsView(props: {
  available: boolean;
  reason?: string;
  items: TradeLog[];
  loading: boolean;
}) {
  if (!props.available) {
    return (
      <Unavailable
        title="TRADE LOGS — UNAVAILABLE"
        reason={props.reason || "Trade history is unavailable via existing interfaces."}
      />
    );
  }
  return (
    <div className="panel flex-1 p-4 overflow-auto">
      <div className="label mb-2">TRADE LOGS</div>
      {props.loading && <p className="text-muted text-xs">Loading…</p>}
      {!props.items.length && !props.loading && (
        <p className="text-muted text-xs py-4">No trades recorded for the selected token.</p>
      )}
      {props.items.length > 0 && (
        <table className="w-full text-[12px]">
          <thead>
            <tr className="text-left text-muted border-b border-line">
              <th className="py-1.5 pr-2 font-medium">Time</th>
              <th className="py-1.5 pr-2 font-medium">Symbol</th>
              <th className="py-1.5 pr-2 font-medium">Side</th>
              <th className="py-1.5 pr-2 font-medium text-right">Price</th>
              <th className="py-1.5 pr-2 font-medium text-right">Amount</th>
              <th className="py-1.5 pr-2 font-medium text-right">Realized PnL</th>
              <th className="py-1.5 font-medium text-right">Fee</th>
            </tr>
          </thead>
          <tbody>
            {props.items.map((t) => (
              <tr key={t.id} className="border-b border-line/50">
                <td className="py-1.5 pr-2 num text-muted">{t.datetime || "—"}</td>
                <td className="py-1.5 pr-2">{t.symbol}</td>
                <td className={`py-1.5 pr-2 ${t.side === "buy" ? "text-gain" : "text-loss"}`}>{t.side || "—"}</td>
                <td className="py-1.5 pr-2 num text-right">${fmtPrice(t.price)}</td>
                <td className="py-1.5 pr-2 num text-right">{fmt(t.amount, 4)}</td>
                <td className={`py-1.5 pr-2 num text-right ${(t.realized_pnl ?? 0) >= 0 ? "text-gain" : "text-loss"}`}>
                  {t.realized_pnl === null ? "—" : fmt(t.realized_pnl)}
                </td>
                <td className="py-1.5 num text-right">{fmt(t.commission, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function PnlDetail(props: { summary: Summary | null; trades: TradeLog[]; tradesAvailable: boolean }) {
  const pnl = props.summary?.pnl;
  if (!pnl?.available) {
    return (
      <Unavailable
        title="TODAY'S PNL — UNAVAILABLE"
        reason={`${pnl?.reason || "Daily PnL cannot be verified."} PnL is displayed only when the UTC baseline is VERIFIED_UTC_BASELINE and external transfer synchronization has succeeded.`}
      />
    );
  }
  const wins = props.trades.filter((t) => (t.realized_pnl ?? 0) > 0);
  const losses = props.trades.filter((t) => (t.realized_pnl ?? 0) < 0);
  const sum = (rows: TradeLog[]) => rows.reduce((a, t) => a + (t.realized_pnl ?? 0), 0);
  return (
    <div className="flex gap-3 flex-1 min-h-0">
      <div className="panel flex-1 p-4">
        <div className="label">PROFITABLE TRADES TODAY</div>
        <div className="num text-[22px] font-semibold text-gain mt-1">+${fmt(sum(wins))}</div>
        <div className="text-[11px] text-muted mt-1 num">{wins.length} trades</div>
      </div>
      <div className="panel flex-1 p-4">
        <div className="label">LOSS-MAKING TRADES TODAY</div>
        <div className="num text-[22px] font-semibold text-loss mt-1">−${fmt(Math.abs(sum(losses)))}</div>
        <div className="text-[11px] text-muted mt-1 num">{losses.length} trades</div>
      </div>
      <div className="panel flex-1 p-4">
        <div className="label">TODAY'S PNL (VERIFIED)</div>
        <div className={`num text-[22px] font-semibold mt-1 ${(pnl.value ?? 0) >= 0 ? "text-gain" : "text-loss"}`}>
          {fmt(pnl.value)}
        </div>
        <div className="text-[11px] text-muted mt-1">Baseline {pnl.baseline_status}</div>
      </div>
    </div>
  );
}

export function OrdersDetail(props: {
  summary: Summary | null;
  environment: string;
  onCancel: (order: OpenOrder) => Promise<void>;
}) {
  const s = props.summary;
  if (!s || s.orders_state === "not_connected") {
    return <Unavailable title="ACTIVE ORDERS — NOT CONNECTED" reason="Connect an exchange account to view active orders and positions for this environment." />;
  }
  if (s.orders_state === "error") {
    return <Unavailable title="ACTIVE ORDERS — UNAVAILABLE" reason={s.error || "Account data could not be loaded."} />;
  }
  return (
    <div className="flex flex-col gap-3 flex-1 min-h-0 overflow-auto">
      <div className="panel p-4">
        <div className="label mb-2">OPEN POSITIONS — {s.symbol}</div>
        {!s.positions.length && <p className="text-muted text-xs">No open positions.</p>}
        {s.positions.map((p: Position, i) => (
          <div key={i} className="flex justify-between text-[12px] py-1 border-b border-line/50 last:border-0">
            <span>
              {p.symbol} · <span className={p.side === "long" ? "text-gain" : "text-loss"}>{String(p.side).toUpperCase()}</span>
            </span>
            <span className="num text-muted">
              qty {fmt(p.contracts, 4)} · entry ${fmtPrice(p.entry_price)} · uPnL{" "}
              <span className={(p.unrealized_pnl ?? 0) >= 0 ? "text-gain" : "text-loss"}>{fmt(p.unrealized_pnl)}</span>
            </span>
          </div>
        ))}
      </div>
      <div className="panel p-4">
        <div className="label mb-2">OPEN ORDERS — {s.symbol}</div>
        {!s.orders.length && <p className="text-muted text-xs">No open orders.</p>}
        {s.orders.map((o) => (
          <div key={o.id} className="flex items-center justify-between text-[12px] py-1.5 border-b border-line/50 last:border-0">
            <span className="num">
              {o.symbol} · {o.side} {o.type} · ${fmtPrice(o.price)} · qty {fmt(o.amount, 4)} · {o.status}
            </span>
            {props.environment === "testnet" ? (
              <button
                className="btn btn-danger !py-1 !px-2 text-[11px]"
                onClick={async () => {
                  if (window.confirm(`Cancel order ${o.id}?`)) await props.onCancel(o);
                }}
              >
                CLOSE ORDER
              </button>
            ) : (
              <span className="text-[10px] text-faint tracking-widest">MAINNET READ-ONLY</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function BalanceDetail({ summary }: { summary: Summary | null }) {
  const s = summary;
  if (!s?.balance) {
    return (
      <Unavailable
        title="ACCOUNT BALANCE — UNAVAILABLE"
        reason={s?.balance_state === "not_connected" ? "Connect an exchange account to view available balance and portfolio." : s?.error || "Balance could not be loaded."}
      />
    );
  }
  const b = s.balance;
  const usedPct = b.equity > 0 ? (b.margin_used / b.equity) * 100 : 0;
  return (
    <div className="flex gap-3 flex-1">
      <div className="panel flex-1 p-4">
        <div className="label">TOTAL EQUITY</div>
        <div className="num text-[26px] font-semibold mt-1">${fmt(b.equity)}</div>
      </div>
      <div className="panel flex-1 p-4">
        <div className="label">AVAILABLE BALANCE</div>
        <div className="num text-[26px] font-semibold mt-1">${fmt(b.available_balance)}</div>
      </div>
      <div className="panel flex-1 p-4">
        <div className="label">MARGIN USED</div>
        <div className="num text-[26px] font-semibold mt-1">${fmt(b.margin_used)}</div>
        <div className="text-[11px] text-muted mt-1 num">{usedPct.toFixed(2)}% of equity</div>
      </div>
    </div>
  );
}
