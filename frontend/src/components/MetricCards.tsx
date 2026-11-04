import { fmt, fmtPrice } from "../api/client";
import type { Summary } from "../api/types";
import Sparkline from "./Sparkline";

interface Props {
  summary: Summary | null;
  loading: boolean;
  onPnl: () => void;
  onOrders: () => void;
  onBalance: () => void;
}

interface CardShellProps {
  title: string;
  badge?: string;
  badgeType?: "gain" | "loss" | "neutral" | "accent";
  sparklineData?: number[];
  onOpen: () => void;
  children: React.ReactNode;
  foot: string;
}

function CardShell({
  title,
  badge,
  badgeType = "neutral",
  sparklineData,
  onOpen,
  children,
  foot,
}: CardShellProps) {
  const badgeClasses = {
    gain: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
    loss: "bg-rose-500/10 text-rose-400 border-rose-500/30",
    neutral: "bg-slate-500/10 text-slate-400 border-slate-500/30",
    accent: "bg-blue-500/10 text-blue-400 border-blue-500/30",
  }[badgeType];

  return (
    <div
      onClick={onOpen}
      className="panel card-interactive p-3.5 flex-1 min-w-0 flex flex-col justify-between group cursor-pointer"
    >
      {/* Top Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="label text-[10px] text-slate-400 tracking-wider font-semibold">
            {title}
          </span>
          {badge && (
            <span
              className={`text-[9.5px] px-1.5 py-0.5 rounded border font-semibold tracking-wide ${badgeClasses}`}
            >
              {badge}
            </span>
          )}
        </div>
        <span className="text-slate-500 group-hover:text-blue-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all text-xs">
          ↗
        </span>
      </div>

      {/* Main Stat & Sparkline */}
      <div className="flex items-center justify-between gap-3 mt-1.5 mb-1">
        <div className="min-w-0 flex-1">{children}</div>
        {sparklineData && sparklineData.length >= 2 && (
          <div className="w-[100px] h-[36px] shrink-0 opacity-80 group-hover:opacity-100 transition-opacity">
            <Sparkline points={sparklineData} height={36} />
          </div>
        )}
      </div>

      {/* Footer Navigation Link */}
      <div className="flex items-center justify-between pt-1 border-t border-white/[0.05] text-[10.5px] text-slate-500 group-hover:text-slate-300 transition-colors">
        <span>{foot}</span>
        <span className="text-[10px] text-slate-600 group-hover:text-slate-400">View</span>
      </div>
    </div>
  );
}

export default function MetricCards({
  summary,
  loading,
  onPnl,
  onOrders,
  onBalance,
}: Props) {
  const pnl = summary?.pnl;
  const isPnlAvailable =
    pnl?.available && pnl.value !== null && pnl.value !== undefined;
  const pnlVal = pnl?.value ?? 0;

  const orderCount = (summary?.orders.length ?? 0) + (summary?.positions.length ?? 0);
  const longCount = summary?.long_count ?? 0;
  const shortCount = summary?.short_count ?? 0;

  return (
    <div className="flex gap-3 shrink-0">
      {/* 1. Today's PnL */}
      <CardShell
        title="TODAY'S PNL"
        badge={
          summary?.environment === "testnet"
            ? "TESTNET"
            : "MAINNET"
        }
        badgeType={isPnlAvailable ? (pnlVal >= 0 ? "gain" : "loss") : "neutral"}
        onOpen={onPnl}
        foot="Realized & unrealized PnL breakdown"
      >
        {loading && !summary ? (
          <span className="text-slate-500 text-xs">Loading…</span>
        ) : isPnlAvailable ? (
          <div>
            <div
              className={`num text-[22px] font-bold tracking-tight ${
                pnlVal >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {pnlVal >= 0 ? "+" : "−"}${fmt(Math.abs(pnlVal))}
            </div>
            <div className="text-[10px] text-slate-400 num">
              Daily Baseline: {pnl?.baseline_status || "VERIFIED"}
            </div>
          </div>
        ) : (
          <div>
            <div className="text-[16px] font-bold tracking-wider text-slate-400">
              UNAVAILABLE
            </div>
            <div className="text-[10px] text-slate-500 truncate" title={pnl?.reason}>
              {summary?.environment === "mainnet"
                ? "Mainnet read-only interface"
                : pnl?.reason || "Baseline required"}
            </div>
          </div>
        )}
      </CardShell>

      {/* 2. Active Orders & Positions */}
      <CardShell
        title="ACTIVE ORDER"
        badge={orderCount > 0 ? `${orderCount} ACTIVE` : "IDLE"}
        badgeType={orderCount > 0 ? "accent" : "neutral"}
        onOpen={onOrders}
        foot="Active orders and position history"
      >
        <div className="flex items-baseline gap-2">
          <span className="num text-[22px] font-bold tracking-tight text-white">
            {summary ? orderCount : "—"}
          </span>
          {summary && orderCount > 0 ? (
            <div className="flex items-center gap-1.5 text-[11px] num">
              <span className="px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-400 font-semibold border border-emerald-500/25">
                {longCount} LONG
              </span>
              <span className="px-1.5 py-0.2 rounded bg-rose-500/15 text-rose-400 font-semibold border border-rose-500/25">
                {shortCount} SHORT
              </span>
            </div>
          ) : (
            <span className="text-[11px] text-slate-500">
              {summary ? "0 open exposure" : ""}
            </span>
          )}
        </div>
        <div className="text-[10px] text-slate-400 mt-0.5">
          {summary?.orders.length ?? 0} open orders · {summary?.positions.length ?? 0} positions
        </div>
      </CardShell>

      {/* 3. Account Balance */}
      <CardShell
        title="ACCOUNT BALANCE"
        badge={
          summary?.balance
            ? `$${fmt(summary.balance.available_balance)} AVAIL`
            : undefined
        }
        badgeType={summary?.balance ? "accent" : "neutral"}
        onOpen={onBalance}
        foot="Portfolio equity and margin allocation"
      >
        {summary?.balance ? (
          <div>
            <div className="num text-[22px] font-bold tracking-tight text-white">
              ${fmt(summary.balance.equity)}
            </div>
            <div className="text-[10px] text-slate-400 num">
              Margin in use: ${fmt(summary.balance.margin_used)} (
              {((summary.balance.margin_used / (summary.balance.equity || 1)) * 100).toFixed(1)}%)
            </div>
          </div>
        ) : (
          <div>
            <div className="text-[15px] font-bold tracking-wider text-slate-400">
              {summary?.balance_state === "not_connected"
                ? "NOT CONNECTED"
                : "UNAVAILABLE"}
            </div>
            <div className="text-[10px] text-slate-500">
              {summary?.balance_state === "not_connected"
                ? "Connect API credentials to view"
                : "Awaiting exchange sync"}
            </div>
          </div>
        )}
      </CardShell>
    </div>
  );
}

export function priceText(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `$${fmtPrice(v)}`;
}
