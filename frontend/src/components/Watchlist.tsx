import { fmtPct, fmtPrice } from "../api/client";
import type { Ticker } from "../api/types";
import Sparkline from "./Sparkline";

interface Props {
  symbols: string[];
  tickers: Record<string, Ticker | null>;
  selected: string;
  onSelect: (symbol: string) => void;
  onAdd: () => void;
  onRemove: (symbol: string) => void;
}

export default function Watchlist({
  symbols,
  tickers,
  selected,
  onSelect,
  onAdd,
  onRemove,
}: Props) {
  const slots: (string | null)[] = [...symbols];
  while (slots.length < 5) slots.push(null);

  return (
    <div className="flex gap-3 shrink-0 h-[100px]">
      {slots.slice(0, 5).map((sym, i) =>
        sym === null ? (
          <button
            key={`empty-${i}`}
            onClick={onAdd}
            className="flex-1 rounded-[10px] border border-dashed border-white/[0.12] bg-white/[0.02] hover:bg-white/[0.05] hover:border-blue-400/50 flex flex-col items-center justify-center text-slate-400 hover:text-white transition-all group"
          >
            <span className="text-[22px] leading-none text-slate-500 group-hover:text-blue-400 group-hover:scale-110 transition-transform">
              +
            </span>
            <span className="text-[9.5px] tracking-wider font-semibold mt-1">
              ADD TOKEN
            </span>
          </button>
        ) : (
          <div
            key={sym}
            onClick={() => onSelect(sym)}
            className={`panel card-interactive flex-1 px-3.5 py-2.5 flex flex-col justify-between cursor-pointer transition-all group ${
              selected === sym
                ? "border-cyan-500/60 bg-[#0E1626] shadow-[0_0_16px_rgba(6,182,212,0.18)] ring-1 ring-cyan-500/40"
                : "hover:border-white/20"
            }`}
          >
            {/* Top row: Symbol & 24h Change Badge */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <span className="text-[12px] font-bold text-white tracking-wide">
                  {sym}
                </span>
                {selected === sym && (
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                )}
              </div>

              <div className="flex items-center gap-1.5">
                <span
                  className={`num text-[10.5px] font-bold px-1.5 py-0.2 rounded border ${
                    (tickers[sym]?.price_change_percent ?? 0) >= 0
                      ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/25"
                      : "bg-rose-500/15 text-rose-400 border-rose-500/25"
                  }`}
                >
                  {fmtPct(tickers[sym]?.price_change_percent)}
                </span>

                <button
                  className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-rose-400 text-sm leading-none transition-opacity ml-0.5"
                  title="Remove from watchlist"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemove(sym);
                  }}
                >
                  ×
                </button>
              </div>
            </div>

            {/* Bottom row: Price & Sparkline */}
            <div className="flex items-end justify-between gap-2 mt-1">
              <div>
                <div className="num text-[15px] font-bold tracking-tight text-white">
                  ${fmtPrice(tickers[sym]?.last_price)}
                </div>
                <div className="text-[9.5px] text-slate-500 num">
                  Vol ${tickers[sym]?.quote_volume ? (tickers[sym]!.quote_volume! / 1e6).toFixed(1) + "M" : "—"}
                </div>
              </div>

              <div className="w-[84px] h-[30px] shrink-0">
                <Sparkline
                  points={tickers[sym]?.sparkline ?? []}
                  height={30}
                  showDot={false}
                />
              </div>
            </div>
          </div>
        ),
      )}
    </div>
  );
}
