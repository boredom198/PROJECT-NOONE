const INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"];

export default function ChartPanel(props: {
  html: string;
  symbol: string;
  interval: string;
  loading: boolean;
  error: string;
  onInterval: (interval: string) => void;
}) {
  return (
    <section className="panel flex-1 min-w-0 flex flex-col overflow-hidden bg-[#0B0F19] border border-white/[0.08]">
      {/* Chart Top Bar with Timeframe Chips */}
      <div className="flex items-center justify-between px-3.5 h-[42px] border-b border-white/[0.06] shrink-0 bg-[#0E1422]/90">
        <div className="flex items-center gap-2">
          <span className="text-[13.5px] font-bold text-white tracking-wide">
            {props.symbol}
          </span>
          <span className="text-[9.5px] font-semibold text-slate-400 px-1.5 py-0.5 rounded bg-white/[0.05] border border-white/[0.08]">
            BINANCE USDⓈ-M · TRADINGVIEW
          </span>
        </div>

        <div className="flex items-center gap-1 p-0.5 bg-black/40 rounded-lg border border-white/[0.06]">
          {INTERVALS.map((iv) => (
            <button
              key={iv}
              onClick={() => props.onInterval(iv)}
              className={`px-2.5 py-1 text-[11px] font-semibold rounded-md num transition-all ${
                props.interval === iv
                  ? "bg-blue-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {iv}
            </button>
          ))}
        </div>
      </div>

      {/* Chart Canvas Container */}
      <div className="relative flex-1 min-h-0 bg-[#05070B]">
        {props.loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-10 text-slate-400 text-xs gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-ping" />
            Loading TradingView Chart…
          </div>
        )}

        {props.error ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-6">
            <div className="text-rose-400 text-[14px] font-bold">Chart unavailable</div>
            <p className="text-slate-400 text-[11px] mt-1">{props.error}</p>
          </div>
        ) : (
          <iframe
            key={`${props.symbol}-${props.interval}`}
            title={`TradingView ${props.symbol}`}
            className="absolute inset-0 w-full h-full border-0"
            sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
            srcDoc={`<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0;height:100%;background:#05070b;overflow:hidden}</style></head><body>${props.html}</body></html>`}
          />
        )}
      </div>
    </section>
  );
}
