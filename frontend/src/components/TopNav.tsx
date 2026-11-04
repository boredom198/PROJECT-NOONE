import type { Environment, View } from "../api/types";

interface Props {
  environment: Environment;
  view: View;
  connected: boolean;
  onDashboard: () => void;
  onTestnet: () => void;
  onTradeLogs: () => void;
  onRelay: () => void;
  onConnect: () => void;
}

export default function TopNav(p: Props) {
  const isDashboardActive = p.view === "dashboard" && p.environment === "mainnet";
  const isTestnetActive = p.environment === "testnet" && p.view === "dashboard";

  return (
    <header className="flex items-center justify-between h-[54px] px-4 border-b border-white/[0.08] bg-[#0A0E17]/95 backdrop-blur-md shrink-0">
      {/* Brand & Terminal Type */}
      <div className="flex items-center gap-2.5 min-w-[240px]">
        <div className="w-6 h-6 rounded-md bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center font-bold text-[12px] text-white shadow-sm shadow-blue-500/30">
          N
        </div>
        <span className="text-[14px] font-bold tracking-[0.16em] text-white">
          PROJECT NOONE
        </span>
        <span className="text-[9px] font-semibold tracking-wider px-1.5 py-0.5 rounded bg-white/[0.06] text-slate-400 border border-white/[0.08]">
          FOR EVERYONE
        </span>
      </div>

      {/* Navigation Pills */}
      <nav className="flex items-center p-1 bg-[#111726] rounded-lg border border-white/[0.06] shadow-inner">
        <button
          onClick={p.onDashboard}
          className={`px-3 py-1 text-[11px] font-semibold tracking-wider rounded-md transition-all ${
            isDashboardActive
              ? "bg-[#1E293B] text-white shadow-sm border border-white/[0.1]"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          DASHBOARD
        </button>

        <button
          onClick={p.onTestnet}
          className={`px-3 py-1 text-[11px] font-semibold tracking-wider rounded-md transition-all flex items-center gap-1.5 ${
            isTestnetActive
              ? "bg-amber-500/20 text-amber-300 shadow-sm border border-amber-500/30"
              : "text-slate-400 hover:text-amber-300"
          }`}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
          TESTNET
        </button>

        <button
          onClick={p.onTradeLogs}
          className={`px-3 py-1 text-[11px] font-semibold tracking-wider rounded-md transition-all ${
            p.view === "tradelogs"
              ? "bg-[#1E293B] text-white shadow-sm border border-white/[0.1]"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          TRADE LOGS
        </button>

        <button
          onClick={p.onRelay}
          className={`px-3 py-1 text-[11px] font-semibold tracking-wider rounded-md transition-all ${
            p.view === "relay"
              ? "bg-[#1E293B] text-white shadow-sm border border-white/[0.1]"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          RELAY BUFFER
        </button>
      </nav>

      {/* Right Action & Status */}
      <div className="min-w-[240px] flex items-center justify-end gap-3">
        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-black/40 border border-white/[0.06]">
          <span
            className={`w-2 h-2 rounded-full ${
              p.environment === "testnet" ? "bg-amber-400" : "bg-blue-400"
            }`}
          />
          <span className="text-[10px] font-medium tracking-wide text-slate-300 uppercase">
            {p.environment === "testnet" ? "Testnet Mode" : "Mainnet View"}
          </span>
        </div>

        <button
          onClick={p.onConnect}
          className={`btn ${
            p.connected
              ? "bg-[#121B2B] border-blue-500/40 text-blue-300 hover:bg-[#18243A]"
              : "btn-primary"
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              p.connected ? "bg-emerald-400 led-pulse-green" : "bg-slate-400"
            }`}
          />
          <span className="text-[11px] font-semibold tracking-wide">
            {p.connected ? "ACCOUNT CONNECTED" : "CONNECT ACCOUNT"}
          </span>
        </button>
      </div>
    </header>
  );
}
