export type Environment = "mainnet" | "testnet";
export type View = "dashboard" | "tradelogs" | "relay" | "pnl" | "orders" | "balance";

export interface CredentialsStatus {
  mainnet: { connected: boolean };
  testnet: { connected: boolean };
}

export interface Bootstrap {
  ok: boolean;
  environment: Environment;
  selected_symbol: string;
  interval: string;
  watchlist: string[];
  credentials: CredentialsStatus;
  market_source: string;
  engine: EngineStatus;
}

export interface Ticker {
  symbol: string;
  last_price: number | null;
  price_change: number | null;
  price_change_percent: number | null;
  high_price: number | null;
  low_price: number | null;
  volume: number | null;
  quote_volume: number | null;
  sparkline: number[];
  source: string;
}

export interface TokenDetail {
  symbol: string;
  last_price: number | null;
  price_change_percent: number | null;
  high_price: number | null;
  low_price: number | null;
  volume_24h: number | null;
  quote_volume_24h: number | null;
  mark_price: number | null;
  index_price: number | null;
  funding_rate: number | null;
  next_funding_time: string | null;
  open_interest: number | null;
  rsi?: number | null;
  macd?: { macd: number; signal: number; hist: number } | null;
  atr?: number | null;
  volatility_pct?: number | null;
  slippage_pct?: number | null;
  source: string;
}

export interface Position {
  symbol: string;
  side: string | null;
  contracts: number;
  entry_price: number | null;
  mark_price: number | null;
  unrealized_pnl: number | null;
  leverage: unknown;
  margin_mode: unknown;
}

export interface OpenOrder {
  id: string;
  client_order_id?: string;
  symbol: string;
  side: string | null;
  type: string | null;
  price: number | null;
  amount: number | null;
  filled: number | null;
  status: string | null;
  datetime: string | null;
}

export interface Pnl {
  available: boolean;
  value: number | null;
  reason?: string;
  baseline_status?: string;
}

export interface Summary {
  ok: boolean;
  environment: Environment;
  symbol: string;
  market_source: string;
  credentials: CredentialsStatus;
  pnl: Pnl;
  balance: { equity: number; available_balance: number; margin_used: number } | null;
  balance_state: string;
  orders: OpenOrder[];
  positions: Position[];
  orders_state: string;
  long_count: number;
  short_count: number;
  error?: string;
}

export interface Rules {
  symbol: string;
  environment: Environment;
  read_only: boolean;
  filters: Record<string, string | null>;
  leverage: number;
  margin_mode: string;
  position_mode?: string;
  risk_per_trade_pct?: number;
  margin_allocation_fraction?: number;
  max_position_notional?: number;
  take_profit_pct?: number;
  stop_loss_pct?: number;
  max_daily_loss_pct?: number;
  max_daily_profit_pct?: number;
  mark_price?: number;
  note?: string;
}

export interface EngineEvent {
  ts: string;
  kind: string;
  message: string;
  action?: string;
  order_state?: string | null;
}

export interface EngineStatus {
  state: string;
  environment: string;
  symbol: string | null;
  model_available: boolean;
  model_path: string;
  last_error: string | null;
  events: EngineEvent[];
}

export interface Baseline {
  available?: boolean;
  status?: string;
  date_utc?: string;
  baseline_equity?: number | null;
  active_freezes: { reason: string; details: string | null; triggered_at: string }[];
  reason?: string;
}

export interface TradeLog {
  id: string;
  order_id: string;
  symbol: string;
  side: string | null;
  price: number | null;
  amount: number | null;
  cost: number | null;
  realized_pnl: number | null;
  commission: number | null;
  datetime: string | null;
  timestamp: number | null;
}
