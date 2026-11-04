import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MetricCards from "./MetricCards";
import Watchlist from "./Watchlist";
import { RelayView } from "./Views";
import type { Summary } from "../api/types";

const summary: Summary = {
  ok: true,
  environment: "mainnet",
  symbol: "BTCUSDT",
  market_source: "Binance",
  credentials: { mainnet: { connected: false }, testnet: { connected: false } },
  pnl: { available: false, value: null, reason: "unavailable" },
  balance: null,
  balance_state: "not_connected",
  orders: [],
  positions: [],
  orders_state: "not_connected",
  long_count: 0,
  short_count: 0,
};

describe("MetricCards honest states", () => {
  it("shows UNAVAILABLE pnl and NOT CONNECTED balance, never fake values", () => {
    render(<MetricCards summary={summary} loading={false} onPnl={() => {}} onOrders={() => {}} onBalance={() => {}} />);
    expect(screen.getByText("UNAVAILABLE")).toBeTruthy();
    expect(screen.getByText("NOT CONNECTED")).toBeTruthy();
    expect(screen.queryByText("$0.00")).toBeNull();
  });
});

describe("Watchlist", () => {
  it("renders populated slot with real ticker props and empty + slots", () => {
    render(
      <Watchlist
        symbols={["BTCUSDT"]}
        tickers={{
          BTCUSDT: {
            symbol: "BTCUSDT", last_price: 78583.1, price_change: 1, price_change_percent: 1.97,
            high_price: null, low_price: null, volume: null, quote_volume: null,
            sparkline: [1, 2, 3], source: "test",
          },
        }}
        selected="BTCUSDT"
        onSelect={vi.fn()}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
      />,
    );
    expect(screen.getByText("BTCUSDT")).toBeTruthy();
    expect(screen.getAllByText("ADD TOKEN").length).toBe(4);
  });
});

describe("RelayView", () => {
  it("shows professional unavailable state with no fabricated counters", () => {
    render(<RelayView environment="mainnet" />);
    expect(screen.getByText(/RELAY BUFFER — UNAVAILABLE/)).toBeTruthy();
    expect(screen.queryByText(/transactions processed/i)).toBeNull();
  });
});
