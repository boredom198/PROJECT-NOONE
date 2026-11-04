"""Deterministic Component 8 tests (synthetic candles; not live proof)."""
from datetime import datetime, timedelta, timezone

import pytest

from noone.market_data.models import Kline, KlineGap, KlineHistory
from noone.trading_env import Action, PerpetualFuturesEnv, TradingConfig

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(i, open_=100.0, high=None, low=None, close=None, closed=True):
    o = float(open_)
    c = float(close if close is not None else o)
    return Kline(symbol="BTCUSDT", interval="1m",
                 open_time=T0 + timedelta(minutes=i),
                 close_time=T0 + timedelta(minutes=i, seconds=59),
                 open=o, high=float(high if high is not None else max(o, c)),
                 low=float(low if low is not None else min(o, c)), close=c,
                 volume=10.0, quote_volume=1000.0, trades=5,
                 is_closed=closed)


def flat(n=10, price=100.0):
    return [candle(i, price) for i in range(n)]


def cfg(**kw):
    base = dict(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0)
    base.update(kw)
    return TradingConfig(**base)


def test_long_open_close_profit_and_sizing():
    candles = flat(6)
    candles[2] = candle(2, 100, close=110)
    candles[3] = candle(3, 110, close=110)
    env = PerpetualFuturesEnv(candles, cfg())
    obs = env.reset()
    assert obs["position_side"] == 0 and obs["equity"] == 10_000
    r = env.step(Action.OPEN_LONG)  # executes at candle1 open=100
    assert r.observation["position_side"] == 1
    # margin 1000 (10%), leverage 5 -> notional 5000 -> qty 50 @100
    assert r.observation["position_quantity"] == pytest.approx(50.0)
    assert r.observation["margin_used"] == pytest.approx(1000.0)
    r = env.step(Action.HOLD)  # candle2 closes 110
    assert r.observation["unrealized_pnl"] == pytest.approx(500.0)
    r = env.step(Action.CLOSE_LONG)  # executes candle3 open=110
    assert r.observation["position_side"] == 0
    assert env.realized_pnl == pytest.approx(500.0)
    assert env.balance == pytest.approx(10_500.0)


def test_short_open_close_and_hold():
    candles = flat(5)
    candles[2] = candle(2, 100, close=90)
    candles[3] = candle(3, 90, close=90)
    env = PerpetualFuturesEnv(candles, cfg())
    env.reset()
    env.step(Action.OPEN_SHORT)
    r = env.step(Action.HOLD)
    assert r.observation["unrealized_pnl"] == pytest.approx(500.0)
    env.step(Action.CLOSE_SHORT)
    assert env.realized_pnl == pytest.approx(500.0)
    r = env.step(Action.HOLD)
    assert r.info["invalid_action"] is False


def test_invalid_actions_are_noops():
    env = PerpetualFuturesEnv(flat(6), cfg())
    env.reset()
    r = env.step(Action.CLOSE_LONG)  # nothing to close
    assert r.info["invalid_action"] is True and env.position is None
    env.step(Action.OPEN_LONG)
    r = env.step(Action.OPEN_SHORT)  # already in a position
    assert r.info["invalid_action"] is True
    assert env.position is not None and env.position.side == 1
    r = env.step(Action.CLOSE_SHORT)  # wrong direction close
    assert r.info["invalid_action"] is True and env.position is not None


def test_fees_and_slippage_not_ignored():
    env = PerpetualFuturesEnv(
        flat(4), cfg(fee_rate=0.001, slippage_rate=0.001))
    env.reset()
    r = env.step(Action.OPEN_LONG)
    # entry slipped up: 100*1.001; fee = notional*0.001 charged
    assert env.position.entry_price == pytest.approx(100.1)
    assert env.total_fees == pytest.approx(5000 * 0.001, rel=1e-3)
    assert env.balance < 10_000
    with pytest.raises(ValueError):
        TradingConfig(slippage_rate=0.01, max_slippage_rate=0.001)
    with pytest.raises(ValueError):
        TradingConfig(leverage=50, max_leverage=20)
    with pytest.raises(ValueError):
        TradingConfig(capital_allocation_fraction=0.9,
                      max_capital_allocation_fraction=0.5)


def test_leverage_and_max_notional_cap():
    env = PerpetualFuturesEnv(
        flat(4), cfg(leverage=10, max_position_notional=2000))
    env.reset()
    env.step(Action.OPEN_LONG)
    assert env.position.quantity == pytest.approx(20.0)  # 2000/100
    assert env.position.margin == pytest.approx(200.0)


def timed_candle(dt, open_=100.0, high=None, low=None, close=None):
    o = float(open_)
    c = float(close if close is not None else o)
    return Kline(symbol="BTCUSDT", interval="1h", open_time=dt,
                 close_time=dt + timedelta(minutes=59),
                 open=o, high=float(high if high is not None else max(o, c)),
                 low=float(low if low is not None else min(o, c)), close=c,
                 volume=10.0, quote_volume=1000.0, trades=5, is_closed=True)


def hourly(start, n, price=100.0):
    return [timed_candle(start + timedelta(hours=i), price)
            for i in range(n)]


FUND = dict(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.01,
            funding_times_utc=("00:00", "08:00", "16:00"))


def test_funding_timestamp_schedule_starts_between_events():
    start = datetime(2026, 1, 1, 5, 0, tzinfo=timezone.utc)  # between 00/08
    env = PerpetualFuturesEnv(hourly(start, 6), TradingConfig(**FUND))
    env.reset()
    env.step(Action.OPEN_LONG)  # executes 06:00 candle, qty 50 @100
    env.step(Action.HOLD)  # 07:00
    r = env.step(Action.HOLD)  # 08:00 funding inside this candle window
    assert r.info["funding_payment"] == pytest.approx(-50.0)
    assert env.total_funding == pytest.approx(-50.0)


def test_funding_opened_after_event_not_charged_and_negative_rate():
    start = datetime(2026, 1, 1, 7, 0, tzinfo=timezone.utc)
    env = PerpetualFuturesEnv(hourly(start, 5), TradingConfig(**FUND))
    env.reset()
    env.step(Action.HOLD)  # 08:00 event passes while flat: no charge
    env.step(Action.OPEN_LONG)  # executes 09:00, after the event
    assert env.total_funding == 0.0
    # Negative rate: long receives.
    env2 = PerpetualFuturesEnv(hourly(start, 5),
                               TradingConfig(**{**FUND, "funding_rate": -0.01}))
    env2.reset()
    env2.step(Action.OPEN_LONG)
    env2.step(Action.HOLD)
    assert env2.total_funding == pytest.approx(50.0)
    # Short pays/receives oppositely under positive rate.
    env3 = PerpetualFuturesEnv(hourly(start, 5), TradingConfig(**FUND))
    env3.reset()
    env3.step(Action.OPEN_SHORT)
    env3.step(Action.HOLD)
    assert env3.total_funding == pytest.approx(50.0)


def test_funding_multiple_events_and_missing_candles():
    start = datetime(2026, 1, 1, 5, 0, tzinfo=timezone.utc)
    env = PerpetualFuturesEnv(hourly(start, 14), TradingConfig(**FUND))
    env.reset()
    env.step(Action.OPEN_LONG)
    for _ in range(11):
        env.step(Action.HOLD)
    assert env.total_funding == pytest.approx(-100.0)  # 08:00 + 16:00 once each
    # Missing 08:00/09:00 candles: event still applied exactly once.
    candles = hourly(start, 3) + hourly(start + timedelta(hours=5), 3)
    env2 = PerpetualFuturesEnv(candles, TradingConfig(**FUND))
    env2.reset()
    env2.step(Action.OPEN_LONG)
    env2.step(Action.HOLD)
    env2.step(Action.HOLD)  # jumps to 10:00 candle, window covers 08:00
    assert env2.total_funding == pytest.approx(-50.0)


def test_stop_loss_and_take_profit():
    candles = flat(6)
    candles[2] = candle(2, 100, low=97, close=99)
    env = PerpetualFuturesEnv(candles, cfg(stop_loss_pct=0.02))
    env.reset()
    env.step(Action.OPEN_LONG)
    r = env.step(Action.HOLD)
    assert r.info["event"] == "stop_loss" and env.position is None
    assert env.realized_pnl == pytest.approx(-100.0)  # 50*(98-100)

    candles = flat(6)
    candles[2] = candle(2, 100, high=103, close=101)
    env = PerpetualFuturesEnv(candles, cfg(take_profit_pct=0.02))
    env.reset()
    env.step(Action.OPEN_LONG)
    r = env.step(Action.HOLD)
    assert r.info["event"] == "take_profit"
    assert env.realized_pnl == pytest.approx(100.0)


def test_daily_loss_limit_force_closes_and_terminates():
    candles = flat(6)
    candles[2] = candle(2, 100, close=90)
    env = PerpetualFuturesEnv(candles, cfg(daily_loss_limit=100))
    env.reset()
    env.step(Action.OPEN_LONG)
    r = env.step(Action.HOLD)
    assert r.terminated and r.info["reason"] == "daily_limit"
    assert env.position is None


def test_liquidation_terminates():
    candles = flat(6)
    candles[2] = candle(2, 100, low=75, close=80)
    env = PerpetualFuturesEnv(candles, cfg(leverage=5))  # liq ~80.5
    env.reset()
    env.step(Action.OPEN_LONG)
    r = env.step(Action.HOLD)
    assert r.terminated and r.info["reason"] == "liquidation"
    assert env.position is None and env.balance < 10_000


def test_gaps_reported_never_filled_and_ordering():
    candles = [candle(2), candle(0), candle(5)]  # unsorted + gap
    hist = KlineHistory(symbol="BTCUSDT", interval="1m",
                        candles=tuple(candles),
                        gaps=(KlineGap(candles[1].open_time,
                                        candles[2].open_time, 2),),
                        duplicates_removed=0, pages_fetched=1,
                        start_time=None, end_time=None)
    env = PerpetualFuturesEnv.from_history(hist, cfg())
    assert len(env.candles) == 3  # missing candles not fabricated
    times = [c.open_time for c in env.candles]
    assert times == sorted(times)  # chronological
    env.reset()
    r = env.step(Action.HOLD)
    assert r.info["gaps_in_history"] == 1


def test_unclosed_candle_never_exposed_lookahead():
    candles = flat(4) + [candle(4, 100, close=99999, closed=False)]
    env = PerpetualFuturesEnv(candles, cfg())
    assert env.dropped_unclosed == 1 and len(env.candles) == 4
    env.reset()
    while True:
        r = env.step(Action.HOLD)
        assert r.observation["close"] != 99999
        if r.done:
            break
    assert r.truncated and r.info["reason"] == "end_of_data"


def test_reward_is_fractional_equity_change():
    candles = flat(5)
    candles[2] = candle(2, 100, close=110)
    candles[3] = candle(3, 110, close=110)
    env = PerpetualFuturesEnv(candles, cfg())
    env.reset()
    env.step(Action.OPEN_LONG)
    r = env.step(Action.HOLD)  # equity 10000 -> 10500
    assert r.reward == pytest.approx(0.05)
    r = env.step(Action.HOLD)  # flat candle, no change
    assert r.reward == pytest.approx(0.0)


def test_reset_and_termination_and_future_news_hidden(tmp_path):
    from dataclasses import replace

    from noone.news_classifier import FinVADERClassifier
    from noone.news_data import NewsArchive
    from noone.news_data.crypto_news import normalize_article

    env = PerpetualFuturesEnv(flat(4), cfg())
    env.reset()
    env.step(Action.OPEN_LONG)
    env.reset()  # full state reset
    assert env.position is None and env.balance == 10_000
    assert env.equity == 10_000

    # News published after the decision candle must not appear.
    arch_path = tmp_path / "n.db"
    future_time = T0 + timedelta(days=30)
    article = normalize_article({
        "title": "Bitcoin surges", "link": "https://example.com/f",
        "description": "BTC rally", "pubDate": future_time.isoformat(),
        "source": "CoinDesk"})
    article = replace(article, retrieved_at=future_time)
    with NewsArchive(arch_path) as arch:
        arch.store_classification(
            FinVADERClassifier().classify_article(article))
    with NewsArchive(arch_path) as arch:
        env = PerpetualFuturesEnv(flat(4), cfg(), news_archive=arch,
                                  news_asset="BTC")
        obs = env.reset()
        assert obs["news_count"] == 0.0  # future news hidden
        r = env.step(Action.HOLD)
        assert r.observation["news_count"] == 0.0


def test_b1_gap_through_fills_long_and_short():
    # Long SL 98 gapped: candle opens 90 -> fill at worse open (90).
    candles = flat(6)
    candles[2] = candle(2, 90, low=89, close=90)
    env = PerpetualFuturesEnv(candles, cfg(stop_loss_pct=0.02))
    env.reset(); env.step(Action.OPEN_LONG)
    r = env.step(Action.HOLD)
    assert r.info["event"] == "stop_loss"
    assert r.info["exit_price"] == pytest.approx(90.0)
    assert env.realized_pnl == pytest.approx(-500.0)  # 50*(90-100)
    # Short SL 102 gapped: candle opens 110 -> fill at 110.
    candles = flat(6)
    candles[2] = candle(2, 110, high=111, close=110)
    env = PerpetualFuturesEnv(candles, cfg(stop_loss_pct=0.02))
    env.reset(); env.step(Action.OPEN_SHORT)
    r = env.step(Action.HOLD)
    assert r.info["exit_price"] == pytest.approx(110.0)
    assert env.realized_pnl == pytest.approx(-500.0)
    # Normal (non-gap) crossing still fills at the level.
    candles = flat(6)
    candles[2] = candle(2, 100, low=97, close=99)
    env = PerpetualFuturesEnv(candles, cfg(stop_loss_pct=0.02))
    env.reset(); env.step(Action.OPEN_LONG)
    r = env.step(Action.HOLD)
    assert r.info["exit_price"] == pytest.approx(98.0)
    assert env.realized_pnl == pytest.approx(-100.0)


def test_b1_gap_through_liquidation_exact():
    candles = flat(6)
    candles[2] = candle(2, 75, low=70, close=72)  # opens below liq 80.5
    env = PerpetualFuturesEnv(candles, cfg(leverage=5))
    env.reset(); env.step(Action.OPEN_LONG)
    r = env.step(Action.HOLD)
    assert r.terminated and r.info["event"] == "liquidation"
    assert r.info["exit_price"] == pytest.approx(75.0)  # worse open, not 80.5
    assert env.realized_pnl == pytest.approx(-1250.0)
    assert env.balance == pytest.approx(8750.0)
    candles = flat(6)
    candles[2] = candle(2, 125, high=130, close=128)  # above short liq 119.5
    env = PerpetualFuturesEnv(candles, cfg(leverage=5))
    env.reset(); env.step(Action.OPEN_SHORT)
    r = env.step(Action.HOLD)
    assert r.info["exit_price"] == pytest.approx(125.0)
    assert env.realized_pnl == pytest.approx(-1250.0)


def test_liquidation_exact_numbers_fees_and_margin_release():
    candles = flat(6)
    candles[2] = candle(2, 100, low=75, close=80)
    env = PerpetualFuturesEnv(
        candles, cfg(leverage=5, fee_rate=0.001))
    env.reset(); env.step(Action.OPEN_LONG)  # open fee 5000*0.001=5
    r = env.step(Action.HOLD)
    assert r.info["exit_price"] == pytest.approx(80.5)  # level (no gap)
    assert env.realized_pnl == pytest.approx(-975.0)  # 50*(80.5-100)
    # balance = 10000 -5 (open fee) -975 -4.025 (close fee 50*80.5*0.001)
    assert env.balance == pytest.approx(9015.975)
    assert env.margin_used == 0.0  # margin released
    assert r.observation["available_balance"] == pytest.approx(9015.975)
    # Short mirror.
    candles = flat(6)
    candles[2] = candle(2, 100, high=125, close=120)
    env = PerpetualFuturesEnv(candles, cfg(leverage=5))
    env.reset(); env.step(Action.OPEN_SHORT)
    r = env.step(Action.HOLD)
    assert r.info["exit_price"] == pytest.approx(119.5)
    assert env.realized_pnl == pytest.approx(-975.0)
    assert env.balance == pytest.approx(9025.0)


def test_b2_malformed_candles_skipped_and_recorded():
    from dataclasses import replace as dc_replace

    for field_name in ("open", "high", "low", "close"):
        bad = dc_replace(candle(2), **{field_name: None})
        env = PerpetualFuturesEnv(flat(4) + [bad], cfg())
        assert len(env.rejected_candles) == 1
        assert field_name in env.rejected_candles[0][1]
        assert len(env.candles) == 4  # skipped, never zero-filled
    # Invalid relationship: high below close.
    bad = dc_replace(candle(2), high=90.0, close=100.0, low=80.0, open=95.0)
    env = PerpetualFuturesEnv(flat(4) + [bad], cfg())
    assert env.rejected_candles and "high" in env.rejected_candles[0][1]
    # Non-positive and non-finite.
    for bad in (dc_replace(candle(2), open=0.0),
                dc_replace(candle(2), close=float("nan"))):
        env = PerpetualFuturesEnv(flat(4) + [bad], cfg())
        assert len(env.rejected_candles) == 1
    # Audit P6 regression: missing low must not liquidate a position.
    bad = dc_replace(candle(2), low=None)
    env = PerpetualFuturesEnv(flat(4)[:2] + [bad] + flat(4)[3:], cfg(leverage=5))
    env.reset(); env.step(Action.OPEN_LONG)
    r = env.step(Action.HOLD)
    assert not r.terminated and env.position is not None


def test_b4_invalid_execution_price_rejected_safely():
    env = PerpetualFuturesEnv(flat(4), TradingConfig(
        fee_rate=0.0, slippage_rate=1.5, max_slippage_rate=2.0,
        funding_rate=0.0))
    env.reset()
    r = env.step(Action.OPEN_SHORT)  # sell exec = 100*(1-1.5) = -50
    assert r.info["invalid_action"] is True
    assert r.info["invalid_reason"] == "invalid_execution_price"
    assert env.position is None and env.balance == 10_000
    assert env.total_fees == 0.0 and env.margin_used == 0.0


def test_daily_profit_limit_and_midnight_reset():
    start = datetime(2026, 1, 1, 23, 58, tzinfo=timezone.utc)
    candles = hourly(start, 6)
    candles[1] = timed_candle(start + timedelta(hours=1), 100, close=110)
    candles[2] = timed_candle(start + timedelta(hours=2), 110, close=110)
    env = PerpetualFuturesEnv(candles, cfg(daily_profit_limit=100))
    env.reset()
    r = env.step(Action.OPEN_LONG)  # +500 unrealized breaches profit limit
    assert r.terminated and r.info["reason"] == "daily_limit"
    assert env.position is None
    assert env.realized_pnl == pytest.approx(500.0)

    # Position held across UTC midnight: day-2 PnL is measured from
    # day-2 start equity, not cumulatively from day 1.
    start = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)
    candles = hourly(start, 6)  # 22,23,00,01,02,03
    candles[1] = timed_candle(start + timedelta(hours=1), 100, close=98.8)
    candles[2] = timed_candle(start + timedelta(hours=2), 98.8, close=98.8)
    candles[3] = timed_candle(start + timedelta(hours=3), 98.8, close=97.6)
    env = PerpetualFuturesEnv(candles, cfg(daily_loss_limit=100))
    env.reset()
    env.step(Action.OPEN_LONG)  # day1: unrealized -60 (within limit)
    env.step(Action.HOLD)  # midnight candle: day2 daily PnL ~0
    r = env.step(Action.HOLD)  # day2: -60 more (total -120, daily -60)
    assert not r.terminated  # reset at midnight prevented termination
    assert r.info["daily_pnl"] == pytest.approx(-60.0)
    assert env.position is not None


def test_news_published_past_first_seen_future_hidden(tmp_path):
    from dataclasses import replace

    from noone.news_classifier import FinVADERClassifier
    from noone.news_data import NewsArchive
    from noone.news_data.crypto_news import normalize_article

    article = normalize_article({
        "title": "Bitcoin surges", "link": "https://example.com/past",
        "description": "BTC rally", "pubDate": T0.isoformat(),
        "source": "CoinDesk"})
    # Published at T0 but NOONE only observes it 30 days later.
    article = replace(article, retrieved_at=T0 + timedelta(days=30))
    path = tmp_path / "seen.db"
    with NewsArchive(path) as arch:
        arch.store_classification(
            FinVADERClassifier().classify_article(article))
    with NewsArchive(path) as arch:
        env = PerpetualFuturesEnv(flat(4), cfg(), news_archive=arch,
                                  news_asset="BTC")
        obs = env.reset()
        assert obs["news_count"] == 0.0  # first_seen_at is in the future


def test_available_after_close_and_reward_with_fees_funding():
    env = PerpetualFuturesEnv(flat(5), cfg(fee_rate=0.001))
    env.reset()
    r = env.step(Action.OPEN_LONG)
    assert r.reward == pytest.approx(-5.0 / 10_000)  # open fee only (flat)
    env.step(Action.CLOSE_LONG)
    assert env.margin_used == 0.0
    assert env.available_balance == pytest.approx(env.balance)
    assert env.balance == pytest.approx(10_000 - 10.0)  # open+close fees
    # Forced-close reward: daily profit limit closes a +500 position;
    # the step reward must equal the realized equity change fraction.
    candles = flat(5)
    candles[2] = candle(2, 100, close=110)
    env = PerpetualFuturesEnv(candles, cfg(daily_profit_limit=100))
    env.reset(); env.step(Action.OPEN_LONG)
    r = env.step(Action.HOLD)
    assert r.terminated and env.position is None
    assert r.reward == pytest.approx(500.0 / 10_000)
