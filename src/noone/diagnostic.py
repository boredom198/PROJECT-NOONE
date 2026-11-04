"""Live diagnostic: real Futures market data + real news + FinVADER.

Usage (from project root, venv active):
    python -m noone.diagnostic
    python -m noone.diagnostic --symbols BTCUSDT,ETHUSDT --limit 5
    python -m noone.diagnostic --ws-seconds 8

Proves data comes from Binance USDⓈ-M Futures (fapi/fstream) — never
Spot — and shows each live article next to its classification so a
human can judge sentiment quality. Failures print as FAIL lines.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from noone.market_data import BinanceFuturesCollector
from noone.market_data.binance_futures import REST_BASE, WS_BASE
from noone.market_data.models import age_seconds, is_stale
from noone.news_classifier import FinVADERClassifier
from noone.news_data import CryptoNewsCollector


def _age(event_time) -> str:
    age = age_seconds(event_time)
    return "unknown" if age is None else f"{age:.1f}s"


def check_market(symbols: list[str]) -> bool:
    ok = True
    print(f"== Binance USDⓈ-M Futures ==\nREST {REST_BASE} | WS {WS_BASE}")
    print("Endpoints: /fapi/v1/ping, time, exchangeInfo, ticker/24hr, "
          "premiumIndex, openInterest, klines, fundingRate")
    with BinanceFuturesCollector() as market:
        try:
            print(f"[OK] ping={market.ping()} serverTime={market.server_time_ms()}")
            info = market.exchange_info()
            perps = [s for s in info if s.contract_type == "PERPETUAL"]
            print(f"[OK] exchangeInfo: {len(info)} symbols, {len(perps)} PERPETUAL")
        except Exception as exc:
            print(f"[FAIL] connectivity: {exc}")
            return False
        for symbol in symbols:
            try:
                snap = market.snapshot(symbol)
                t, m, oi = snap["ticker_24h"], snap["mark_price"], snap["open_interest"]
                klines, funding = snap["klines"], snap["funding_history"]
                stale = is_stale(m.event_time, 30)
                print(f"[{symbol}] last={t.last_price} mark={m.mark_price} "
                      f"index={m.index_price} funding={m.last_funding_rate} "
                      f"OI={oi.open_interest} klines={len(klines)} "
                      f"fundingHist={len(funding)} markAge={_age(m.event_time)} "
                      f"stale30s={stale}")
                if t.last_price is None or m.mark_price is None or not klines:
                    print(f"[FAIL] {symbol}: missing core fields")
                    ok = False
            except Exception as exc:
                print(f"[FAIL] {symbol}: {exc}")
                ok = False
    return ok


async def _ws_probe(symbol: str, seconds: float) -> int:
    streams = [f"{symbol.lower()}@aggTrade",
               f"{symbol.lower()}@markPrice@1s",
               f"{symbol.lower()}@kline_1m"]
    count = 0
    kinds: dict[str, int] = {}
    with BinanceFuturesCollector() as market:
        try:
            async with asyncio.timeout(seconds):
                async for event in market.stream_events(streams):
                    count += 1
                    kinds[event.kind] = kinds.get(event.kind, 0) + 1
        except TimeoutError:
            pass
    if kinds:
        print(f"  WS event kinds: {kinds}")
    return count


def check_history(symbol: str) -> bool:
    from datetime import timedelta

    from noone.market_data.models import utcnow

    print("\n== Historical Futures klines (paginated) ==")
    end = utcnow()
    try:
        with BinanceFuturesCollector() as market:
            hist = market.historical_klines(
                symbol, interval="15m", start_time=end - timedelta(hours=6),
                end_time=end, total_limit=40, page_limit=10,
                only_closed=True)
    except Exception as exc:
        print(f"[FAIL] history {symbol}: {exc}")
        return False
    times = [c.open_time for c in hist.candles]
    ordered = times == sorted(times) and len(set(times)) == len(times)
    print(f"[{'OK' if ordered and hist.candles else 'FAIL'}] {symbol} 15m: "
          f"candles={len(hist.candles)} pages={hist.pages_fetched} "
          f"gaps={len(hist.gaps)} duplicatesRemoved={hist.duplicates_removed} "
          f"incomplete={len(hist.incomplete_candles)} orderedUnique={ordered}")
    for gap in hist.gaps:
        print(f"  gap: {gap.after_open_time} -> {gap.before_open_time} "
              f"missing={gap.missing_candles}")
    return bool(hist.candles) and ordered


def check_news_archive(limit: int, archive_path: str) -> bool:
    from noone.market_data.models import utcnow
    from noone.news_data import NewsArchive

    print(f"\n== News archive (SQLite) ==\npath={archive_path}")
    try:
        with CryptoNewsCollector() as news:
            articles = news.fetch_latest(limit=limit)
    except Exception as exc:
        print(f"[FAIL] news fetch: {exc}")
        return False
    classifier = FinVADERClassifier()
    classified = classifier.classify_batch(articles)
    with NewsArchive(archive_path) as arch:
        new_count = 0
        for item in classified:
            before = arch.count()
            arch.store_classification(item)
            new_count += arch.count() - before
            a, s = item.article, item.sentiment
            assets = s.assets.mentioned_assets + s.assets.affected_assets
            print(f"- [{s.label:8s} {s.compound:+.3f}] assets={list(assets)} "
                  f"uncertainAssets={list(s.assets.uncertain_assets)} "
                  f"{a.title[:70]}")
        available = arch.available_at(utcnow())
        print(f"[OK] archived total={arch.count()} newThisRun={new_count} "
              f"availableAtNow={len(available)}")
    return True


def check_news(limit: int) -> bool:
    print("\n== Free Crypto News API + Modified FinVADER ==")
    print("GET https://cryptocurrency.cv/api/news (keyless)")
    try:
        with CryptoNewsCollector() as news:
            articles = news.fetch_latest(limit=limit)
    except Exception as exc:
        print(f"[FAIL] news fetch: {exc}")
        return False
    print(f"[OK] fetched {len(articles)} real articles")
    if not articles:
        print("[FAIL] news API returned zero articles")
        return False
    classifier = FinVADERClassifier()
    print(f"classifier: {classifier.classify_text('probe').method} "
          f"degraded={classifier.degraded}")
    for item in classifier.classify_batch(articles):
        a, s = item.article, item.sentiment
        pub = a.published_at.isoformat() if a.published_at else "unknown"
        print(f"- [{s.label:8s} compound={s.compound:+.3f} proxy={s.confidence_proxy:.2f} "
              f"uncertain={s.uncertain}] {a.title[:90]}")
        print(f"  source={a.source} published={pub} url={a.url} "
              f"api_sentiment={a.api_sentiment} matched={list(s.matched_terms[:6])}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="NOONE live diagnostic")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--ws-seconds", type=float, default=0.0,
                        help="Also probe Futures WebSocket for N seconds")
    parser.add_argument("--archive", default="data/news_archive.sqlite3",
                        help="SQLite news archive path (point-in-time)")
    parser.add_argument("--skip-archive", action="store_true",
                        help="Do not write the news archive")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"NOONE live diagnostic @ {datetime.now(timezone.utc).isoformat()}")
    ok = check_market(symbols)
    if args.ws_seconds > 0 and symbols:
        try:
            count = asyncio.run(_ws_probe(symbols[0], args.ws_seconds))
        except Exception as exc:  # e.g. sandbox blocks asyncio sockets
            print(f"[FAIL] WS fstream {symbols[0]}: probe error: {exc}")
            count = 0
        print(f"[{'OK' if count else 'FAIL'}] WS fstream {symbols[0]}: "
              f"{count} messages in {args.ws_seconds:.0f}s")
        ok = ok and count > 0
    if symbols:
        ok = check_history(symbols[0]) and ok
    ok = check_news(args.limit) and ok
    if not args.skip_archive:
        ok = check_news_archive(args.limit, args.archive) and ok
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
