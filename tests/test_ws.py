"""Offline WebSocket tests (mocked frames — NOT live proof).

Live streaming proof is in tests/test_live_ws.py (NOONE_LIVE=1) and
`python -m noone.market_data.ws_diagnostic`.
"""
from noone.market_data.binance_futures import build_stream_url, parse_ws_message

AGG = {"stream": "btcusdt@aggTrade",
       "data": {"e": "aggTrade", "E": 1793647820945, "s": "BTCUSDT",
                "a": 123, "p": "77000.10", "q": "0.063",
                "T": 1793647820916, "m": True}}
MARK = {"stream": "btcusdt@markPrice@1s",
        "data": {"e": "markPriceUpdate", "E": 1793647828000, "s": "BTCUSDT",
                 "p": "77001.8", "i": "77014.7", "P": "76961.4",
                 "r": "0.00010000", "T": 1793664000000}}
KLINE = {"stream": "btcusdt@kline_1m",
         "data": {"e": "kline", "E": 1793647827878, "s": "BTCUSDT",
                  "k": {"t": 1793647800000, "T": 1793647859999,
                        "i": "1m", "o": "76986.1", "h": "77010.0",
                        "l": "76980.0", "c": "77001.0", "v": "12.5",
                        "q": "962000", "n": 42, "V": "6.0", "Q": "462000",
                        "x": False}}}


def test_url_uses_market_endpoint_and_dedupes():
    url = build_stream_url(["BTCUSDT@aggTrade", "btcusdt@aggTrade",
                            "btcusdt@kline_1m"])
    assert url.startswith("wss://fstream.binance.com/market/stream?streams=")
    # Stream-name case preserved (aggTrade), symbol lowercased, deduped.
    assert url.count("aggTrade") == 1 and "btcusdt@aggTrade" in url
    assert "kline_1m" in url
    assert "/ws/" not in url.replace("/market/stream", "")  # no legacy path


def test_parse_trade_mark_kline():
    trade = parse_ws_message(AGG)
    assert trade.kind == "trade" and trade.data.price == 77000.10
    assert trade.source == "websocket" and trade.data.source == "websocket"
    assert trade.event_time is not None and trade.received_at >= trade.event_time
    mark = parse_ws_message(MARK)
    assert mark.kind == "mark_price" and mark.data.mark_price == 77001.8
    assert mark.data.last_funding_rate == 0.0001 and mark.data.source == "websocket"
    kline = parse_ws_message(KLINE)
    assert kline.kind == "kline" and kline.data.close == 77001.0
    assert kline.data.is_closed is False and kline.data.source == "websocket"
    assert kline.data.taker_buy_volume == 6.0


def test_control_and_malformed_frames_skipped():
    assert parse_ws_message({"result": None, "id": 1}) is None  # ACK
    assert parse_ws_message({"unexpected": True}) is None
    assert parse_ws_message("not-a-dict") is None
    assert parse_ws_message({"stream": "x", "data": "junk"}) is None


def test_rest_models_default_source_rest():
    from noone.market_data.binance_futures import parse_premium_index

    rest_mark = parse_premium_index({"symbol": "BTCUSDT", "markPrice": "1"})
    assert rest_mark.source == "rest"  # REST/WS remain distinguishable
