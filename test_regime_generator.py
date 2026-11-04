import math
from datetime import datetime, timedelta, timezone
import numpy as np
from noone.market_data.models import Kline

def generate_regime_candles(
    *,
    regime_type: str,
    n_candles: int,
    base_price: float = 60_000.0,
    start_time: datetime,
    seed: int = 100,
) -> list[Kline]:
    """Generates realistic 15m crypto futures candles for specific market regimes.
    
    Regimes:
    - 'bull': upward drift (~+15% over ~400 candles), lower downside volatility, occasional pullbacks.
    - 'bear': downward drift (~-15% over ~400 candles), higher volatility, dead-cat bounces.
    - 'sideways': zero net drift (~0% over ~400 candles), high volatility, cyclical swings (+-4-5%).
    """
    rng = np.random.RandomState(seed)
    candles = []
    p = base_price
    
    # Drift and volatility parameters calibrated for 15m candles
    if regime_type == "bull":
        target_total_ret = 0.15
        mu = target_total_ret / n_candles  # ~ +0.000375 per bar
        sigma = 0.0032
        reversion_strength = 0.0001
        mean_p = base_price * 1.08
    elif regime_type == "bear":
        target_total_ret = -0.15
        mu = target_total_ret / n_candles  # ~ -0.000375 per bar
        sigma = 0.0040  # bear markets typically have higher volatility
        reversion_strength = 0.0001
        mean_p = base_price * 0.92
    elif regime_type == "sideways":
        mu = 0.0
        sigma = 0.0045  # high volatility chop
        # Cyclical wave component to create realistic swings of +-4%
        cycle_period = 80  # ~20 hours per full cycle
        reversion_strength = 0.0005
        mean_p = base_price
    else:
        raise ValueError(f"Unknown regime: {regime_type}")
        
    for i in range(n_candles):
        open_time = start_time + timedelta(minutes=15 * i)
        close_time = open_time + timedelta(minutes=14, seconds=59)
        
        # Base innovation
        eps = rng.normal(0.0, sigma)
        
        # Drift and mean-reversion adjustments
        if regime_type == "bull":
            # Occasional sharp pullbacks (every ~40-60 bars)
            shock = -0.006 if (i > 10 and rng.rand() < 0.03) else 0.0
            ret = mu + eps + shock - reversion_strength * ((p - mean_p) / base_price)
        elif regime_type == "bear":
            # Occasional dead-cat bounces (every ~40-60 bars)
            shock = +0.007 if (i > 10 and rng.rand() < 0.035) else 0.0
            ret = mu + eps + shock - reversion_strength * ((p - mean_p) / base_price)
        elif regime_type == "sideways":
            # Wave component for swing trading environment
            wave_drift = 0.0006 * math.cos(2 * math.pi * i / cycle_period)
            ret = wave_drift + eps - reversion_strength * ((p - base_price) / base_price)
            
        close_p = p * math.exp(ret)
        
        # Candle wicks
        wick_high = abs(rng.normal(0.0010, 0.0006))
        wick_low = abs(rng.normal(0.0010, 0.0006))
        high_p = max(p, close_p) * (1.0 + wick_high)
        low_p = min(p, close_p) * (1.0 - wick_low)
        
        vol = float(rng.lognormal(mean=3.2, sigma=0.5))
        q_vol = vol * close_p
        
        candles.append(
            Kline(
                symbol="BTCUSDT",
                interval="15m",
                open_time=open_time,
                close_time=close_time,
                open=float(p),
                high=float(high_p),
                low=float(low_p),
                close=float(close_p),
                volume=vol,
                quote_volume=q_vol,
                trades=int(rng.randint(25, 250)),
                is_closed=True,
            )
        )
        p = close_p
        
    return candles

# Test the three regimes
t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
bull = generate_regime_candles(regime_type="bull", n_candles=400, base_price=60000.0, start_time=t0, seed=101)
bear = generate_regime_candles(regime_type="bear", n_candles=400, base_price=60000.0, start_time=t0, seed=202)
side = generate_regime_candles(regime_type="sideways", n_candles=400, base_price=60000.0, start_time=t0, seed=303)

for name, cs in [("BULL", bull), ("BEAR", bear), ("SIDEWAYS", side)]:
    p0 = cs[0].open
    p1 = cs[-1].close
    highs = [c.high for c in cs]
    lows = [c.low for c in cs]
    ret = (p1 - p0) / p0 * 100.0
    max_p = max(highs)
    min_p = min(lows)
    print(f"{name:8s}: Start={p0:.1f}, End={p1:.1f}, Return={ret:+.2f}%, Max={max_p:.1f}, Min={min_p:.1f}")

