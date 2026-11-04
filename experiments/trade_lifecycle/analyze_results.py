"""Comprehensive Analysis of Trade Lifecycle Evaluation Results.

PROJECT NOONE - Component 1 Final Investigation
Processes experiments/trade_lifecycle/results/lifecycle_evaluation_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

res_path = Path("experiments/trade_lifecycle/results/lifecycle_evaluation_results.json")
with open(res_path, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data["summary"]
details = data["detailed_results"]

print("=" * 110)
print("1. AGGREGATE CHECKPOINT COMPARISON TABLE (15 CHECKPOINTS x 6 ENVIRONMENTS)")
print("=" * 110)
header = (
    f"{'Checkpoint':<24} | {'Group':<18} | {'Seed':>4} | {'Runs':>4} | "
    f"{'PermFlat':>8} | {'HeldEnd':>7} | {'Trades':>6} | {'PolicyClose':>11} | "
    f"{'LongOpen':>8} | {'ShortOpen':>9} | {'AvgRet':>7} | {'AvgMaxDD':>8}"
)
print(header)
print("-" * 110)

# Group by checkpoint
checkpoints = sorted(list(details.keys()))
ckpt_stats = []

for ckpt in checkpoints:
    runs = [r for r in summary if r["checkpoint"] == ckpt]
    n_runs = len(runs)
    perm_flat = sum(1 for r in runs if r["permanently_flat"])
    held_end = sum(1 for r in runs if r["held_until_end"])
    tot_trades = sum(r["trades_opened"] for r in runs)
    tot_policy_closed = sum(r["policy_closed"] for r in runs)
    avg_ret = sum(r["total_return_pct"] for r in runs) / n_runs if n_runs else 0.0
    avg_dd = sum(r["max_drawdown_pct"] for r in runs) / n_runs if n_runs else 0.0

    # Check Long / Short capability across all trades of this checkpoint
    has_long = False
    has_short = False
    for env_name, env_res in details[ckpt].items():
        for t in env_res["trades"]:
            if t["direction"] == "LONG":
                has_long = True
            elif t["direction"] == "SHORT":
                has_short = True

    grp = runs[0]["group"]
    seed = runs[0]["seed"]
    ckpt_stats.append({
        "checkpoint": ckpt,
        "group": grp,
        "seed": seed,
        "runs": n_runs,
        "perm_flat": perm_flat,
        "held_end": held_end,
        "tot_trades": tot_trades,
        "tot_policy_closed": tot_policy_closed,
        "has_long": has_long,
        "has_short": has_short,
        "avg_ret": avg_ret,
        "avg_dd": avg_dd,
    })

    long_str = "YES" if has_long else "NO"
    short_str = "YES" if has_short else "NO"
    print(
        f"{ckpt:<24} | {grp:<18} | {seed:>4d} | {n_runs:>4d} | "
        f"{perm_flat:>8d} | {held_end:>7d} | {tot_trades:>6d} | {tot_policy_closed:>11d} | "
        f"{long_str:>8} | {short_str:>9} | {avg_ret:>+6.2f}% | {avg_dd:>7.2f}%"
    )

print("-" * 110)

print("\n" + "=" * 110)
print("2. COMPLETE TRADE LIFECYCLES: SAMPLES WITH ENTRY/EXIT Q-VALUES & OBSERVATIONS")
print("=" * 110)

# Find examples of complete trade lifecycles (where exit_type == 'policy_close' and returned_to_flat == True)
lifecycle_samples = []
for ckpt in checkpoints:
    for env_name, env_res in details[ckpt].items():
        for t in env_res["trades"]:
            if t["exit_type"] == "policy_close" and t["returned_to_flat"]:
                lifecycle_samples.append((ckpt, env_name, t))

print(f"Total complete policy-driven trade lifecycles identified: {len(lifecycle_samples)}")

# Select 3 representative examples:
# 1. multi_regime_seed_42 on bear_to_bull (Capital Preservation Short exit)
# 2. trend_only_seed_42 on strong_bull (Long entry and quick exit)
# 3. Exp_A_10k_seed_43 on high_vol_sideways (Active trade)
sample_targets = [
    ("multi_regime_seed_42", "bear_to_bull"),
    ("trend_only_seed_42", "strong_bull"),
    ("Exp_A_10k_seed_43", "high_vol_sideways"),
]

for target_ckpt, target_env in sample_targets:
    target_trades = [
        t for (c, e, t) in lifecycle_samples if c == target_ckpt and e == target_env
    ]
    if not target_trades:
        continue
    t = target_trades[0]
    print(f"\n--- Trade Lifecycle Sample: {target_ckpt} in {target_env} (Trade #{t['trade_id']}) ---")
    print(f"  Direction:          {t['direction']}")
    print(f"  Entry Step:         {t['entry_step']} (Time: {t['entry_time']})")
    print(f"  Entry Exec Price:   ${t['entry_price']:.2f} (Size: {t['position_size']:.4f}, Margin: ${t['margin_used']:.2f})")
    print(f"  Entry Action:       {t['entry_action']} (Raw: {t['entry_raw_action']}, Valid: {t['entry_is_valid']})")
    print(f"  Entry Q-values:     HOLD={t['entry_q_values']['HOLD']:+.4f}, OL={t['entry_q_values']['OPEN_LONG']:+.4f}, CL={t['entry_q_values']['CLOSE_LONG']:+.4f}, OS={t['entry_q_values']['OPEN_SHORT']:+.4f}, CS={t['entry_q_values']['CLOSE_SHORT']:+.4f}")
    print(f"  Entry Key Features: Return15={t['entry_observations']['log_return_15']:+.4f}, EMA9_dev={t['entry_observations']['close_ema9_dev']:+.4f}, RSI14={t['entry_observations']['rsi_14']:.1f}, ADX14={t['entry_observations']['adx_14']:.1f}")
    print(f"  Position Duration:  {t['duration_steps']} steps")
    print(f"  Exit Step:          {t['exit_step']} (Time: {t['exit_time']})")
    print(f"  Exit Exec Price:    ${t['exit_price']:.2f}")
    print(f"  Exit Action:        {t['exit_action']} (Type: {t['exit_type']}, Raw: {t['exit_raw_action']})")
    print(f"  Exit Q-values:      HOLD={t['exit_q_values']['HOLD']:+.4f}, OL={t['exit_q_values']['OPEN_LONG']:+.4f}, CL={t['exit_q_values']['CLOSE_LONG']:+.4f}, OS={t['exit_q_values']['OPEN_SHORT']:+.4f}, CS={t['exit_q_values']['CLOSE_SHORT']:+.4f}")
    print(f"  Exit Key Features:  Return15={t['exit_observations']['log_return_15']:+.4f}, EMA9_dev={t['exit_observations']['close_ema9_dev']:+.4f}, RSI14={t['exit_observations']['rsi_14']:.1f}, ADX14={t['exit_observations']['adx_14']:.1f}")
    print(f"  Accounting:         RealizedPnL=${t['realized_pnl']:+.2f}, EntryFee=${t['entry_fee']:.2f}, ExitFee=${t['exit_fee']:.2f}, Funding=${t['funding_accrued']:+.2f}, NetPnL=${t['net_pnl']:+.2f}")
    print(f"  Returned to Flat:   {t['returned_to_flat']} (Steps flat after exit: {t['steps_flat_after_exit']})")

print("\n" + "=" * 110)
print("3. CHECKPOINT CATEGORIZATION BY TRADING BEHAVIOUR")
print("=" * 110)

cat_a = [s["checkpoint"] for s in ckpt_stats if s["perm_flat"] == s["runs"]]
cat_b = [s["checkpoint"] for s in ckpt_stats if s["tot_policy_closed"] == 0 and s["held_end"] > 0]
cat_c = [s["checkpoint"] for s in ckpt_stats if s["tot_policy_closed"] > 0]

print(f"Category A: Permanently Flat (Zero Trades across all environments, 100% HOLD):")
for c in cat_a:
    print(f"  - {c}")

print(f"\nCategory B: Permanent Position (Opens on Bar 0, Never Closes, Holds to End of Episode):")
for c in cat_b:
    print(f"  - {c}")

print(f"\nCategory C: Active Lifecycle Checkpoints (Generates Entries AND Policy Exits):")
for c in cat_c:
    st = next(s for s in ckpt_stats if s["checkpoint"] == c)
    print(f"  - {c:<24} | Trades Opened: {st['tot_trades']:>2d} | Policy Closed: {st['tot_policy_closed']:>2d} | Long: {st['has_long']} | Short: {st['has_short']}")

