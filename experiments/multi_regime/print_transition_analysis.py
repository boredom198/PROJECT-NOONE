"""Comprehensive Transition & Reversal Analysis Script.

Analyzes:
1. Baseline Performance across 6 transitions.
2. 36 Model Runs (Performance, Exposure, Drawdown, Trades).
3. Adaptation Dynamics: Boundary Position, Time to Close, Time to Open Opposite.
4. Step-by-step Boundary Telemetry & Q-value differentials around step 200.
5. Structural Q-value analysis: why does Q(HOLD) dominate?
"""
from __future__ import annotations

import json
from pathlib import Path

results_path = Path("experiments/multi_regime/results/transition_results.json")
telemetry_path = Path("experiments/multi_regime/results/boundary_telemetry.json")

with open(results_path, "r", encoding="utf-8") as f:
    res = json.load(f)

with open(telemetry_path, "r", encoding="utf-8") as f:
    telem = json.load(f)

baselines = res["baselines"]
models = res["models"]

print("=" * 110)
print("1. BASELINE BENCHMARKS ACROSS 6 REGIME TRANSITIONS")
print("=" * 110)
print(f"{'Sequence':<18} | {'Always HOLD':<14} | {'Buy and Hold':<18} | {'Momentum':<22}")
print(f"{'':<18} | {'Return':>7} {'MaxDD':>6} | {'Return':>8} {'MaxDD':>8} | {'Return':>8} {'MaxDD':>6} {'Trades':>6}")
print("-" * 110)

for seq, b_dict in baselines.items():
    ah = b_dict["always_hold"]
    bh = b_dict["buy_and_hold"]
    mom = b_dict["momentum"]
    print(
        f"{seq:<18} | {ah['total_return_pct']:>+6.2f}% {ah['max_drawdown_pct']:>5.2f}% | "
        f"{bh['total_return_pct']:>+7.2f}% {bh['max_drawdown_pct']:>7.2f}% | "
        f"{mom['total_return_pct']:>+7.2f}% {mom['max_drawdown_pct']:>5.2f}% {mom['completed_trades']:>6d}"
    )

print("\n" + "=" * 110)
print("2. TRAINED MODELS ACROSS 6 TRANSITIONS (36 RUNS)")
print("=" * 110)
header = (
    f"{'Model':<20} | {'Sequence':<18} | {'Tot Ret':>8} | {'MaxDD':>6} | "
    f"{'Trades':>6} | {'Fees':>6} | {'BoundPos':>8} | {'CloseT':>6} | {'OppT':>6} | {'P1 Ret':>7} | {'P2 Ret':>7}"
)
print(header)
print("-" * 110)

for m_name, seqs in models.items():
    for seq_name, r in seqs.items():
        pos_b = r["pos_at_boundary"]
        pos_str = "FLAT" if pos_b == 0 else ("LONG" if pos_b == 1 else "SHORT")
        t_c = str(r["time_to_close"]) if r["time_to_close"] is not None else "-"
        t_o = str(r["time_to_open_opposite"]) if r["time_to_open_opposite"] is not None else "-"
        print(
            f"{m_name:<20} | {seq_name:<18} | {r['total_return_pct']:>+7.2f}% | {r['max_drawdown_pct']:>5.2f}% | "
            f"{r['completed_trades']:>6d} | ${r['total_fees']:>5.2f} | {pos_str:>8} | {t_c:>6} | {t_o:>6} | "
            f"{r['p1_return_pct']:>+6.2f}% | {r['p2_return_pct']:>+6.2f}%"
        )
    print("-" * 110)

print("\n" + "=" * 110)
print("3. EXPOSURE COMPARISON: PRE-TRANSITION (P1) vs POST-TRANSITION (P2)")
print("=" * 110)
print(f"{'Model':<20} | {'Sequence':<18} | {'P1 Flat%':>8} {'P1 Long%':>8} {'P1 Short%':>9} | {'P2 Flat%':>8} {'P2 Long%':>8} {'P2 Short%':>9}")
print("-" * 110)
for m_name in ["multi_regime_seed_42", "multi_regime_seed_43", "trend_only_seed_42", "trend_only_seed_43"]:
    for seq_name, r in models[m_name].items():
        print(
            f"{m_name:<20} | {seq_name:<18} | "
            f"{r['p1_flat_pct']:>7.1f}% {r['p1_long_pct']:>7.1f}% {r['p1_short_pct']:>8.1f}% | "
            f"{r['p2_flat_pct']:>7.1f}% {r['p2_long_pct']:>7.1f}% {r['p2_short_pct']:>8.1f}%"
        )
    print("-" * 110)

print("\n" + "=" * 110)
print("4. BOUNDARY TELEMETRY: multi_regime_seed_42 on BEAR_TO_BULL (Step 195 to 225)")
print("=" * 110)
# We want to inspect the step-by-step decisions around boundary for bear_to_bull
b2b_telem = telem["multi_regime_seed_42"]["bear_to_bull"]
print(f"{'Step':>4} | {'Price':>8} | {'Pos':>5} | {'UnrlPnL':>8} | {'Act':>11} | {'RawAct':>11} | {'Q(HOLD)':>8} | {'Q(CLOSE_S)':>10} | {'Q(OPEN_L)':>9} | {'Q(CS)-Q(H)':>10}")
print("-" * 105)
for step_data in b2b_telem:
    q = step_data["q_values"]
    q_hold = q.get("HOLD", 0.0)
    q_cs = q.get("CLOSE_SHORT", 0.0)
    q_ol = q.get("OPEN_LONG", 0.0)
    diff = q_cs - q_hold
    pos_s = "FLAT" if step_data["pos_side"] == 0 else ("LONG" if step_data["pos_side"] == 1 else "SHORT")
    print(
        f"{step_data['step']:>4d} | {step_data['price']:>8.1f} | {pos_s:>5} | "
        f"{step_data['unrealized_pnl']:>+8.2f} | {step_data['action']:>11} | {step_data['raw_action']:>11} | "
        f"{q_hold:>8.4f} | {q_cs:>10.4f} | {q_ol:>9.4f} | {diff:>+10.4f}"
    )

print("\n" + "=" * 110)
print("5. BOUNDARY TELEMETRY: multi_regime_seed_42 on BULL_TO_BEAR (Step 195 to 205)")
print("=" * 110)
b_bear_telem = telem["multi_regime_seed_42"]["bull_to_bear"]
print(f"{'Step':>4} | {'Price':>8} | {'Pos':>5} | {'UnrlPnL':>8} | {'Act':>11} | {'RawAct':>11} | {'Q(HOLD)':>8} | {'Q(CLOSE_S)':>10} | {'Q(OPEN_L)':>9} | {'Q(CS)-Q(H)':>10}")
print("-" * 105)
for step_data in b_bear_telem:
    q = step_data["q_values"]
    q_hold = q.get("HOLD", 0.0)
    q_cs = q.get("CLOSE_SHORT", 0.0)
    q_ol = q.get("OPEN_LONG", 0.0)
    diff = q_cs - q_hold
    pos_s = "FLAT" if step_data["pos_side"] == 0 else ("LONG" if step_data["pos_side"] == 1 else "SHORT")
    print(
        f"{step_data['step']:>4d} | {step_data['price']:>8.1f} | {pos_s:>5} | "
        f"{step_data['unrealized_pnl']:>+8.2f} | {step_data['action']:>11} | {step_data['raw_action']:>11} | "
        f"{q_hold:>8.4f} | {q_cs:>10.4f} | {q_ol:>9.4f} | {diff:>+10.4f}"
    )

