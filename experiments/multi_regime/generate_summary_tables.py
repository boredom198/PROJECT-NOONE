import json
from pathlib import Path

res_path = Path("experiments/multi_regime/results/transition_results.json")
with open(res_path, "r", encoding="utf-8") as f:
    res = json.load(f)

models = res["models"]
baselines = res["baselines"]

print("=========================================================================================")
print("ALL 6 MODELS ACROSS ALL 6 TRANSITIONS (36 COMBINATIONS)")
print("=========================================================================================")
for m_name in models:
    print(f"\nMODEL: {m_name}")
    print(f"  {'Sequence':<18} | {'Total':>7} | {'P1 Ret':>7} | {'P2 Ret':>7} | {'MaxDD':>6} | {'BoundPos':>8} | {'CloseT':>6} | {'OppT':>6} | {'P1(F/L/S)':>10} | {'P2(F/L/S)':>10} | {'Trades':>6} | {'Invalid':>7}")
    print("  " + "-" * 115)
    for s_name, data in models[m_name].items():
        pos_b = data["pos_at_boundary"]
        pos_str = "FLAT" if pos_b == 0 else ("LONG" if pos_b == 1 else "SHORT")
        c_t = str(data["time_to_close"]) if data["time_to_close"] is not None else "-"
        o_t = str(data["time_to_open_opposite"]) if data["time_to_open_opposite"] is not None else "-"
        p1_fls = f"{data['p1_flat_pct']:2.0f}/{data['p1_long_pct']:2.0f}/{data['p1_short_pct']:2.0f}%"
        p2_fls = f"{data['p2_flat_pct']:2.0f}/{data['p2_long_pct']:2.0f}/{data['p2_short_pct']:2.0f}%"
        print(
            f"  {s_name:<18} | {data['total_return_pct']:>+6.2f}% | {data['p1_return_pct']:>+6.2f}% | {data['p2_return_pct']:>+6.2f}% | "
            f"{data['max_drawdown_pct']:>5.2f}% | {pos_str:>8} | {c_t:>6} | {o_t:>6} | {p1_fls:>10} | {p2_fls:>10} | "
            f"{data['completed_trades']:>6d} | {data['raw_invalid_count']:>7d}"
        )

print("\n" + "=" * 90)
print("BASELINES SUMMARY ACROSS 6 TRANSITIONS")
print("=" * 90)
for seq, b in baselines.items():
    print(f"\nSEQUENCE: {seq}")
    for k, v in b.items():
        print(
            f"  {k:<14} | Ret: {v['total_return_pct']:>+6.2f}% | MaxDD: {v['max_drawdown_pct']:>5.2f}% | "
            f"Trades: {v['completed_trades']:>3d} | WinRate: {v['win_rate_pct']:>5.1f}% | "
            f"Fees: ${v['total_fees']:>5.2f} | Funding: ${v['total_funding']:>5.2f}"
        )
