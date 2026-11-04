import json

with open("experiments/multi_regime/results/benchmark_results.json") as f:
    data = json.load(f)

print("=== DATASET REGIME SUMMARY ===")
for k, v in data["dataset_statistics"].items():
    print(f"{k:<12}: Ret={v['return_pct']:+6.2f}%, VolAnn={v['vol_annual_pct']:4.1f}%, MaxDD={v['max_drawdown_pct']:5.2f}%, UpBars={v['up_bar_ratio']*100:.1f}%")

print("\n=== BASELINES SUMMARY ===")
for regime, b_dict in data["baselines"].items():
    print(f"-- Regime: {regime} --")
    for b_name, b_res in b_dict.items():
        print(f"  {b_name:<15}: Ret={b_res['total_return_pct']:+6.2f}%, Trades={b_res['completed_trades']:2d}, Flat/Long/Short={b_res['time_flat_pct']:.0f}%/{b_res['time_long_pct']:.0f}%/{b_res['time_short_pct']:.0f}%, Fees=${b_res['total_fees']:.2f}")

print("\n=== DQN MODELS EVALUATION SUMMARY ===")
for m in data["dqn_models"]:
    print(f"\n=======================================================")
    print(f"Model: {m['run_id']} | Type: {m['model_name']} | Seed: {m['seed']} | Trained: {m['train_seconds']:.1f}s")
    print(f"=======================================================")
    for reg, ev in m["evaluations"].items():
        mq = ev["mean_q_values"]
        print(f"  [{reg:<10}]: Ret={ev['total_return_pct']:+6.2f}%, Trades={ev['completed_trades']:2d}, WinRate={ev['win_rate_pct']:.1f}%, Flat/Long/Short={ev['time_flat_pct']:.0f}%/{ev['time_long_pct']:.0f}%/{ev['time_short_pct']:.0f}%, Fees=${ev['total_fees']:.2f}, RawInv={ev['raw_invalid_count']}")
        print(f"      Masked Actions: {ev['masked_action_counts']}")
        print(f"      Raw Actions:    {ev['raw_action_counts']}")
        print(f"      Mean Q-Values : HOLD={mq['HOLD']:.4f}, L={mq['OPEN_LONG']:.4f}, CL={mq['CLOSE_LONG']:.4f}, S={mq['OPEN_SHORT']:.4f}, CS={mq['CLOSE_SHORT']:.4f}")

