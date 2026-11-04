import json

data = json.load(open('experiments/trade_lifecycle/results/lifecycle_evaluation_results.json'))
t42_trades = data['detailed_results']['trend_only_seed_42']
print("================ TREND ONLY SEED 42 TRADES ================")
for env_name, env_res in t42_trades.items():
    print(f"\n--- Environment: {env_name} (Total Trades: {len(env_res['trades'])}) ---")
    for t in env_res['trades'][:5]:  # print first 5
        print(
            f"  Trade #{t['trade_id']}: {t['direction']:<5} | Entry Step: {t['entry_step']:>3d} (P=${t['entry_price']:.1f}) -> "
            f"Exit Step: {t['exit_step']:>3d} ({t['exit_action']:<11}, {t['exit_type']:<15}, P=${t['exit_price']:.1f}) | "
            f"RealizedPnL: ${t['realized_pnl']:>+7.2f} | Fees: ${t['entry_fee']+t['exit_fee']:.2f} | NetPnL: ${t['net_pnl']:>+7.2f} | "
            f"ReturnedFlat: {t['returned_to_flat']}"
        )
    if len(env_res['trades']) > 5:
        print(f"  ... and {len(env_res['trades']) - 5} more trades.")
