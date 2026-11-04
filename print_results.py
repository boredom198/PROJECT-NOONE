import json

with open("runs/experiments_exploration_schedule/experiment_summary.json") as f:
    data = json.load(f)

for r in data:
    ed = r["evaluation_diagnostics"]
    td = r["training_diagnostics"]
    print("=" * 70)
    print(f"{r['run_id']} | T={r['total_timesteps']} | Seed={r['seed']} | Time={r['train_seconds']:.1f}s")
    print("TRAINING DIAGNOSTICS:")
    print(f"  Transitions: {td['total_transitions']}")
    print(f"  Valid actions: {td['valid_actions']}")
    print(f"  Invalid actions: {td['invalid_actions']} (Total: {td['total_invalid_actions']})")
    print(f"  Total penalties applied: {td['total_penalties_applied']:.4f}")
    print(f"  Completed round trips: {td['completed_round_trips']}")
    print(f"  Position time: Flat={td['time_flat_pct']:.2f}%, Long={td['time_long_pct']:.2f}%, Short={td['time_short_pct']:.2f}%")
    print(f"  Epsilon verification:")
    for k, v in td["epsilon_checks"].items():
        print(f"    {k}: step={v['step']}, eps={v['epsilon']:.4f}, expected={v['expected']:.4f}, verified={v['verified']}")
    print(f"  Q-distributions: Q_mean={td['q_distributions'].get('q_mean'):.4f}, Q_std={td['q_distributions'].get('q_std'):.4f}, Target_Q_mean={td['q_distributions'].get('target_q_mean'):.4f}, Target_Q_std={td['q_distributions'].get('target_q_std'):.4f}")
    print(f"  Loss: initial={td['loss_initial']:.6f}, final_mean={td['loss_final_mean']:.6f}")
    print("EVALUATION DIAGNOSTICS:")
    print(f"  Raw actions: {ed['raw_action_counts']}")
    print(f"  Masked actions: {ed['masked_action_counts']}")
    print(f"  Raw invalid count: {ed['raw_invalid_count']} ({ed['raw_invalid_rate_pct']:.2f}%)")
    print(f"  HOLD frequency: {ed['hold_frequency_pct']:.2f}% (raw: {ed['raw_hold_frequency_pct']:.2f}%)")
    print(f"  Completed trades: {ed['num_trades']}")
    print(f"  Total return: {ed['total_return_pct']:.4f}%")
    print(f"  Max drawdown: {ed['max_drawdown_pct']:.4f}%")
    print(f"  Win rate: {ed['win_rate_pct']:.2f}%")
    print(f"  Avg win: {ed['avg_winning_trade']:.4f}, Avg loss: {ed['avg_losing_trade']:.4f}")
    print(f"  Total fees: {ed['total_fees']:.4f}, Total funding: {ed['total_funding']:.4f}")
    print(f"  Final equity: {ed['final_equity']:.2f}")
    print(f"  Baselines -> Always HOLD: {ed['baseline_always_hold_return_pct']:.4f}%, Random: {ed['baseline_random_return_pct']:.4f}%")

