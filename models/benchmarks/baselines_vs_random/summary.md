# Reactive RL baselines vs random (50000000 steps, no-trades)

_Generated 2026-06-10T18:10:38+00:00 · opponent: random · 3 agent(s)_

| Algo | Train steps | Trades | Games | Win% vs random | 95% CI | M2 gate | Seat wins | Train time | Git SHA |
|---|---|---|---|---|---|---|---|---|---|
| PPO | 50M | off | 1000/1000 | 95.3 | [93.8, 96.4] | PASS | [953, 12, 20, 15] | 45.9m | c67c167 |
| A2C | 50M | off | 1000/1000 | 87.4 | [85.2, 89.3] | FAIL | [874, 39, 44, 43] | 50.4m | c67c167 |
| DQN | 50M | off | 1000/1000 | 71.8 | [68.9, 74.5] | FAIL | [718, 112, 87, 83] | 1.3h | c67c167 |

