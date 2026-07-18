# Batched trainer

`BatchedTrainer` drives one frozen `fastcatan.BatchedEnv` end-to-end: batched
current-seat signatures, legal masks, POV observations, one GPU forward per
active neural policy, vectorized random-legal or native batched-AB opponents,
and one batched environment step.  Policy slot 0 is the learner; all four slots
are independently permuted across seats per environment and reshuffled on that
environment's auto-reset.

Learner transitions span opponent turns.  A pending learner decision is closed
by its next decision (zero reward plus bootstrap) or by terminal reward from
`last_winner` (+1 for the learner seat, -1 otherwise, including no-winner).  The
actor input path is always the 1,084-float `write_obs_pov_batch` output.

## One-command smoke

```bash
models/batched_trainer/train_batched_smoke.sh
```

The command runs the ported wiring tests, frozen leakage referee, then a 30-minute
PPO run with a hard 100k learner-decisions/s assertion, finite-loss/entropy gates,
a moving raw-policy promotion probe, W&B logging, and a final checkpoint.  It
automatically caps OpenMP and leakage workers at two while the reproduction tmux
session exists.  `substrate-v1` freezes all `bin/` additions, so the executable
lives here on the explicitly editable research surface.

Research-run defaults are 4,000 environments, `(512,512,256)` trunk,
262,144-decision rollouts, 65,536-sample minibatches, one update epoch, AdamW
`3e-4`, `gamma=0.997`, `lambda=0.95`, and three random-legal opponents.  Replace
the opponent pool with any three comma-separated vectorizable specs: `random`,
`self`, `ab1`, `ab2`, or `checkpoint:/path/net.pt`.

Use `--anchor-ref`, `--anchor-coef`, and `--anchor-coef-final` for an annealed,
frozen forward-KL anchor.  `BatchedTrainer.update_distillation` uses the same
network/optimizer/checkpoint path with integer or dense MCTS policy targets plus
search value targets.
