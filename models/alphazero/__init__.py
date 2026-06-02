"""AlphaZero for fastCatan.

Real-simulator MCTS (not a learned model — that's MuZero, see models/train_muzero.py).
The C++ env exposes snapshot()/load_snapshot()/reseed(), so the search branches the
true game state and resamples chance (dice/dev/steal) per simulation via reseed.

Design decisions (see git history / PLAN):
  - Full-state ("cheating") MCTS: search sees ground-truth hidden info. This is a
    FAIR comparison vs catanatron's AlphaBeta, which is also a full-information bot.
    The policy/value net still only sees POV-relative obs; only the search uses the
    true state.
  - 4-player value: the value head predicts win-prob for the seat to move. Leaf eval
    forwards all 4 seat POVs -> a length-4 value vector; backup is max^n (each seat
    maximizes its own Q).
  - Chance: reseed() before every step. Deterministic actions are unaffected (no RNG
    draw -> stable successor); chance actions resample, so outcome-keyed children
    appear in proportion to their probability.

Modules:
  net.py       PolicyValueNet (shared trunk -> policy + tanh value heads)
  mcts.py      full-state stochastic MCTS over a scratch fastcatan.Env
  selfplay.py  self-play game generation + training loop
  evaluate.py  vs-random evaluation (MCTS owns the env)
"""
