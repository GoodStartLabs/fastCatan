"""Standalone correctness check for BeliefTracker vs true env.player_resource.
Runs the generator's exact regime (p2p banned) with AB-d1 play."""
import numpy as np, random
import fastcatan as fc
from models.belief.belief_tracker import BeliefTracker
from examples.player_base import build_p2p_trade_filter

env = fc.Env()
rng = random.Random(7)
mask = np.zeros(fc.MASK_WORDS, np.uint64)
p2p = build_p2p_trade_filter()
errs, tots = [], []
GAMES = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 150
for g in range(GAMES):
    env.reset(rng.getrandbits(64))
    tr = BeliefTracker(); tr.reset(env)
    for ply in range(4000):
        if int(env.phase) == 3:
            break
        env.action_mask(mask)
        m = mask & ~p2p
        if not int(m.any()):
            m = mask
        legal = []
        for wi, w in enumerate(m):
            w = int(w); b = wi * 64
            while w:
                bit = (w & -w).bit_length() - 1; legal.append(b + bit); w &= w - 1
        if not legal:
            break
        cp = int(env.current_player); ph = int(env.phase)
        a = env.ab_decide(cp, 1, False) if len(legal) > 1 else legal[0]
        if a not in legal:
            a = rng.choice(legal)
        _, done = env.step(int(a))
        tr.after_step(env, cp, int(a), ph)
        if ply % 12 == 0:
            e, t = tr.validate_error(env)
            errs.append(e); tots.append(t)
        if done:
            break
me, mt = float(np.mean(errs)), float(np.mean(tots))
print(f"games={GAMES} samples={len(errs)}")
print(f"mean L1 err/seat (cards) = {me:.4f}")
print(f"mean true total/seat (cards) = {mt:.4f}")
print(f"relative L1 error = {me/max(mt,1e-9):.4f}")
print(f"per-resource mean abs err = {me/5:.4f} cards")
