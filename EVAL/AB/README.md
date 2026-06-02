# EVAL/AB/ — M4: Alpha-Beta eval + final-model thesis gate

The thesis claim lives here: **the trained RL agent beats Catanatron's
Alpha-Beta with statistical significance — win rate > 25% over ≥1000 four-player
games, 95% CI** (`PLAN.md` M4, root). 0.25 is the 4-player chance baseline.

The agent plays through `EVAL/bridge/CatanatronBridge` *inside Catanatron's reference
engine*, so the numbers are directly comparable to Catanatron paper baselines.

## Native AlphaBeta — a fast, faithful training opponent

The eval above runs Catanatron's **real** AlphaBeta through the bridge (~6.4 s/game,
crashes on P2P trades) — fine for the final gate, far too slow to *train* against.
So the same player is ported natively into the fastcatan C++ engine
(`src/catan/search.cpp`, `include/search.hpp`), exposed as:

```python
env = fastcatan.Env(); env.reset(seed)
env.ab_decide(pov, depth=2, prune=False)   # -> best flat action id (0xFFFFFFFF if none)
env.ab_value(pov)                          # -> Catanatron base_fn heuristic value
```

It is a faithful port of `catanatron.players.minimax.AlphaBetaPlayer`
(depth-2 expectimax over dice / dev-draw / robber-steal chance nodes, alpha-beta,
`list_prunned_actions`) + `value.base_fn` (`DEFAULT_WEIGHTS`). The engine refactor
that makes the chance forks possible is `rules.cpp::expand_action`
(forced-outcome cores split out of the RNG handlers; the RNG sim path stays
byte-identical — perft hash unchanged).

**Fidelity (validated, `test_native_ab_fidelity.py`, run via the bridge):**
- `ab_value` == `base_fn(DEFAULT_WEIGHTS)` to **machine precision** (worst rel
  error 1.9e-16 over 4800 state×seat pairs; exact in MAIN phase).
- On deterministic 1:1-action decisions, Catanatron's depth-1 pick achieves
  **exactly** fastcatan's best value (100 %) — every raw move difference is a
  pure value tie (different tie-break order).
- Two deliberate, documented deviations (both *more* correct than the
  reference): BUY_DEV forks the true remaining deck; robber-steal forks the
  victim's real hand (Catanatron uses an info-set blur / flat 1/5).

**Train against it** (`models/env.py`, `models/train_ppo.py`):

```bash
python -m models.train_ppo --opponent alphabeta --ab-depth 1 --num-envs 768 ...
```

Throughput (single env): `random` ~51k learner-steps/s, **depth-1 ~45k**
(≈ Catanatron `ValueFunctionPlayer`, nearly free and already crushes a random
learner), depth-2 ~5k (~10× slower). Depth-1 is the recommended training
opponent; bump to depth-2 / `--ab-prune` for a stronger curriculum. This is the
"opponent-in-pool" lever for the M4-blocked-on-M3 gap.

Pure-engine checks live in `tests/test_alphabeta.py`; the catanatron-fidelity
gate in `test_native_ab_fidelity.py` (this dir).

## Files

| File | Role |
|---|---|
| `policy.py` | wraps a trained checkpoint as a bridge `PolicyFn` (`obs, mask, rng -> int`). Registry mirrors `models/eval.py`; only `ppo` wired today. Raises on obs/action-dim mismatch. |
| `tournament.py` | the harness: policy-via-bridge vs `AlphaBetaPlayer`/`ValueFunctionPlayer`/`RandomPlayer`. Win rate + 95% Wilson CI + thesis gate → `results/*.json`. |
| `soak.py` | 10⁸-step stability soak (pure fastcatan): finite-obs + mask-integrity + leak checks. |
| `REPRODUCIBILITY.md` | toolchain, build flags, the **two-env** setup, **catanatron git pin**, seeds, train config. |
| `results/` | tournament result JSONs + `validation_1084.md` (pipeline validation). |

## Environment — anaconda

The RL interface is **obs 1084 / actions 286**, and all M4 work runs in the
**anaconda** interpreter (see `REPRODUCIBILITY.md` §5):

- **`/home/sinan/anaconda3/bin/python`** — 1084/286 fastcatan + catanatron 3.3.0
  + sb3. **Train and eval M4 here.**

Catanatron is a **pinned git build, not PyPI** (3.3.0 @ `41ba0db`, "deterministic
discards"); installed editable from `/home/sinan/Desktop/msc/catanatron` and
recorded in root `requirements.txt`. Verified at this commit: 281/281
`EVAL/bridge/tests` pass (2026-05-27). `soak.py` needs only fastcatan and runs in
either env. See `REPRODUCIBILITY.md` §6 for why the pin must be a commit (newer
builds move `models.tiles` → `models.map` and break bridge import).

## Run (anaconda)

```bash
AP=/home/sinan/anaconda3/bin/python

# 1) Train the vs-random seed on the 1084/286 interface (768 envs).
$AP -m models.train_ppo --num-envs 768 --total-steps 50_000_000 --run-name ppo_1084_50m
#    >=30M for gate margin; 50M is the verified M2/M3 seed.

# 2) Thesis gate (slow — AlphaBeta ~6.4 s/game unpruned, ~1.8 h/1000): vs Alpha-Beta.
#    --no-trades is required: Catanatron's AlphaBeta crashes on P2P trade actions.
PYTHONHASHSEED=0 PYTHONPATH=EVAL $AP -m AB.tournament \
    --ckpt models/checkpoints/ppo_1084_50m/ppo_final.zip \
    --games 1000 --opponent alphabeta --ab-depth 2 --ab-prune --seed 42 --no-trades

# Smoke (seconds): vs random.
PYTHONPATH=EVAL $AP -m AB.tournament --games 20 --opponent random --ckpt models/checkpoints/ppo_1084_50m/ppo_final.zip

# 10^8 soak (~minutes at ~70k steps/s).
PYTHONPATH=EVAL $AP -m AB.soak --steps 100000000 --seed 7
```

The evaluated model is a `--ckpt` flag — swap in any future M3 self-play
checkpoint (must be 1084/286) freely.

## Status (2026-05-28)

- [x] PPO→bridge policy adapter (`policy.py`) — smoke: 30/30 legal picks.
- [x] Tournament harness (`tournament.py`) — win rate + 95% CI + gate + JSON.
- [x] Soak harness (`soak.py`) — smoke: 10k steps, RSS flat (1.00×), STABILITY PASS.
- [x] catanatron pinned to git `41ba0db` (3.3.0, not PyPI), installed editable +
      recorded in root `requirements.txt`. Bridge verified: 281/281 tests pass.
- [x] Build is 1084/286 in anaconda (`editable.rebuild=true` so it can't go stale).
- [x] 1084 pipeline validated end-to-end: `test_obs_identity` 5/5 (encoder↔C++
      parity) + uniform-bridge games vs Value/AlphaBeta complete. See
      `results/validation_1084.md`.
- [x] Reproducibility doc.
- [x] **M2 seed `ppo_1084_50m` trained** (50M, 1084/286) — 95.5% vs random native,
      89.5% via bridge vs `RandomPlayer`.
- [~] Final model vs Alpha-Beta, ≥1000 games — harness ran live on `ppo_1084_50m`:
      **0/200 vs AlphaBeta** (`--no-trades`, gate FAIL), a real result (same model
      beats `RandomPlayer` 89.5% through the same bridge). Needs the stronger M3
      self-play model.
- [ ] Full 10⁸-step soak (smoke green; ~24 min full).
- [ ] Record thesis-gate result.
