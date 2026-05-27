# models/ — RL Training Plan

> ## ⚠️ STATUS / CORRECTIONS — 2026-05-27 (read this first)
>
> **The design sections below (esp. §1, §4, §5) are ASPIRATIONAL and do NOT
> match the code.** What is actually true:
>
> - **Env is single-env, not a BatchedEnv VecEnv.** `env.py` = `FastCatanEnv`
>   (one `fastcatan.Env`, learner=seat 0, seats 1-3 random, stepped *inside*
>   `step()`). §1's `FastCatanVecEnv` over `BatchedEnv` was never built.
> - **`train_ppo.py` vectorizes via SB3.** Default is now **`DummyVecEnv`**
>   (`--subproc` to opt into SubprocVecEnv). DummyVecEnv is ~1.45× faster here
>   because the C++ sim is so cheap that per-step IPC pickling dominates.
> - **Shapes: `OBS_SIZE=1084`, `NUM_ACTIONS=286`** (286 mask bits in `uint64[5]`).
>   ⚠️ This is the **current build**, but the `.so` was a stale **724 / 296** build
>   until the **2026-05-27 rebuild** (the 1084 build is now the **anaconda**
>   interpreter; `.venv` is still 724). **`ppo_capped_50m` and every existing
>   checkpoint were trained at 724/296 and are OBSOLETE against the 1084 build**
>   (won't load — wrong obs *and* action dim); retrain on 1084 before M4 eval / M3
>   warm-start. (Verified: headers compile to 1084/286; `ppo_capped_50m` loads as
>   obs (724,) act 296. See `AB/REPRODUCIBILITY.md` §4–5.)
> - **Reward (env.py):** +1 learner win; **−1 for every non-win terminal**
>   (opponent win, no-winner, and a `turn_count>=MAX_TURNS` stall cap → −1,
>   `terminated=True`). ✅ **FIXED (2026-05-27):** two-part fix in `env.py`.
>   (1) **Primary** — a per-turn **trade-compose cap** in `action_masks()`
>   (`MAX_TRADE_COMPOSE_PER_TURN=20` masks the ADD/REMOVE/OPEN block, ids 268–288,
>   once spent; CANCEL/ACCEPT/DECLINE/CONFIRM stay legal so the mask is never
>   emptied). The real stall is a *within-turn* `ADD_WANT`→`CANCEL` churn that
>   never opens a trade nor ends the turn, so `turn_count` (only bumped on
>   END_TURN, `rules.cpp:540`) froze and the old cap never fired — and a pure
>   step-counter cap alone is **insufficient**: it truncates *winnable* games
>   (the policy wins, just after thousands of churn steps, so capping mid-churn
>   counts a win as a loss). (2) **Backstop** — episode cap recounted in
>   *learner steps* (`MAX_EPISODE_STEPS=3000`), not `turn_count`. The **M1 fuzz
>   independently proved a no-winner terminal matters**:
>   ~3/10⁷ random games never reach a winner (board built out + dev deck
>   exhausted ⇒ last VP unreachable — a rule-correct *deadlock*, not a bug; see
>   PLAN.md §M1). `step_one` has no no-progress terminal, so a no-winner
>   truncation → −1 is mandatory, not just defensive.
> - **PPO PASSES THE M2 GATE (2026-05-27).** `checkpoints/ppo_capped_50m/ppo_final.zip`
>   (MaskablePPO, 768 envs, 50M steps, ~32 min) wins **99.4% vs random**
>   (1000-game sampling, CI [0.987, 0.997], 0 no-winner) **and 99.5%** (200-game
>   deterministic, CI [0.972, 0.999]) — both clear the 0.90 CI-low bar. **10M
>   steps was too few** (training win-rate still climbing at ~0.83); it converges
>   ~15–20M. See [[ppo-training-reality]]. Throughput is NOT the bottleneck — the
>   BatchedEnv-direct PPO loop is deferred. **Obs normalization DONE (frozen)**;
>   reward shaping **rejected** (keep sparse ±1 — avoids reward hacking, clean
>   for M3 self-play).
> - **Eval modes:** with the trade-compose cap, both sampling and deterministic
>   terminate cleanly (0 no-winner in 1200 games). Deterministic underperformed
>   *only* on the undertrained 10M model (75.5% — argmax locks into degenerate
>   trajectories); the converged 50M model scores ~99% either way. Prefer
>   sampling while a model is still training.
> - `models/eval.py` is **single-env** (`FastCatanEnv`), not BatchedEnv.
>   Eval reward = `2·winrate − 1`, so `ep_rew_mean` is a live win-rate proxy.
> - Benchmarks: `bench/bench_throughput.py` (Python-path breakdown + bottleneck
>   naming + fastcatan-vs-catanatron equal footing), `bench/bench_comprehensive.py`
>   (distribution parity), `bench/bench_step.cpp` + `bench/bench_batched.cpp`
>   (pure-C++ floor). Catanatron quirks: see [[catanatron-seat-shuffle]].
>
> **M2 GATE: MET ✅** stall-cap fix → retrained 50M (`ppo_capped_50m`) →
> 1000-game gate passed (sampling 99.4%, deterministic 99.5%). ⚠️ Run on the
> **stale 724/296 build** — it *proves* the gate but is **obsolete vs the 1084
> build** (see ⚠️ at top); M3/M4 retrain there. The pre-cap runs (`ppo_random_10m`,
> `ppo_random_768`) and the undertrained 10M `ppo_capped_768` were **deleted**
> (2026-05-27) — trained without the cap or before convergence (`ppo_random_768`
> = 66.7% capped). Next: M3 self-play.
>
> **Obs/reward FROZEN (done).** Obs count fields normalized by structural Catan
> maxima — divisors in `src/catan/obs.cpp` `namespace norm`, mirrored in
> `bridge/obs_encoder.py` (`N_*`) + `ui/obs_decoder.py`; parity guarded by
> `bridge/tests/test_obs_identity.py` (keep all three in sync). Reward sparse
> ±1 terminal, non-win terminals = −1. The obs change **invalidated all old
> checkpoints** — `checkpoints/*` were deleted; retrain on the frozen interface.
> (✅ stall-cap-on-`turn_count` bug above is now fixed — primary fix is the
> per-turn trade-compose cap in `action_masks()`, with a `step()`-count backstop;
> M1 fuzz confirmed the deadlock the backstop guards against is real.)
>
> ---
>
> *(historical scaffolding plan below — kept for intent, but trust the block
> above where they conflict)*

## Context

Project enters **M2** (PLAN.md:193). Need first RL agent. Targets:
- Industry-standard algos, each in own file, all simple to read. Coverage:
  Q-Learning (DQN), Actor-Critic (A2C), PPO (MaskablePPO via sb3-contrib),
  MuZero (model-based + MCTS).
- All share one Gymnasium env (`env.py`) over `fastcatan.Env` (1084-dim obs,
  286 discrete actions, 320-bit/296-used legal-action mask).
- Learner controls seat 0; seats 1–3 = uniform-random-legal opponents
  (M2 gate: >90% vs random).
- Keep scripts *simple* — each file self-contained, no premature abstraction.
  Self-play (M3) and AlphaBeta eval (M4) come later in their own scripts.

## Layout

```
models/
├── PLAN.md              (this file)
├── env.py               Gymnasium env wrapping single fastcatan.Env (POV=seat0, opponents=random)
├── train_ppo.py         PPO (SB3 MaskablePPO)        — clipped surrogate, baseline
├── train_a2c.py         A2C (custom, ~200 lines)     — simplest actor-critic w/ mask
├── train_dqn.py         DQN/Q-Learning (custom)      — Q-net + target net + replay + eps-greedy
├── train_muzero.py      MuZero scaffold (custom)     — repr+dyn+pred nets + MCTS, demonstrative
├── eval.py              Load PPO checkpoint, play N games vs random, win rate + 95% CI
└── checkpoints/         (gitignored) snapshots + tensorboard logs
```

No package `__init__.py`. Scripts run via `python -m models.train_<algo>` or direct.

### Algo overview (intentionally simple)

| File | Style | Lib | When to read |
|------|-------|-----|--------------|
| `train_dqn.py` | value-based, off-policy | pure torch | learn DQN from scratch |
| `train_a2c.py` | policy-gradient + baseline, on-policy | pure torch | bridge from REINFORCE to PPO |
| `train_ppo.py` | clipped policy-gradient, on-policy | sb3-contrib | M2 gate run |
| `train_muzero.py` | model-based, MCTS planning | pure torch | reference for model-based RL |

Each file is self-contained — no shared utility module — so they can be read top-to-bottom in isolation.

## Design decisions

### 1. VecEnv adapter, not single-env

`BatchedEnv` is already vectorized (N envs, contiguous buffers, zero-copy numpy). Wrap it as `sb3_contrib`-compatible `VecEnv` directly — do **not** wrap one-at-a-time and then re-vectorize via `SubprocVecEnv` (would throw away the C++ vectorization).

`env.py` exposes:
- `FastCatanVecEnv(num_envs, seed, opponent_policy="random") -> VecEnv`
- Implements `reset()`, `step_async()/step_wait()`, `action_masks()` (required by MaskablePPO).
- Observation space: `Box(low=-inf, high=inf, shape=(OBS_SIZE,), dtype=float32)`.
- Action space: `Discrete(NUM_ACTIONS)` (296).
- **Inner loop**: after learner steps for seat 0, advance C++ env until `current_player()==0` again by sampling opponent actions from `opponent_policy(obs, mask)`. Random uses uniform-over-legal from `examples/player_base.py:legal_actions`.

### 2. Reward = native sim signal

Use as-is: +1 on win-action, -1 if action lets opponent win, 0 else (bindings.cpp). No shaping in M2. Locking obs+reward is itself an M2 deliverable (`PLAN.md:196`).

### 3. Action masking

`MaskablePPO.predict(obs, action_masks=...)` from `sb3_contrib`. Pull mask via `BatchedEnv.write_masks(buf)` → unpack uint64[5] → bool[296] per env. Helper already exists in `examples/player_base.py:legal_actions` — reuse.

### 4. Training defaults (`train_ppo.py`)

Industry defaults, light tweaks for discrete-action self-play:
- `policy="MlpPolicy"` (Box obs → no CNN needed). **Net size is the `--net-arch` CLI arg** (comma-separated, applied to pi & vf). Default `64,64` = the SB3 MlpPolicy default — *not* 256×256 as earlier drafts claimed. 64-wide on a 1084-dim obs is small: fine vs random (M2), likely a ceiling vs Alpha-Beta (M4). Scale at M3 (e.g. `--net-arch 256,256`) as a sweep axis; bigger net = slower fps (cheap C++ sim ⇒ the policy net is the throughput bottleneck).
- `n_steps=512`, `batch_size=4096`, `n_epochs=4`
- `learning_rate=3e-4`, `gamma=0.999` (Catan episodes are long, ~60–200 actions/seat)
- `ent_coef=0.01`, `clip_range=0.2`
- `num_envs=512` default (BatchedEnv handles this trivially)
- Tensorboard logging to `models/checkpoints/tb/`
- `CheckpointCallback` every 500k steps

### 5. Eval (`eval.py`)

- Load `.zip` checkpoint.
- Spin a small `BatchedEnv` (N=64), run 1000 games seat-0=learner vs seats-1..3=random.
- Compute win rate, 95% Wilson CI. Threshold: 0.90.
- Reuses `examples/player_base.legal_actions` for opponent action sampling.

## Files to add/touch

| File | New? | Purpose |
|------|------|---------|
| `models/PLAN.md` | new | this doc |
| `models/env.py` | new | `FastCatanVecEnv` |
| `models/train_ppo.py` | new | MaskablePPO trainer |
| `models/eval.py` | new | win-rate eval vs random |
| `pyproject.toml` | edit | add `[project.optional-dependencies] rl = ["torch", "stable-baselines3>=2.3", "sb3-contrib>=2.3", "gymnasium>=0.29", "tensorboard"]` |
| `.gitignore` | edit | add `models/checkpoints/` |

## Reused code (no new abstractions)

- `python/fastcatan/__init__.py` — `BatchedEnv`, `OBS_SIZE`, `MASK_WORDS`, `NUM_ACTIONS`
- `examples/player_base.py:legal_actions` — bit-unpack mask
- `examples/random_player.py` — opponent policy template
- `bridge/run_eval.py` — eval-loop shape (for later M4 cross-check vs Catanatron)

## Verification

1. **Smoke**: `python -m models.train_ppo --num-envs 64 --total-steps 100_000` runs without crash, tensorboard shows non-zero `ep_rew_mean` after ~30s.
2. **Mask correctness**: assert in `env.step_wait()` that `mask[action] == 1` for each picked action (debug flag).
3. **Determinism**: same seed → same first-1000-step obs/reward stream.
4. **Gate check**: after ≥5M steps, `python -m models.eval --ckpt latest --games 1000` reports win rate >0.90 vs random. M2 gate met.
5. **Throughput**: log steps/sec; should land within 2× of pure `BatchedEnv.step` benchmark (Risk 3 in root PLAN.md).
