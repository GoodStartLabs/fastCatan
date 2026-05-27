# models/selfplay/ — M3 Self-Play

> Bucket M3 from root `PLAN.md`. Other milestones (M1 fuzz, M2 retrain, M4
> AlphaBeta) are owned elsewhere; this dir is self-contained M3.

## Goal

Take the M2 PPO agent (seat-0 learner vs 3 random opponents) and push it past
itself by replacing the random opponents with **frozen snapshots of the agent's
own past selves**. Three deliverables:

1. **Iterative self-play schedule** — frozen snapshot rotation.
2. **Hyperparam sweep** — learning rate × entropy × snapshot interval.
3. **Gate** — latest model beats the N-rounds-ago snapshot **> 55%** head-to-head.

## Why this works on the existing engine (no C++ changes)

`Env.write_obs(seat, buf)` already emits a **perspective-flipped** observation:
the obs for any seat is in the exact format the seat-0 learner trained on (root
`PLAN.md`: *"Single-agent Gymnasium + perspective flip for self-play"*). So a
policy trained as seat 0 plays seats 1/2/3 unchanged — just feed it that seat's
POV obs. Self-play needs **zero** simulator changes: opponents are driven from
Python, same place `FastCatanEnv` already drives the random opponents.

## Shape note (read before training)

The interface is now **`OBS_SIZE=1084`, `NUM_ACTIONS=286`** — rebuilt 2026-05-27
in the **anaconda** env (`/home/sinan/anaconda3/bin/python`); `.venv` is still the
stale **724/296** build. ⚠️ The existing checkpoints (this dir's `sp_smoke_*` /
`sweep`, and the M2 `ppo_capped_50m`) were all trained at **724/296** and are
**obsolete against the 1084 build** — they will not load against it. Everything
here reads `fastcatan.OBS_SIZE` / `NUM_ACTIONS` **dynamically**, so the code
survives the rebuild, but **retrain the seed checkpoint on 1084 first**, then
warm-start self-play from that. (The `ppo_random_768` / `ppo_random_10m` named in
older versions of this note were deleted 2026-05-27.)

## Layout

```
models/selfplay/
├── PLAN.md            (this file)
├── opponents.py       Opponent interface, RandomOpponent, PolicyOpponent, OpponentPool
├── selfplay_env.py    SelfPlayEnv(FastCatanEnv): seats 1-3 = pool snapshots
├── train_selfplay.py  rotation loop: warm-start, train rounds, freeze, rotate, inline gate
├── gate.py            head-to-head latest-vs-N-ago, win rate + Wilson CI, >0.55 gate
├── eval_seats.py      per-seat win rate (newest seat0 vs 3 older); + --equal-baseline
└── sweep.py           grid runner (lr × ent × snapshot-interval × arch) → results table
```

Namespace package (no `__init__.py`, matching `models/`); run via
`python -m models.selfplay.<mod>`.

## Design

### Opponent pool & rotation (`opponents.py`, `train_selfplay.py`)

- `OpponentPool` holds frozen snapshots + one `RandomOpponent` (index 0 / fallback).
- **Per episode** it samples a `{seat: opponent}` map for seats 1-3:
  - prob `p_random` → random opponent (anti-collapse / keeps the M2 skill alive),
  - else uniform over the last `window` snapshots (recency-weighted self-play).
- Trainer keeps **one** `MaskablePPO` model + **one** `DummyVecEnv`; the pool is
  **shared and mutated** between `learn()` chunks. Each round:
  1. `model.learn(steps_per_round, reset_num_timesteps=(round==0))`
  2. `model.save(snap_N.zip)`; `pool.add(PolicyOpponent.load(snap_N.zip))`
  3. inline gate: `snap_latest` vs `snap_{latest-gate_lag}`.
- `snapshot interval = steps_per_round` (one freeze per round) → the swept knob.

Warm-start uses `model.set_parameters(init_ckpt)` (weights only) so sweep
hyperparams (lr, ent_coef) take effect while reusing M2 weights. Default policy
arch only (must match the checkpoint).

### Gate (`gate.py`) — balanced 2-vs-2

`play_2v2`: **2 seats latest, 2 seats N-ago**, seat assignment rotated every game
(interleaved complementary `{0,2}`/`{1,3}`) so each model occupies every seat
equally — cancels seat bias. Metric = the latest *team's* share of decided games;
each team holds 2 of 4 seats so **equal policies → 0.50**, and **>0.55 = better**
(the thesis's intended semantics). Win share + 95% Wilson CI; PASS iff conclusive
(low no-winner) AND share > threshold. Sampling policy (argmax trips the
trade-loop stall, root `PLAN.md`).

Verified on the 5M smoke (`--no-p2p-trade`, N=120): r4-vs-r4 = **0.483** (parity,
neutral confirmed), r4-vs-r3 = **0.642** PASS, r4-vs-r1 = **0.875** PASS. The old
1-vs-3 form (neutral 0.25, `>0.55` = "win >2× fair share") is gone; single-seat-
vs-three diagnostics live in `eval_seats.py`.

### Sweep (`sweep.py`)

Grid over `--lr × --ent-coef × --steps-per-round`. Shells out one
`train_selfplay` run per cell (isolation), each writes `summary.json`
(config + final gate rate + pass). Aggregates into a markdown/CSV table.

## Verification

1. **Imports + plumbing**: tiny run (`--num-envs 2 --steps-per-round 512
   --num-rounds 1 --gate-games 8`) completes; pool grows; gate prints a rate.
2. **Opponent legality**: every opponent action is in the legal mask (env steps
   never raise on an illegal action).
3. **Gate sanity** (2-vs-2): `latest == nago` → share ≈ 0.50 (verified 0.483);
   clearly-stronger latest → share > 0.55 (r4-vs-r1 = 0.875).
4. **Gate run**: after a real schedule, `python -m models.selfplay.gate` reports
   latest-vs-N-ago > 0.55. **M3 gate met.**

## Known finding — the trade-loop stall blocks the gate (read before a run)

Smoke (2026-05-27) verified the plumbing AND surfaced a real blocker:

- **Env is correct.** Random-vs-random in `SelfPlayEnv`: 12/12 games reach a
  winner. Policy-vs-random via `models.eval`: ~0.68, 28/30 decided. The
  harness/winner-detection are sound.
- **Strong policy in all 4 seats stalls.** `ppo_random_768` as both `latest`
  and `nago`: **100% no-winner** — every game hits the step cap with nobody at
  10 VP. This is the TRADE_OPEN/CANCEL loop (root PLAN.md): 4 trade-happy
  policies never close out. The gate is then *undecidable* (decided=0), which
  `gate_result.conclusive` now reports honestly instead of a bogus 0.0 FAIL.
- **Two fixes:**
  1. **`--no-p2p-trade`** (this dir, Python mask AND-NOT): forbids p2p trades in
     train AND gate. With it, the same matchup goes **12/12 decided, 0 no-winner**.
     Opt-in; use the SAME setting in train and gate.
  2. **Proper fix = C++ mask cap on TRADE_OPEN re-opens per turn** — the open M2
     item in root PLAN.md. Lives in the simulator, not here. Once it lands,
     drop `--no-p2p-trade` and the gate works on the full game (trading intact),
     which is what the thesis wants.
- **Neutral baseline is 0.25, and the >0.55 gate bar is miscalibrated.**
  Equal policies (newest at all 4 seats, `--no-p2p-trade`, N=120 via
  `eval_seats.py --equal-baseline`) split **0.233/0.250/0.233/0.283 ≈ 0.25 each
  — no seat-0 advantage** (an earlier N=12 "~0.42" reading was noise; retracted).
  This motivated the gate recalibration. **DONE**: the gate is now balanced
  2-vs-2 (`play_2v2`, see Design) — 2 seats latest, 2 seats N-ago, rotated, so
  neutral = **0.50** and `>0.55` is meaningful. Verified: equal policies → 0.483.

## Smoke runs (2026-05-27, warm-start `ppo_random_10m`, `--seed-pool --no-p2p-trade`)

Throughput: ~1.3k fps (opponents on CPU, learner CUDA; ~2× the 610 fps with
opponents on CUDA — single-obs inference has no GPU-launch overhead on CPU).

| run | schedule | gate (latest vs prior, N=100) |
|-----|----------|-------------------------------|
| 1M  | 4 × 250k | r1 0.49, r2 0.32, r3 0.32 — never passed |
| 5M  | 5 × 1M   | **r1 0.56 PASS**, r2 0.54, r3 0.42, r4 0.34 |

**Self-play works and improves monotonically** — mechanically flawless (0%
no-winner, every gate conclusive, clean CIs). Per-seat 4-way eval (5M, N=200,
`eval_seats.py`, newest=seat0 vs r3/r2/r1 at seats 1-3):

| seat | round | win rate |
|------|-------|----------|
| 0 (newest) | r4 | **0.420** |
| 1 | r3 | 0.335 |
| 2 | r2 | 0.185 |
| 3 | r1 | 0.060 |

A clean recency-ordered skill gradient (r4>r3>r2>r1). Combined with neutral=0.25
(equal baseline above), **every gate round beat its predecessor** — the 5M gates
0.56/0.54/0.42/0.34 are all > 0.25, i.e. each round improved; the falling
numbers are a *shrinking per-round margin* (diminishing returns), NOT the
forgetting/instability an earlier draft claimed (retracted).

Findings:
1. **Snapshot interval is load-bearing** (a sweep axis): 1M/round produced a much
   bigger r0→r1 jump (gate 0.56) than 250k/round (0.49). Longer per-round
   training lets the learner catch the frozen opponent before the next freeze.
2. **Diminishing returns by r4** (gate margin 0.56→0.34): later rounds add less.
   Levers to sustain gains — lr decay / KL target, larger/older `--pool-window`,
   longer interval — are the sweep's job (lr × ent × interval × arch).

## Status

- [x] dir + scaffolding (`opponents`, `selfplay_env`, `train_selfplay`, `gate`, `sweep`).
- [x] plumbing smoke green (warm-start, snapshot rotation, inline gate, summary,
      sweep grid expansion + `--net-arch` axis, gate-logic edge cases).
- [x] gate > 0.55 recorded once (5M r1 = 0.56, conclusive, 0% no-winner).
- [x] monotonic improvement confirmed (per-seat 4-way: r4>r3>r2>r1; every gate >0.25).
- [x] gate recalibrated to balanced 2-vs-2 (neutral 0.50; verified equal→0.483,
      r4>r3 = 0.642 PASS). Under it the 5M consecutive gates pass.
- [x] diminishing-returns levers folded into the sweep: `--lr-schedule
      linear` (per-round global lr decay — verified the optimizer lr steps
      4e-4→1e-4; works around SB3's per-`learn()` progress reset) and
      `--target-kl` (PPO KL early-stop). `--pool-window` already an axis.
      `sweep.py` now grids lr × ent × interval × arch × **sched × kl**.
- [ ] run the sweep — **BLOCKED**: all `checkpoints/` were wiped by the M2
      retrain (`ppo_random_10m/768` gone), so there is no seed to `--init-from`.
      Binary is still 724/296 (not yet rebuilt to 1084/286). Needs a fresh M2
      checkpoint on the current binary before a warm-started sweep; running
      from scratch is a poor M3 test (M2 took 10M just to beat random).
- [ ] decide stall fix long-term: keep `--no-p2p-trade` vs C++ TRADE_OPEN cap
      (the thesis wants trading intact → prefer the C++ cap before final runs).
- **artifacts:** smoke ckpts in `checkpoints/sp_smoke_{1m,5m}/` (~16 MB, untracked —
  `checkpoints/` has no .gitignore rule yet; scratch, safe to delete).
