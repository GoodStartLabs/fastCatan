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

The interface is **`OBS_SIZE=1084`, `NUM_ACTIONS=286`** — run in the **anaconda**
env (`/home/sinan/anaconda3/bin/python`). Everything here reads
`fastcatan.OBS_SIZE` / `NUM_ACTIONS` **dynamically**, so the code is interface-
agnostic. Warm-start self-play from the verified M2 seed
`models/checkpoints/ppo_1084_50m/ppo_final.zip`.

## Layout

```
models/selfplay/
├── PLAN.md            (this file)
├── opponents.py       Opponent interface, RandomOpponent, PolicyOpponent, OpponentPool
├── league.py          League: bounded "best-N" archive + PFSP matchmaking (opt-in --league)
├── selfplay_env.py    SelfPlayEnv(FastCatanEnv): seats 1-3 = pool/league snapshots
├── train_selfplay.py  rotation loop: warm-start, train rounds, freeze, rotate, inline gate, --resume
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
  2. `model.save(snap_N.zip)`; `pool.add_candidate(PolicyOpponent.load(snap_N.zip), snap_N.zip)`
     (`add_candidate` is a window-pool alias for `add`; the League uses the path for eviction)
  3. inline gate: `snap_latest` vs `snap_{latest-gate_lag}` (loaded from `snap_paths` on disk).
- `snapshot interval = steps_per_round` (one freeze per round) → the swept knob.

Warm-start uses `model.set_parameters(init_ckpt)` (weights only) so sweep
hyperparams (lr, ent_coef) take effect while reusing M2 weights. Default policy
arch only (must match the checkpoint).

**Crash recovery (`--resume`).** Re-running is a *restart* (round loop from 0,
empty pool, snapshot numbering + gate log reset). `--resume` makes it a real
*continue*: glob `snap_*.zip` → rebuild the pool → `MaskablePPO.load()` the
latest (optimizer + `num_timesteps`, so numbering keeps going) → start at round
N → append the gate log. The schedule/gate flags are persisted to
`run_config.json` on a fresh run and restored on resume (so a continue can't
miscalibrate the linear-lr decay or gate rules); the gate log is streamed to
`gate_log.jsonl` per round so there's something to append to. Just re-pass
`--run-name`. The inline gate now loads its two contestants from the append-only
`snap_paths` (on disk), not the pool, so "latest vs N-ago" is correct under
resume AND when a league has evicted the N-ago snapshot.

### League & PFSP (`league.py`, opt-in `--league`)

An alternative to the sliding-window pool: a **bounded archive of the `--league-
size` (default 32) best snapshots** sampled by **prioritized fictitious self-play**
(AlphaStar). It's a drop-in (`sample() -> {seat: opponent}`), so `SelfPlayEnv` is
unchanged and you can A/B window-pool vs league.

- **PFSP weight** over `p_i` = learner's smoothed win-rate vs member *i* (Laplace
  prior → 0.5 when unseen): `--pfsp hard` (default) `w=(1-p)^β` favors opponents
  you *lose* to (fix weaknesses); `--pfsp even` `w=p(1-p)` favors evenly-matched.
  `p_random` random opponents are still mixed in (anti-collapse).
- **Stats are free.** `SelfPlayEnv.step` credits every league opponent at the
  table (a win iff seat 0 won, terminal reward > 0) into the shared pool — no
  extra games. `--league-decay` (e.g. 0.9/round) fades old counts so PFSP tracks
  the *improving* learner.
- **"Best 32" = hybrid recency + difficulty.** Always keep the most-recent
  `--league-recent` (default 8; the newest are strongest and a just-frozen model
  has no stats); fill the rest with the hardest-for-the-learner (lowest `p`),
  evicting the *easiest* non-recent member on overflow. Bounded → it can forget
  old easy-to-beat strategies; that's the deliberate cost of the cap (an
  unbounded league wouldn't, at higher memory/compute).
- **Credit assignment** is the 4-player approximation: each distinct member in a
  game shares that game's binary outcome (vs AlphaStar's clean 1-v-1). Monotone
  and good enough for matchmaking; documented in `league.py`.
- **Resume:** archive membership + counts are persisted to `league_state.json`
  each round and restored on `--resume` (the bounded/evicted subset can't be
  reconstructed from the snap set alone).

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
- [x] `--resume` crash-continue: glob snaps → rebuild pool → `MaskablePPO.load`
      latest (optimizer + `num_timesteps`) → start at round N → append gate log;
      `run_config.json` + `gate_log.jsonl` persisted, gate loads from `snap_paths`.
      Helper logic unit-tested (numeric snap ordering, gate-log append, config
      restore).
- [x] league / PFSP (`league.py`, `--league`): bounded best-N archive, PFSP
      hard/even sampling, hybrid recency+difficulty eviction (`--league-recent`),
      free per-opponent stats from rollouts, `--league-decay`, resume via
      `league_state.json`. Logic unit-tested (eviction, dedup crediting, PFSP-hard
      sampling, decay, state round-trip). **NOT yet run at scale.**
- [x] ran the 12-cell arch sweep (2026-05-28, per-arch seeds via `--init-dir`,
      `--trade-compose-cap 20`) → `checkpoints/arch_sweep/sweep_results.{md,csv}`.
      **Verdict: inconclusive, not a clean win.** Win-shares where decided were
      strong (0.65–0.84 → the learner IS beating its past selves), but the 2v2
      gate stalled **50% no-winner (128-128) to 85% (256-256)**, so only 1/12
      cells (128-128/ent0.01/constant, share 0.70) squeaked past the 50%-decided
      conclusive bar. cap=20 is NOT enough at this scale (see [[selfplay-trade-cap-fixed]]).
- [x] decided stall fix (2026-05-29): **training-signal + relaxed cap**, not
      `--no-p2p-trade` and not (yet) the C++ cap. `TIE_REWARD=-2` (no-winner now
      strictly worse than a loss → learner pushed to close out), `MAX_TRADE_
      COMPOSE_PER_TURN` 20→40, `MAX_EPISODE_STEPS` 3000→4000, gate/eval max-steps
      4000→5000. The mask cap is now a backstop; the reward is the primary lever.
- [x] re-ran the sweep with cap40/-2/step-caps (2026-05-29) → **FAILED, worse**:
      mean 0.93 no-winner, 0/12 conclusive (vs 1/12 at cap20). Loosening the
      compose cap, not the reward, was the cause: trade-steps inflate steps/turn so
      the *learner-step* length cap guillotines games before they reach the
      ~300-900 turns needed to win (random wins by turn 945). Archived
      `sweep_results_cap40_tie-2.*` vs `_cap20_tie-1.*`.
- [x] **fixed: length now bounded by TURNS, not steps** (2026-05-29). C++
      `MAX_TURNS=2000` length cap (state.hpp / rules.cpp `step_one`), compose cap
      40→50, python step caps demoted to backstops (MAX_EPISODE_STEPS 40000,
      gate/eval max-steps 150000). Smoke on the worst cap40/-2 ckpts: **0%
      no-winner on all cells** (was 79-100%), all decided; the turn cap never fired
      (games end 173-595 turns) — the step cap (5000) was guillotining trade-heavy
      games that need 7k-61k steps. Details in memory `cap-simplification-plan`.
- [x] **reran the full 12-cell sweep on the fixed env** (2026-05-30,
      cap50+turn-cap) → `sweep_results_cap50_turncap.*`. **Decidability perfect:
      12/12 conclusive, 0.000 no-winner across every cell AND round.** Real
      learning gradient: **6/12 cells PASS (>0.55)** — M3 gate MET with trades ON.
      Best: 256,256 ent0.01 constant **0.825**, then 256,256 ent0.01 linear 0.765,
      128,128 ent0.01 linear 0.755. Pattern: ent0.01 >> ent0 (all top-3 are
      ent0.01; ent0 collapses, esp. on 256,256); 64,64 weakest (1/4 pass).
- **artifacts:** smoke ckpts in `checkpoints/sp_smoke_{1m,5m}/` (~16 MB, untracked —
  `checkpoints/` has no .gitignore rule yet; scratch, safe to delete).
