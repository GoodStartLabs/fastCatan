# Code review — trade-stall fix (read before the next sweep)

> Review of the in-progress trade-stall fix (working tree, 2026-05-29). **No code
> changed — findings only.** Verified against the live `fastcatan` action layout
> and by importing every changed module. File:line refs are to the current tree.
>
> **Bottom line:** the new `ComposeCapper` is the *real* fix and is implemented
> correctly — the previous sweep capped only the learner (seat 0), so 3/4 training
> seats and **all 4 gate seats were uncapped**; that, not the cap *value*, is why
> cap=20 "wasn't enough." But two of the three headline changes push the wrong
> way for the thing that actually fails (gate **decidability**, not win-share):
> raising the cap 20→40, and a `-2` reward that only acts during training. The
> gate is also structurally ~3× stricter than training, independent of the cap.

---

## Verified action layout (286 build)

```
TRADE_BASE          = 210   bank/port trades 210..234  (NOT capped — self-limiting)
TRADE_ADD_GIVE_BASE = 268   268..272  (5)  ┐
TRADE_ADD_WANT_BASE = 273   273..277  (5)  ├ growth/churn block = TRADE_COMPOSE_IDS
TRADE_OPEN          = 278                  ┘  (np.arange(268,279) = 268..278, 11 ids)
TRADE_ACCEPT  = 279  TRADE_DECLINE = 280   ┐
TRADE_CONFIRM_BASE = 281  281..284 (4)     ├ resolve/exit — stay legal (mask never empties)
TRADE_CANCEL  = 285                        ┘
NUM_ACTIONS = 286
```

Confirmed by import: `ComposeCapper(cap=2)` after 3× ADD_WANT → OPEN(278)=masked,
ADD_WANT(273)=masked, **CANCEL(285)/ACCEPT(279)/DECLINE(280)/CONFIRM(281)=legal,
bank TRADE_BASE(210)=legal.** The cap is correct and matches the learner-side
`FastCatanEnv` bookkeeping (same id range, same ROLL/END_TURN reset). Good.

---

## What's right (keep)

- **`ComposeCapper` extends the cap to opponents (train) + gate + eval** via the
  shared `play_one`. This is the actual bug fix: the old `action_masks()` cap fired
  only for seat 0, so the May-28 sweep gated four *uncapped* policies — exactly the
  50–85% no-winner seen. Per-seat counter, reset on each seat's ROLL/END_TURN,
  reset once per game in `play_2v2`/`eval_seats`/`SelfPlayEnv.reset`. Correct.
- **Symbolic ids** (`_A.TRADE_*`) replace the old 296-era `268:289` literals — no
  longer interface-fragile.
- **`int(...)` coercions** on `opp.act(...)`/`seat_policies[...].act(...)` before
  `step()` — avoids numpy-int leaking into the C++ call. Fine.
- Cap threaded end-to-end and persisted: CLI on train/gate/eval, `_RESUME_KEYS`,
  `summary.json`, sweep `--trade-compose-cap`, `run_arch_sweep.sh CAP=40`.

---

## HIGH

### H1. Raising the cap 20→40 works *against* the metric that fails
`env.py:58` (`MAX_TRADE_COMPOSE_PER_TURN = 40`)

The gate fails on **decidability** (`conclusive` needs ≥50% of games to reach a
winner within `max_steps`), not on win-share — every May-28 cell that *was*
conclusive also passed/was strong. `play_one` counts **one step per seat-action**
(`eval_seats.py:51,67`) capped at `max_steps=5000`. Budget math, 4 seats:

```
real actions to reach 10 VP ≈ 4 seats × ~25–35 turns × ~5 = ~500–700 total steps
churn headroom at cap C      ≈ 4 seats × ~30 turns × C
   C=40 → up to ~4800 churn steps  → 500 real + 4800 ≈ 5300 > 5000  → no-winner
   C=20 → up to ~2400 churn steps  → fits 5000 with room
```

So at the gate's own budget, **cap=40 is at/over the edge and cap=20 is safer** —
the opposite of the change. The code comment even states "Lower = games terminate
faster (more likely decidable)," then raises it anyway, betting the `-2` reward
will keep policies away from the cap. But see H2: the reward does nothing in the
gate. Recommendation: keep the cap **low for decidability** (≤20; 10 still allows
~2 legit offers/turn) and/or make it a sweep axis, rather than 40.

### H2. `TIE_REWARD = -2` only shapes the learner; the gate is frozen-vs-frozen
`env.py:34,181,245`

The `-2` is a *training* signal. The gate (`play_2v2`) and per-seat eval load
**frozen** `MaskablePPO` snapshots and only call `.predict()` — no learning, no
reward. So whatever churn the frozen policies (especially the older `nago`
snapshot, and the pool snapshots during training) still exhibit under *sampling*
is unaffected by `-2`. Decidability in the gate is therefore set by the **cap +
step budget** (H1, H3), not by the reward. Treating `-2` as "the primary
anti-churn lever" (per the code comments / `run_arch_sweep.sh`) is the core
misconception: it cannot rescue a frozen-vs-frozen gate. Keep the cap as the
mechanical guarantee.

### H3. Gate/eval budget is ~3× *stricter* than training truncation (unit mismatch)
`env.py:67,242` (train) vs `gate.py:44`, `eval_seats.py:78` (gate/eval)

- Training truncates at `MAX_EPISODE_STEPS = 4000` **learner steps** — `_ep_steps`
  increments once per `FastCatanEnv.step` (seat-0 action only); opponents step in
  `_step_opponents` **without** incrementing it. So 4000 learner steps ≈ **≥16 000
  total** seat-actions (×4+, more with opponent churn).
- The gate/eval cap `max_steps = 5000` counts **every** seat-action.

A policy that comfortably closes out inside training's budget (~16k total) can
still hit the **5000-total** wall in the gate → scored no-winner. This inflates
gate no-winner *independently of the cap*, and the just-raised `5000` is still far
below training's effective horizon. Put both budgets in the **same unit** and make
the gate at least as lenient as training (e.g. count total steps in training too,
or raise the gate cap toward the training-equivalent total). Otherwise the gate
under-reports decidability no matter what the cap is.

---

## MEDIUM

### M1. `-2` terminal vs `gamma=0.999` over a ~4000-step horizon — weak, and variance-raising
`train_selfplay.py:200` (`gamma=0.999`), `env.py:34`

`0.999^4000 ≈ 0.018`, so a terminal `-2` is worth ~`-0.04` discounted at the
early/мid-turn churn actions you want to discourage — a faint gradient for "stop
churning." Meanwhile in 4-player self-play the learner loses ~75% of decided games
by symmetry; with win`=+1`/loss`=-1`/stall`=-2` the value target sits well below
zero and stall episodes dominate the return scale, raising critic variance. If the
intent is "don't stall," a **small per-step penalty on compose churn** (dense,
barely discounted) or a milder terminal (`-1.5`) is likely both stronger and
calmer than a big, heavily-discounted terminal. At minimum, watch
`train/value_loss` and `explained_variance` across the sweep.

### M2. Two parallel cap mechanisms must be hand-synced
`env.py` (`FastCatanEnv._compose_count` in `step()`/`action_masks()`) vs
`ComposeCapper`

Seat 0 is capped by `_compose_count`; seats 1–3 by a `ComposeCapper`. Today they
agree (same `TRADE_COMPOSE_IDS`, same ROLL/END_TURN reset), and I verified both.
But they're duplicated logic in two places — a future tweak to one (range, reset
rule, or "count OPEN or not") silently desyncs learner vs opponents. Consider
making `FastCatanEnv` drive seat 0 through a `ComposeCapper` too (single source of
truth), or extract the reset/count rule into one function both call.

### M3. Module-constant levers aren't recorded in run provenance
`env.py:34,67` (`TIE_REWARD`, `MAX_EPISODE_STEPS`)

`summary.json`/`run_config.json` record `trade_compose_cap` (good) but not
`TIE_REWARD` or `MAX_EPISODE_STEPS`. Since the whole experiment hinges on the `-2`
reward and the 4000 cap, a future change to either leaves old runs
indistinguishable. Either log them in `summary["config"]`, or (better, given they
are now experimental knobs) promote them to CLI args alongside `--trade-compose-cap`.

---

## LOW / docs

- **L1. Stale help text.** `sweep.py:63` says `--trade-compose-cap` "Omit =
  train_selfplay default (20)", but `MAX_TRADE_COMPOSE_PER_TURN` is now **40**.
  (`run_arch_sweep.sh` passes `40` explicitly, so the run is fine — the text lies.)
- **L2. `models/PLAN.md` not updated to match this fix.** Still says
  "`MAX_TRADE_COMPOSE_PER_TURN=20` … ids **268–288**" and "`MAX_EPISODE_STEPS=3000`".
  Now 40 / **268–278** / 4000. (`models/selfplay/PLAN.md` *is* updated.) The
  `268–288` range was always wrong for the 286 build; new code is correct, but the
  doc still shows the old literal.
- **L3. Sweep doesn't grid the new levers.** `sweep.py` grids
  lr×ent×interval×arch×sched×kl but not `trade_compose_cap` (single forwarded
  value) nor the reward — so the two changes most likely to move decidability can't
  be A/B'd in one sweep. Given H1/H2, adding a small cap axis (e.g. `10 20 40`)
  would directly answer "what cap actually makes the gate conclusive."
- **L4. `ComposeCapper.update` ignores actions outside its two cases.** Correct by
  design (build/dev/robber/bank don't touch the budget), just note it leaves the
  count *non-monotone-safe* only if a reset action is ever added without updating
  the `==`/range checks — see M2.

---

## Suggested validation order (before re-launching the 12-cell sweep)

1. **Fix the budget unit mismatch (H3)** — without it the gate under-counts
   decided games regardless of the cap.
2. **1-cell smoke**, trades ON, and read `no_winner_rate` from the gate **at
   cap=10/20/40**. Pick the largest cap that holds `no_winner_rate < 0.5` (ideally
   ≪0.5) — that is the empirical answer H1 is arguing about. Don't assume 40.
3. Only then run the full sweep. The `-2` reward (H2/M1) is an orthogonal training
   experiment — keep it if it improves *win-share among decided games*, but don't
   count on it for decidability.
4. The May-28 sweep ckpts can't be re-gated fairly (trained under `-1` + uncapped
   opponents) — a fresh run is required, as `models/selfplay/PLAN.md` already notes.
