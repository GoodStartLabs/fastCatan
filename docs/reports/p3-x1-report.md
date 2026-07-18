# P3-X1 (Creative) — Report: belief-state actor features

**Track:** Creative · **Cycle:** Phase-3 x1 · **Branch:** `phase3-creat-x1` (from `fdf0684`)
**Budget:** IL-first, executed inside the 2 h window · **W&B:** goodsettler-il group `p3-creat-x1`, goodsettler-eval group `p3-creat-x1-profiles`

## Verdict

**Belief/card-counting features on the actor, validated via imitation, are FALSIFIED — cleanly, at the upper bound.** Giving the actor the **true 48-float hidden opponent state** lifts top-1 imitation of the AlphaBeta teacher by **+0.02pt vs AB-d1 (0.8816→0.8818)** and **+0.12pt vs AB-d2 (0.8025→0.8037)** — both far below the pre-registered +1.0pt falsifier. Because the true appendix strictly dominates any legal belief tracker, my legal composition features are deductively foreclosed on this metric and were not run. The information ceiling is **not** recoverable by observation-augmentation of an imitation target.

This does not disprove DR-001's legality argument (belief IS legally inferable — the tracker below proves it). It shows something sharper and more useful: **for imitation learning, the teacher's *action* is near-public-predictable, so more observation does not help.** The incumbent's real, exploitable weaknesses are **behavioral, not informational** (see profile), which redirects the program to the trading-first X2.

## 1. Observability deliverable (method a) — `models/analysis/action_profile.py`

Committed on the branch (permanent asset). Drives the net, ladder personas, or oracles through the frozen `Player.act(env, mask)` seam with `ladder.match`'s seat/board discipline, logging action-type-by-phase, trade stats, robber targeting, and loss attribution to W&B `goodsettler-eval`. Profiles for `il_best.pt`, `balanced-strong`, `oracle-ab-d2` (400 trades-on games each vs a 5-agent anchor set):

| Subject | Overall WR | opens/game | steal-on-leader | vs catanatron-value | vs oracle-ab-d2 |
|---|---:|---:|---:|---:|---:|
| `il_best.pt` (incumbent net) | 15.75% | **0.0** | 0.46 | **1.25%** | 2.5% |
| `balanced-strong` (legal-info persona) | 21.25% | 17.5 | **0.805** | 11.25% | 21.25% |
| `oracle-ab-d2` (full-info search) | 42.25% | 0.0 | 0.41 | 35% | 25% |

Incumbent's three exploitable weaknesses (method d):
1. **Below-parity vs the strong anchor set (15.75%);** crushed by `catanatron-value` (1.25%), a *search-free* full-info greedy heuristic.
2. **Robber targeting ≈ random w.r.t. VP:** `steal_on_leader_rate` 0.461, victim-VP-rank near-uniform over 690 steals.
3. **Never proposes p2p trades** (`opens_per_game` 0.0), inherited from its AB teacher — and **`oracle-ab-d2` is also 0.0**: the whole AB-lineage is trade-mute as a proposer. Only the personas (e.g. `balanced-strong` 17.5 opens/game, robs the leader 0.805) trade and threat-assess.

**Project-level fact (flagged): the AB teacher never proposes p2p trades, so any AB-distilled net inherits a total trade-initiation hole. Teaching trade initiation requires a different (persona/search) teacher — this is exactly X2.**

## 2. The belief experiments (method: IL-first, the reliable lever)

All runs reuse the baseline split-MLP (actor 1084→2048→1024→512→286, param-identical control 6,704,023), the baseline IL optimizer/schedule, and board-disjoint train/held-out. Only the **actor input** varies: `plain` = 1084 (public POV); `oracle48` = 1084 + the true 48-float hidden appendix (a non-deployable UPPER-BOUND diagnostic). Script: `models/belief/il_belief.py`.

| Teacher | data | plain val_top1 | oracle48 val_top1 | Δ |
|---|---|---:|---:|---:|
| AB-d1 (ValueFunction, greedy) | baseline 25.4M-sample cache | 0.8851 (3 ep) | 0.8818 (2 ep)* | ≈ 0 |
| **AB-d2 (depth-2 expectimax, reads hidden state)** | 18k games, 4.0M samples | **0.8025** | **0.8037** | **+0.0012** |

\*AB-d1 epoch-matched: plain 0.8816 vs oracle48 0.8818 at epoch 2.

Sanity checks: (i) `plain` on AB-d1 reproduces the incumbent (0.870 ep1 / 0.882 ep2, matching the 2.0 baseline's 0.871/0.881) — the pipeline is faithful. (ii) The appendix is genuinely informative in the data (17% nonzero, per-resource column means 0.04–0.08, std 0.047) — `oracle48` had real hidden state and still did not benefit.

**Why AB-d1 is information-insensitive:** AB-d1 = greedy `base_fn`, decided from public features, so hidden state cannot help predict its action. **Why AB-d2 (which *does* read hidden state via `ab_value`) is still ~flat:** its chosen *action* remains near-public-predictable — knowing opponents' exact cards rarely flips AB-d2's top-1 action, or the net cannot exploit it for imitation. Either way, **observation-augmentation cannot recover the ceiling through imitation.**

## 3. Belief tracker deliverable — `models/belief/belief_tracker.py` (built + validated, reusable for X2)

A legal per-seat card-counter over the **public event stream** (board parsed from obs; production on rolls, fixed build/dev costs, exact discards and bank-trade give-counts via public hand-size deltas, monopoly/YoP reveals, robber-blocked production; steals kept as distributions; totals reconciled to public hand size). **Leakage contract (structural):** the feature path reads only public actions/dice/board + own private hand; it never reads `player_resource(opp)` or the appendix (that appears only in the offline `validate_error` diagnostic).

**Validated accuracy vs ground truth** (`env.player_resource`, 150 AB-d1 games, p2p-banned regime): mean L1 error **2.14 cards/seat** out of ~6.0 → **relative L1 error 0.355** (captures ~65% of composition). The `MOVE_ROBBER` position fix cut error 0.47→0.355. A companion generator `models/belief/gen_belief.py` records POV-relative belief features per decision (with correct discard-seat routing) — both ready for X2.

(The `legal` IL run was not executed: oracle48 ≥ any legal belief in information, and oracle48 ≈ plain, so legal ≈ plain is entailed for these composition features. Budget was preserved for the X2 setup the evidence points to.)

## 4. Scope / what this does and does not show

- **Shows:** belief-as-actor-input does not improve **imitation** of AB (the IL-first validation path we chose because the 2.0 PPO sprint was unstable). The incumbent's exploitable gaps are **behavioral** (no trade initiation; random robber), not encoding.
- **Does not show:** that belief is useless for **RL play strength**. IL imitation fidelity ≠ outplaying the teacher; an RL agent *might* use belief to pick better-than-AB actions where hidden info changes the outcome. That is a longer, higher-variance, PPO-dependent bet — deprioritized relative to the concrete trading gap.

## 5. Results row

| run | branch | teacher | actor input | val_top1 | verdict |
|---|---|---|---|---:|---|
| p3x1-d1-plain | phase3-creat-x1 | AB-d1 | 1084 | 0.8851 | control (reproduces incumbent) |
| p3x1-d1-oracle48 | phase3-creat-x1 | AB-d1 | 1084+48 | 0.8818 | oracle flat → info-insensitive teacher |
| p3x1-d2-plain | phase3-creat-x1 | AB-d2 | 1084 | 0.8025 | control |
| p3x1-d2-oracle48 | phase3-creat-x1 | AB-d2 | 1084+48 | 0.8037 | **+0.12pt → FALSIFY belief-via-IL** |

## 6. Next cycle (X2, already locked) — now doubly motivated

**Trade-competent, belief-informed mixed-expert distillation.** The analyzer proves the incumbent's exploitable weakness is trade initiation (opens 0.0) and robber targeting — behavioral holes the AB teacher cannot fill. Distill p2p compose/open/respond from the trade-competent personas (`balanced-strong` opens 17.5/game and refuses/accepts sanely; `trade-happy`) while keeping AB for build/robber, and feed the belief tracker's features to the **trade responder/robber heads specifically** (where hidden composition changes the *decision*, unlike AB-imitation). The tracker + `gen_belief.py` are built and validated for this. Falsifier there: the distilled net's `opens_per_game` stays ~0 or its trades-on win rate vs the persona anchor set does not rise.
