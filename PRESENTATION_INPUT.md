# Slide-Deck Input — Thesis Progress Presentation (8 slides)

Hard facts attached to every bullet so the slides can be built directly. Numbers are
measured and in the thesis record. `AB` = Catanatron's Alpha-Beta; `AB-d2` = depth 2 (the
gate opponent). "Parity" = 25% (4-player random-chance win rate). CIs are 95% Wilson.

Legend in this doc: **❓** = a question you wrote, answered with facts. **⚠️** = verify
against your own logs before claiming.

---

## Slide 1 — Title

- **Title:** *Can a Learned Agent Beat Alpha-Beta at 4-Player Catan?*
- Subtitle: *A high-throughput simulator + a neuro-symbolic MCTS agent*
- Thesis bar (state it here once): **> 25% win rate vs Alpha-Beta over ≥1000 four-player
  games, 95% CI lower bound > 25%** (25% = 4-player chance).
- Name · affiliation · date.

---

## Slide 2 — The baseline: Alpha-Beta with a heuristic evaluation function

- **Opponent / bar to beat:** Catanatron's `AlphaBetaPlayer` — the standard strong Catan
  bot. The thesis target is to beat it with statistical significance.
- **Search:** depth-2 minimax + alpha-beta pruning, **expectimax over chance nodes** (dice
  roll, dev-card draw, robber steal). Full-information.
- **Heuristic evaluation function** (`base_fn`, hand-tuned linear weights): scores a
  position from features — **public victory points (dominant, lexicographic weight ≈3·10¹⁴),
  settlement/city production, board reachability/expansion, longest-road & largest-army
  potential, hand size**.
- **Setting:** 4-player free-for-all — the agent plays **1 seat vs 3 Alpha-Betas**, so
  **25% = chance**, and >25% with significance = genuinely better.
- Why it's hard: full-information lookahead + a strong tuned value function, under
  stochasticity and a ~300-action space.

---

## Slide 3 — The simulator (fastCatan)

- **~7× faster than Catanatron** in 4-player random games (games/s, equal footing).
  C++23 core + nanobind Python bindings. Pure-C++ step **10–50M steps/s**; full RL loop
  **~1M steps/s**; M1 target was 5·10⁵ → exceeded >10×.
- **❓ How do I prove the simulator is correct? Are unit tests enough?** → **No — unit
  tests only check cases you thought of.** The strategy is a **differential oracle**:
  - **Cross-engine differential:** co-step fastCatan **and** Catanatron on the *same* action
    stream and assert **full state + observation parity every single ply**. This caught &
    fixed **5 real rule bugs** unit tests missed (longest-road off-by-one, bank-shortage
    resource yields, road-through-enemy, history-dependent longest-road, obs trade-response).
  - **10⁷-game invariant fuzz: 0 violations over 4.04·10¹⁰ steps** (resource conservation,
    hand/VP/piece bounds, mask legality, terminal correctness).
  - **Deterministic perft hash** (fixed seed → fixed trajectory hash).
  - The native Alpha-Beta port matches Catanatron's value function to **1.9·10⁻¹⁶**.
  - Takeaway line: *"Tests verify what I expect; the differential oracle verifies against
    ground truth I didn't think of."*

---

## Slide 4 — Observation & action space

- **Observation: 1084 floats**, encoded from the **current player's perspective**
  (perspective-flipped, so one network plays any seat). Count features normalized by
  structural Catan maxima. Encoder is frozen and **bit-parity verified** against the engine.
- **Action space: flat Discrete, 286 used actions / 320-bit mask.** Breakdown:
  - roll dice, end turn
  - build: **54 settlements + 54 cities + 72 roads**
  - dev cards: buy + play {knight, year-of-plenty, road-building, monopoly}
  - robber: **19 hexes** + steal-target
  - discard sub-phase (shed-on-7)
  - **trade sub-phase — compositional** (add-give / add-want / open / accept / decline /
    confirm / cancel): composes a trade instead of enumerating all resource combinations.
- **Incremental legal-action mask:** updated per move (no board re-scan), the key to
  throughput; debug builds assert `incremental == recomputed` every step.

---

## Slide 5 — First try: PPO (reactive policy)

- **MaskablePPO** (action-masked PPO), reactive: state → action, no lookahead.
- **After 50M steps: 95.5% win rate vs random** (native eval) / 89.5% via the Catanatron
  bridge → **M2 gate (>90% vs random) MET.**
- Config: **768 parallel environments, 50M steps, sparse ±1 reward** (run `ppo_1084_50m`).
- **❓ Hyperparameter optimization vs random — 50M games each, best chosen?** Accurate
  version: the large grid was the **self-play** sweep (lr × entropy × snapshot-interval ×
  architecture × lr-schedule × target-KL); vs-random was used as an anchor metric. The
  **95.5% headline is the single 50M-step gate run**, not "every sweep cell ran 50M vs
  random." **⚠️ Confirm the exact per-cell budget against your sweep logs before stating
  it on the slide.**
- **Key caveat (sets up slide 6):** crushing random ≠ any progress vs Alpha-Beta.

---

## Slide 6 — Self-play

- Iterative self-play + **PFSP league** (pool of frozen past-self snapshots).
- **After ~200M self-play steps: 86.7% vs random**; self-play gate PASS (latest beats its
  100M-step-ago self **66%** in balanced 2-vs-2, where 50% = neutral).
- **vs Alpha-Beta: 0/200 = 0%** — **no change** from the pre-self-play model.
- Not a one-off — the wall held across every reactive lever:
  - more capacity (512→2048 nets): vs-random ↑ **98.5%**, vs-AB **flat 0**
  - reward shaping: ≈ 0
  - PPO trained **directly** vs Alpha-Beta: **0/500**
- **Conclusion:** reactive-RL gains are **orthogonal to minimax** — self-play never
  generates Alpha-Beta's value-greedy lines, so the policy is out-of-distribution against it.

---

## Slide 7 — AlphaZero / MCTS approach

- **AlphaZero-style MCTS over the exact simulator** (snapshot / restore / reseed for the
  stochastic chance nodes); the network is the prior. Full-information search — fair, since
  Alpha-Beta is full-information too. ≥**512 stochastic sims/move**.
- **❓ Policy network trained on?** → **imitation of Alpha-Beta**: **160k Alpha-Beta-vs-
  Alpha-Beta games**, masked cross-entropy on the teacher's move. *Distribution beats
  optimization:* this clone reaches **0.975 vs random after ~2 min of training** (PPO needed
  50M steps for ~the same).
- **❓ Value network → currently the heuristic is given.** Correct: leaves are evaluated by
  the **symbolic Alpha-Beta heuristic** (two-scale lexicographic squash), **not yet learned**
  — this is the "neuro-symbolic hybrid." (Replacing it = slide 8.)
- **Sims is the scaling axis** (vs AB-d1): 256→23% · 512→28% · 1024→30%.
- **Result: 65/200 = 32.5% win rate, 95% CI [26.4 – 39.3] vs Alpha-Beta-d2** in the
  4-player setting → **CI lower bound 26.4 > 25 = first statistically-significant win.**
  (Native ladders: AB-d1 29.0%, AB-d2 29.5%.)
- *(Optional sub-bullet)* unlocked by one fix: Catanatron **shuffles seating** internally;
  correcting the agent's seat per decision moved the bridge result **~6% → 32.5%**.

---

## Slide 8 — Replacing the Catanatron heuristic (the de-catanatronization)

- **Why:** the 32.5% agent still **calls Catanatron at inference** (heuristic leaf value +
  a copy of Alpha-Beta as its in-tree opponent model). To claim *a learned agent* beat
  Alpha-Beta, the agent must be **self-contained** — Catanatron only for training data +
  the final exam.
- **Three stages (now complete):**
  - **Stage 1 — learn the leaf value** (distill the heuristic into the value head):
    **20.0% [15.1–26.1].** Fidelity-to-heuristic ↔ win rate is ~linear, then plateaus.
  - **Stage 2 — learn the in-tree opponent** (drop the Alpha-Beta copy) → **fully
    self-contained: 18.0% [13.3–23.9].**
  - **Stage 3 — train the value on stronger-play / search-improved targets: 17.5%
    [12.9–23.4] — NULL.** The target was fit well (value-MSE 0.014, teacher top-1 0.90) but
    **win rate did not move.**
- **Finding (the result of this half):** the self-contained learned agent **saturates at
  ~17%** — below parity (25%) and below the hybrid (29.5%). The cap is now pinned to an
  **information limit**: the per-player observation **cannot see hidden enemy state**
  (unrevealed dev cards / hands) that the Alpha-Beta heuristic reads, so a
  **partial-information learner cannot match a full-information judge.** Confirmed
  irreducible — unchanged by more data, more sims, and better targets (all three tested).
- **One-line takeaway:** *the heuristic's edge is information, not just computation — the
  hybrid keeps it (32.5%), the perspective-pure learned agent is information-bounded (~17%).*

---

# Suggested visuals (2 charts carry slides 6–8)

**Chart A — Win rate vs Alpha-Beta-d2 (use on slide 6 or 7).** Bars + 95% CI, dashed line
at 25% "parity":
- PPO reactive: 0% · Self-play 200M: 0% · AlphaZero pure self-play: ~0%
- **Hybrid MCTS (heuristic leaves): 32.5% [26.4–39.3] — green, above the line**

**Chart B — De-catanatronization ledger (slide 8).** Descending bars + 25% parity line:
- Hybrid (uses Catanatron): 29.5%
- + learned leaf value: 20.0%
- + learned opponent (self-contained): 18.0%
- + stronger-play value targets: 17.5%
- Annotate the 29.5→17 drop as "cost of removing the heuristic = information cap".

---

# Data appendix (exact numbers for chart rendering — don't fabricate beyond this)

- **Win rate vs AB-d2** (512 sims, 200g unless noted): PPO 0/200 · self-play 0/200 ·
  AZ pure self-play ≈0 · **hybrid 65/200 = 32.5% [26.4–39.3]** · self-contained best 36/200
  = 18.0% [13.3–23.9].
- **Hybrid native ladders:** AB-d1 29.0% [25.5–32.8] (≥512 sims, 600g); AB-d2 29.5%
  [23.6–36.2] (512 sims).
- **Sims scaling (hybrid vs AB-d1):** 256→23.0%, 512→28.25%, 1024→30.5%.
- **De-cat stages vs AB-d2 (512 sims, 200g):** stage1 20.0% [15.1–26.1] · stage2 (self-
  contained) 18.0% [13.3–23.9] · stage3 17.5% [12.9–23.4] · hybrid ref 29.5% · parity 25%.
- **Stage-1 fidelity↔wins:** symbolic ρ1.00→29.5% · two-scale ρ0.83→16.5% · naive ρ0.71→
  9.5% · outcome-head ρ0.44→20.0%.
- **Reactive vs random:** PPO 95.5% native / 89.5% bridge; self-play 86.7%; arch-sweep up to
  98.5% — all with **0 vs Alpha-Beta**.
- **Distribution beats optimization:** Alpha-Beta-clone = 0.975 vs random after ~42s data +
  ~78s training.
- **Throughput:** ~7× Catanatron games/s; pure-C++ 10–50M steps/s; RL loop ~1M; target 5·10⁵.
- **Correctness:** 10⁷-game fuzz, 0 violations / 4.04·10¹⁰ steps; differential found 5 bugs;
  native AB value matches Catanatron to 1.9·10⁻¹⁶.
- **Obs/actions:** 1084 floats / 286 actions (320-bit mask).
- **PPO config:** MaskablePPO, 768 envs, 50M steps, sparse ±1.

> ⚠️ One claim to verify before presenting: the PPO **hyperparameter-sweep budget** on
> slide 5 (whether each cell ran 50M games vs random). Confirm against your sweep logs;
> the 95.5% figure itself is solid (the `ppo_1084_50m` gate run).
