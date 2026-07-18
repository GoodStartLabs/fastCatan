# P3-X4 (Creative) — Report: TRUE DAgger for trade conversion (FALSIFIED)

**Track:** Creative · **Cycle:** Phase-3 x4 · **Branch:** `phase3-creat-x1`
**Warm-start:** `x3_plain` · **Teacher (labels):** guarded mixed (AB-d1 build + `balanced-strong` robber/trade + X3 conditional partner guard)

## Verdict — FALSIFIED

Pre-registered falsifier (stated before running): *confirm_rate fails to reach ≥0.17 OR anchor-set win rate drops below X3's CI (12.7%).* **The second condition is triggered.** confirm_rate *did* reach 0.172, but win rate **crashed to 7.75% [5.5, 10.8]** — far below X3's 16.0% [12.7, 19.9]. **`x4_plain` is net-harmful and must not be promoted or used as a warm-start; `x3_plain` remains the platform.**

The confirm_rate "improvement" is a **mirage**: the net did not learn to compose *better* offers — it learned to trade *far less*. `opens_per_game` collapsed 15.8 → 2.8 and coherence fell 0.841 → 0.736. One-round teacher-relabel DAgger on the net's own trade-churn states over-corrected toward trade *abandonment*.

## Method

TRUE DAgger (`models/belief/gen_dagger.py`, 4k train / 500 val games, 531k recorded learner states, leader_trade_survival 0.877): the **student (`x3_plain`) plays the learner seat** — generating its own (often awkward) offer-composition states — while the guarded mixed **teacher labels** what to do from them; opponents are the guarded mixed teacher. Fine-tuned from the `x3_plain` warm-start (2 epochs).

## Results (analyzer, 400 trades-on games, same anchor set/seeds)

| metric | X3 (platform) | **X4 (DAgger)** |
|---|---:|---:|
| overall win rate | 16.0% [12.7,19.9] | **7.75% [5.5,10.8]** |
| opens_per_game | 15.8 | 2.8 |
| **confirm_rate** | 0.121 | **0.172** |
| coherence (trade→build) | 0.841 | 0.736 |
| vs balanced-strong | 21.25% | 15.0% |
| vs catanatron-value | 3.75% | 0.0% |
| vs oracle-ab-d2 | 6.25% | 0.0% |

Every per-opponent rate fell; the net both trades less and plays worse overall.

## Why it failed

The net's own state distribution is dominated by trade-churn states (it opens ~16/game, many mid-composition). When the persona teacher labels those *awkward, net-generated* partial-offer states, the correct action is frequently to **abandon** the poorly-composed trade (cancel / build instead) rather than to salvage it. Imitating that signal teaches the net to **avoid entering trade states at all** — dropping opens to 2.8 — and 2 epochs of fine-tuning on this skewed distribution also perturbed the build policy, crashing win rate. Higher confirm_rate is an artifact of proposing only a few (safer) trades, not of better offers.

## Conclusion & next

- **`x3_plain` stays the designated platform / joint-run warm-start.** `x4_plain` is a recorded negative (kept on-branch, marked not-for-promotion).
- **Trade conversion is not fixable by one-shot IL-DAgger.** Imitating a relabel on the net's own composition states teaches abandonment, not better composition. Conversion belongs in the **RL stage**, where the reward can directly credit *confirmed* (successful) trades rather than imitate a teacher — recommend the joint anchored-RL run add a small aux/terminal signal for confirmed-trade success, from `x3_plain`.
- If IL-DAgger is retried, it needs (a) multiple aggregation rounds with **WR-gated early stopping** (stop before the policy degrades) and (b) labels restricted to *completable* offer states, not the full churn distribution.

## Results row

| run | teacher | init | opens/g | confirm_rate | coherence | anchor WR | verdict |
|---|---|---|---:|---:|---:|---:|---|
| p3x4-plain | guarded mixed (DAgger, net states) | x3_plain | 2.8 | 0.172 | 0.736 | 7.75% | FALSIFIED — WR below X3 CI; opens collapsed |
