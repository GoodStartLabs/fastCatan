# P3-X5 (Creative) — DRAFT proposal, for gating next session

**Track:** Creative · **Status:** DRAFT (not gated) · **Warm-start:** `x3_plain` (the designated platform)
**Context:** X1–X4 established a trade-competent net at incumbent parity (`x3_plain`: 16.0% anchor WR, opens 15.8, coherence 0.841) and produced two clean negatives — belief-via-imitation (X1, X2) and IL-DAgger conversion (X4). Both point the remaining trading work into **RL**, where the reward can credit *play* and *successful trades* rather than imitation. This draft presents two competing RL designs with pre-registered falsifiers.

---

## Design A (recommended) — Trade-success auxiliary reward (shaping-ON vs shaping-OFF control)

**One coherent idea:** from the `x3_plain` warm-start, PPO with the program-default terminal categorical reward **plus a small auxiliary shaping term that credits *confirmed, net-positive* p2p trades**. This is the direct, principled home for the "conversion belongs in RL" lesson X4 proved: a reward that rewards *successful* trades (not imitation of a relabel) should raise trade quality without the abandonment collapse IL-DAgger produced.

**Why RL, not more IL:** X4 showed imitating a relabel on the net's own composition states teaches trade-*avoidance*. RL closes the loop — the agent proposes, sees whether the trade completes and whether it helped, and is reinforced accordingly. Program principle: shaping is OFF by default, so this must be run as **shaping-ON as the one idea vs a shaping-OFF control** (same warm-start, same budget), never smuggled into a conservative run.

**Aux-reward spec (to pin at gate):** on a *confirmed* p2p trade, `+β · clip(Δposition)` where Δposition is a bounded, self-only estimate of the trade's value (e.g. did it enable a build this turn / reduce cheapest-build gap), `β` small and decaying. Credit only net-positive completed trades — never raw open/confirm count.

**Anti-Goodhart verification (required, monitored live):** the failure mode is trade-*spam* — farming the aux term with trivial/rapid trades. Monitor during training and in eval: `opens_per_game`, `confirm_rate`, and the incoherence probe `trade_to_build_rate`. **Goodhart tripwires (pre-registered):** reject if `opens_per_game` rises materially above the persona's ~17 without a CI-separated win-rate gain, OR if `trade_to_build_rate` falls below X3's 0.841, OR if end-of-game unspent hand rises (hoarding). These are checked automatically by `action_profile.py` (already emits all three).

**Predicted slices to improve:** anchor-set WR above X3's CI (>19.9% upper, i.e. a real gain), `confirm_rate` toward the persona's 0.217 *via better offers* (not fewer), WR vs `catanatron-value`/`oracle-ab-d2` (the hardest rungs) up.

**Falsifier (pre-registered):** shaping-ON fails to beat the shaping-OFF control on anchor-set win rate with CI separation, **OR** any Goodhart tripwire fires. Either outcome rejects the aux reward and returns trading strength to the opponent-mix / offer-range curriculum levers.

---

## Design B (alternative / possible parallel) — Belief-for-RL

**One coherent idea:** add the legal belief-tracker features (`models/belief/belief_tracker.py`) to the **actor** and train with **PPO** from a warm-start. X1/X2 falsified belief for *imitation*; its only remaining possible value is in *play* — an RL agent might use inferred opponent composition to make better robber and trade-*response* decisions where hidden state changes the outcome, in ways imitation of a hidden-state-blind teacher never rewarded.

**Design notes:** the belief actor has a wider input (1084 + 18), so the `x3_plain` warm-start transfers only the 1084 slice (belief-input weights fresh) — expect it to need more steps; run at equal budget vs a plain-actor RL control from the same warm-start. Belief features must pass the event-stream leakage referee before this runs (structural argument exists; a decision-invariance test is the remaining gate).

**Falsifier (pre-registered):** belief-augmented RL ≤ plain-actor RL on anchor-set win rate at equal budget (no CI separation). This is the decisive test that **closes the belief thread**: belief would then have shown no value for imitation *or* play at this scale, and should be dropped from the research surface.

---

## Recommendation

Run **Design A** as X5 (it directly attacks the demonstrated, unsolved conversion/trade-quality gap and has the cleanest control). Hold **Design B** as either a parallel competing run (if compute allows two RL runs) or the following cycle — it is the definitive close-out test for belief, worth doing once to retire the thread. Both are RL and both depend on a stable PPO loop from `x3_plain`; if the joint anchored-RL run validates that loop tonight, X5 inherits a proven training path.

**Budget:** RL, so longer than the IL cycles — propose 10M learner decisions or 2h wall, ≤3 env procs, from `x3_plain`, opponent mix weighted toward `balanced-strong` + `catanatron-value`. Every run (incl. the shaping-OFF control) gets a results row and the three Goodhart metrics logged.
