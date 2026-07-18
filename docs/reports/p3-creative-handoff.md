# Phase-3 Creative Track — Morning Handoff

**Night of 2026-07-18.** Branch `phase3-creat-x1` (from `fdf0684`), pushed to GoodStartLabs/fastCatan, `check_frozen` green throughout. Four cycles, all IL-budget (≤2h each), each pre-registered and falsifiable.

## Night arc (one line each)

- **X1 — belief-state actor features → FALSIFIED.** True hidden opponent state (oracle upper bound) lifts AB imitation ≤0.12pt; AB's actions are near-public-predictable, so observation-augmentation can't close the ceiling via imitation.
- **X2 — trade-competent mixed-expert distillation → CAPABILITY, WR flat.** First self-contained net that trades (opens 0.0→17.0, coherence 0.832); win rate flat (13.5%) as strong leader-blockers exploited indiscriminate trading.
- **X3 — guarded relabel + warm-start → VALIDATED (platform).** Conditional partner guard (leader-trade survival 0.865) recovered WR to incumbent parity (16.0% vs 15.75%) while keeping coherent trading. Produced **`x3_plain`**, the designated platform.
- **X4 — TRUE DAgger for conversion → FALSIFIED.** confirm_rate hit target (0.172) but WR crashed to 7.75% (opens collapsed 15.8→2.8); IL-DAgger over-teaches trade-abandonment on the net's own churn states.

**Champion status:** `il_best` remains the formal incumbent (promotion needs CI-separated superiority; X3 is parity). **`x3_plain` is the designated platform** — a net that plays at incumbent strength *and* trades coherently — and the joint anchored-RL run warm-starts from it.

## What is settled (don't re-derive)

- **Belief-via-imitation is dead** (X1 oracle teacher, X2 legal-info persona teacher — both null). Belief's only untested value is in *play* (RL); see X5 Design B, the definitive close-out test.
- **Trade competence is achievable and coherent via distillation**, but only pays off once trading is made discriminating (partner guard) — trading is win-rate-neutral at parity, not yet a gain.
- **Trade conversion / quality is NOT an IL problem** (X4). It belongs in RL, where the reward can credit *confirmed, net-positive* trades. Program shaping-OFF default means this is a creative cycle (shaping-ON vs shaping-OFF control), not a conservative tweak.
- **The AB teacher (and oracle-ab-d2) never propose p2p trades** — a project-level fact; teaching trade initiation requires a persona/search teacher.

## Reusable assets (committed on branch)

- `models/analysis/action_profile.py` — observability: action-type-by-phase, trade stats, robber targeting, loss attribution, **and the incoherence probe** (`trade_to_build_rate`, opens/game, end-hand hoarding). Drives net/persona/oracle through the frozen seam. This is the standing eval + Goodhart monitor.
- `models/belief/belief_tracker.py` — legal card-counter over the public event stream (rel-L1 0.355 vs ground truth); leakage-clean feature path. Ready for belief-for-RL.
- `models/belief/il_belief.py` — parameterized actor (plain/oracle48/legal) + `--init` warm-start.
- `models/belief/{gen_mixed,gen_guarded,gen_dagger}.py` — mixed-expert, guarded-relabel, and TRUE-DAgger generators (persona/AB teachers, belief recording, conditional trade guards).
- Checkpoints: `x3_plain.zip` (platform), `x2_plain.zip`, `x4_plain.zip` (recorded negatives).
- Reports: `docs/reports/p3-x1..x4-report.md`; proposal `docs/proposals/p3-x1-creative.md`; next-cycle draft `docs/proposals/p3-x5-draft.md`.

## Ranked next levers (for gating)

1. **Trade-success aux reward, RL from `x3_plain`** (X5 Design A) — shaping-ON vs shaping-OFF control, with pre-registered Goodhart tripwires (opens-spam / coherence-collapse / hoarding, all monitored by `action_profile.py`). Directly attacks the unsolved conversion/quality gap. **Recommended X5.**
2. **Belief-for-RL** (X5 Design B) — add belief features to the actor, PPO from warm-start; falsifier closes the belief thread for good. Parallel run or next cycle.
3. **Opponent-mix / offer-range curriculum** — the north-star's named axis; weight sampling toward `balanced-strong`/`catanatron-value`, stage the `INITIAL_OFFER_CAP`. Cheapest, conservative-friendly.

## One caution for the RL stage

The 2.0 PPO sprint was unstable in a short budget (reward drifted −0.36→−0.92). Warm-starting from `x3_plain` (a competent, trade-capable policy) is exactly meant to dodge that cold-start; still, gate RL runs on the three Goodhart metrics and WR-vs-control, not on a single training curve.
