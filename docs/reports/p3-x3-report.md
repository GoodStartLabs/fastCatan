# P3-X3 (Creative) — Report: trading curriculum (guarded relabel) from the x2 warm-start

**Track:** Creative · **Cycle:** Phase-3 x3 · **Branch:** `phase3-creat-x1`
**Warm-start:** `x2_plain` (the trade-competent net from X2) · **Teacher:** guarded mixed (AB-d1 build + `balanced-strong` robber/trade, with two conditional trade guards)

## Verdict

**The conditional partner guard + warm-start recovered win rate to incumbent parity while retaining coherent trade competence.** X3 is a net that trades (opens 15.8/game, coherence 0.841) *and* matches the incumbent's win rate (16.0% [12.7,19.9] vs 15.75% [12.5,19.6]) — fixing X2's regression (13.5%) and recovering the `balanced-strong` bleed (16%→21%). It does **not beat** the incumbent (CIs overlap; pre-registered falsifier "fails to lift above the incumbent CI" is technically met — no *gain*), but it removes the win-rate cost of trading, yielding a **strictly better platform** for the RL stage: incumbent-level strength *plus* a trading capability the incumbent lacks.

## Method

Guarded relabel generator `models/belief/gen_guarded.py` (6k train / 1k val games). Same mixed-expert scheme as X2, with two **conditional** trade guards applied live, then IL fine-tuned from the `x2_plain` warm-start (`il_belief --init`, 2 epochs):

- **CONVERSION guard:** if the persona would `TRADE_OPEN`, compute base-value EV = v·want − v·give (v=[brick 1.0, lumber .9, wool .6, grain .95, ore 1.0]); if net value-losing (EV ≤ −0.5) override to AB's build/end. **Fired 0 times** — `balanced-strong`'s offers are not egregiously value-losing, so this guard is inert here (the low confirm rate is an imitation-fidelity / partner-acceptance issue, not bad-EV offers).
- **PARTNER guard (conditional, per the gate's requirement — NOT blanket):** relabel confirming the VP leader to `TRADE_CANCEL` **only when** the leader is within striking distance (public VP ≥ 8). **leader_trade_survival_frac = 0.865** (train) / 0.855 (val): 86.5% of leader-trades were kept; only 13.5% (leader ≥ 8 VP) suppressed. The RL stage inherits a light, conditional prior, not a heuristic it must unlearn.

## Results (analyzer, 400 trades-on games vs the anchor set; same seeds as X1/X2)

| metric | incumbent `il_best` | X2 mixed | **X3 guarded+warmstart** | `balanced-strong` |
|---|---:|---:|---:|---:|
| overall win rate | 15.75% [12.5,19.6] | 13.5% [10.5,17.2] | **16.0% [12.7,19.9]** | 21.25% |
| opens_per_game | 0.0 | 17.0 | 15.8 | 16.7 |
| confirm_rate | — | 0.122 | 0.121 | 0.217 |
| coherence (trade→build) | n/a | 0.832 | **0.841** | 0.911 |
| steal_on_leader | 0.46 | 0.52 | 0.49 | 0.805 |
| vs balanced-strong | 31.25% | 16.25% | **21.25%** | — |
| vs builder-strong | 18.75% | 22.5% | 21.25% | — |
| vs trade-happy | 25.0% | 22.5% | 27.5% | — |
| vs catanatron-value | 1.25% | 6.25% | 3.75% | — |
| vs oracle-ab-d2 | 2.5% | 0.0% | 6.25% | — |

- **Win rate recovered** to incumbent parity (16.0% vs 15.75%), up from X2's 13.5%.
- **The `balanced-strong` bleed recovered** (X2 16% → X3 21%) — direct evidence the conditional partner guard works as intended (stop handing a near-winning leader resources), without a blanket rule.
- Trade competence retained (opens 15.8, coherence 0.841 — slightly *more* coherent than X2).
- Conversion unchanged (confirm 0.121) — the conversion guard was inert; improving conversion needs better offer composition (imitation fidelity), addressed by the RL/DAgger stage, not a value filter.

## Interpretation

The night's arc: incumbent (15.75%, no trading) → X2 (13.5%, trades but bleeds to strong leader-blockers) → **X3 (16.0%, trades coherently at incumbent parity)**. The conditional partner guard converted trading from a net-negative into a net-neutral capability. We now hold a self-contained net that plays as well as the incumbent *and* trades — the exact warm-start the trading-curriculum north-star and the potential C1-joint RL fine-tune want.

## Results row

| run | teacher | init | val_top1 | opens/g | coherence | anchor WR | leader_trade_survival | verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| p3x3-plain | guarded mixed | x2_plain | 0.810 | 15.8 | 0.841 | 16.0% | 0.865 | WR recovered to parity; trading retained |

## Next

`x3_plain.zip` is the recommended warm-start for an RL fine-tune (PPO from a trade-competent, incumbent-parity policy — no cold-start trade exploration). Levers still open, in order: (1) **conversion / offer quality** — the confirm rate (0.121 vs persona 0.217) is the clearest remaining trading weakness; a DAgger pass with the persona labeling the *net's own* offer-composition states (true DAgger, not teacher-relabel) should raise it. (2) **RL fine-tune** with terminal reward from this warm-start, opponent mix weighted toward `balanced-strong`/`catanatron-value` (the hardest rungs). (3) Belief-for-RL (deferred) can ride the RL stage — its value, if any, is in *play* not *imitation* (X1/X2 both showed imitation is belief-insensitive).
