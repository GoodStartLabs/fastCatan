# P3-X2 (Creative) — Report: trade-competent, belief-informed mixed-expert distillation

**Track:** Creative · **Cycle:** Phase-3 x2 · **Branch:** `phase3-creat-x1` (X1+X2 tooling)
**Base build teacher:** native AB-d1 (same as incumbent `il_best`, so win-rate deltas isolate trading)
**W&B:** goodsettler-il group `p3-creat-x1` (runs `p3x2-plain`, `p3x2-legal`), goodsettler-eval group `p3-creat-x2`

## Verdict

**Trade competence is now a demonstrated, coherent capability — a first for this project — but it does not (yet) raise win rate.** The distilled net went from the incumbent's `opens_per_game` **0.0 → 17.0** (matching the persona teacher's 16.7) at a coherence of **0.832** (vs the persona's 0.911), while overall win rate is **statistically unchanged** (13.5% [10.5, 17.2] vs incumbent 15.75% [12.5, 19.6], same anchor set/seeds — CIs overlap). Pre-registered falsifier #1 (opens stays ~0) is **not** triggered; falsifier #2 (win rate rises) **is** — trading was learned but did not convert to wins at this budget. This establishes the trade-competent net as the **platform** the trading-curriculum north-star needs (that work was blocked because no self-contained net traded; now one does, coherently).

## Method

Mixed-expert self-play generator `models/belief/gen_mixed.py` (10k trades-on games, 5.25M decisions, 52.8% persona-driven, zero no-winners). Per decision, the teacher is chosen by domain:
- **robber (MOVE_ROBBER / STEAL) + all p2p-trade decisions → `balanced-strong` persona** (leader-blocking robber + coherent targeted trading);
- **main-phase build/dev/bank/end, discard, initial placement, YoP/Mono → native AB-d1** (same build teacher as `il_best`). In main phase the persona is consulted first; if it wants to trade, that is taken, else AB's build/end.

IL-trained the split-MLP (param-identical 6,704,023) on this data; exported to a tournament-loadable checkpoint; evaluated with `models/analysis/action_profile.py` (now with the incoherence probe) over 400 trades-on games vs the 5-agent anchor set.

## Results

| metric | incumbent `il_best` | **X2 mixed net** | `balanced-strong` (ref) |
|---|---:|---:|---:|
| overall win rate (anchor set) | 15.75% [12.5,19.6] | **13.5% [10.5,17.2]** | 21.25% |
| opens_per_game | **0.0** | **17.0** | 16.7 |
| trade confirm_rate | — | 0.122 | 0.217 |
| steal_on_leader | 0.46 | **0.52** | 0.805 |
| **trade_to_build_rate (coherence)** | n/a | **0.832** | 0.911 |
| vs catanatron-value | 1.25% | **6.25%** | 11.25% |
| vs builder-strong | 18.75% | 22.5% | — |
| vs balanced-strong | 31.25% | 16.25% | — |
| vs oracle-ab-d2 | 2.5% | 0.0% | 21.25% |

Per-opponent: trading **helped** vs `catanatron-value` (5×) and `builder-strong`, but **hurt** vs `balanced-strong` (31→16%) — a strong leader-blocking trader appears to exploit the net's high-volume, low-conversion trading (confirm 0.122: it proposes ~17/game, few succeed). Net effect on aggregate win rate: a wash (statistically unchanged).

**Belief rider:** legal (belief-augmented) top-1 = 0.8121 vs plain 0.8112 (**+0.09pt, flat**) — the same null as X1. The trade/robber teacher (`balanced-strong`) is a *legal-info* persona that doesn't use hidden state, so belief cannot help imitate it. Belief-via-imitation is now falsified twice (X1 oracle teacher, X2 persona teacher); only belief-for-RL-play remains open (deferred).

## Interpretation

- **Success:** the mixed-expert distillation teaches a genuinely new behavior (coherent p2p trading) that neither AB nor the incumbent could produce. Coherence 0.832 confirms the trades are mostly spent on builds — the mixed-expert seam introduced only *mild* incoherence (−8pt vs the persona), not plan collapse.
- **Why no win-rate gain:** (1) low trade **conversion** (confirm 0.122 — high proposal volume, most rejected, wasting turns); (2) mild incoherence (0.832); (3) trading against a strong leader-blocker (`balanced-strong`) is net-negative — the net trades indiscriminately, including in situations a strong player would refuse; (4) the net imitates the mixed teacher at 0.811 (noisier than `il_best`'s 0.885 AB clone), and the mixed teacher itself isn't clearly stronger than pure AB.

## Results row

| run | teacher | actor input | val_top1 | opens/g | coherence | anchor WR | verdict |
|---|---|---|---:|---:|---:|---:|---|
| p3x2-plain | mixed (AB-d1 build + persona trade/robber) | 1084 | 0.8112 | 17.0 | 0.832 | 13.5% | trade capability gained; WR flat |
| p3x2-legal | mixed | 1084+18 belief | 0.8121 | — | — | — | belief null again (+0.09pt) |

## Next cycle (X3) — trading curriculum from this warm-start (the north-star's primary axis, now unblocked)

The trade-competent net is the platform. Highest-leverage next moves, in order:
1. **Fix trade conversion + partner selection.** The net proposes 17/game at 12% confirm and trades into `balanced-strong`. A DAgger/curriculum pass that (a) suppresses proposals to the VP leader (partner-selection curriculum) and (b) relabels low-value proposals toward "don't trade" should raise conversion and stop the `balanced-strong` bleed — directly targeting the two loss mechanisms above.
2. **Offer-range curriculum** (the substrate's `INITIAL_OFFER_CAP` seam already exists) to teach small, high-EV trades before full-range.
3. **RL fine-tune from the trade-competent warm-start** (deferred belief-for-RL can ride here) — now that opens>0, PPO has a trading policy to improve rather than discover from scratch (the exploration problem that stalled the 2.0 sprint).
Falsifier for X3: partner-restricted / higher-conversion trading still fails to lift anchor-set win rate above the incumbent's CI.
