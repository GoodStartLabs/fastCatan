# P3-X1 (Creative) — Belief-state actor features: card-counting to break the encoding ceiling

**Track:** Creative · **Cycle:** Phase-3 x1 · **Status:** PROPOSED (awaiting gate)
**Base:** `phase2-baseline @ fdf0684` · **Exec branch (on approval):** `phase3-creat-x1`
**Incumbent:** `il_best.pt` (IL from native AB-d1; top-1 0.881)

---

## The one idea

Add a per-seat, leakage-clean **belief tracker over hidden opponent state**
(resource composition, dev-cards-by-type, hidden-VP) computed **only from the
public event stream** — card counting — and feed it as extra **actor**
observation features. Validate **IL-first**: does a belief-augmented actor
imitate the full-information native-AB teacher measurably better than the plain
1084-obs actor, and does that lift translate to ladder win-rate on the
trades-on legal-information anchor set. This is DR-001 route (a) — the program's
own #1-ranked route to G2/G3 — executed with the **reliable** lever (imitation /
distribution matching) rather than the **currently unstable** one (short PPO).

## Why now — evidence from the incumbent (analyzer, methods a + d)

Built `models/analysis/action_profile.py` (this cycle's observability
deliverable) and profiled `il_best.pt` over **400 trades-on games** vs a
5-agent legal-info + oracle anchor set. The incumbent's three biggest
exploitable weaknesses, with numbers:

1. **The information deficit is decisive, and it is not about search depth.**
   Overall win rate **15.75%** [12.5, 19.6] — below the 25% four-player parity
   line. Crucially: **1.25% vs `catanatron-value`** and **2.5% vs
   `oracle-ab-d2`**. `catanatron-value` is a *search-free* greedy full-state
   value player (base_fn, no lookahead). A search-free full-information heuristic
   beating our actor ~79:1 isolates **hidden-state blindness of the actor**, not
   search depth, as the dominant gap — the cleanest possible motivation for
   belief features.
2. **Robber targeting is ≈ random with respect to VP.** Over 690 steals,
   `steal_on_leader_rate = 0.461` and the victim-VP-rank distribution is nearly
   uniform (leader 0.27 / 2nd 0.27 / 3rd 0.22 / last 0.24). Expert robber equity
   (rob the leader / the fullest hand) is a belief-dependent decision the actor
   cannot make because it cannot see hands or reliably rank threats.
3. **No trade initiation.** `opens_per_game = 0.0` — the incumbent *never*
   proposes a p2p trade; it only responds (mid-game accept 0.26 / decline 0.09).
   This is inherited faithfully from its AB teacher, which prunes p2p trades —
   and note **`oracle-ab-d2` also opens 0.0**, so the *entire strong side of the
   substrate is trade-mute as a proposer*. (Addressed as the complementary
   trade-treatment idea below; belief features attack #1 and #2 head-on and
   upgrade the trade *responder*.)

Secondary: worst seat is **seat 0 (0.11)** — the first-mover seat, where the
opening is played worst; modal losing VP gap is 5–6 (often blown out, not close).

**The clinching comparison** — same 400-game anchor set, three subjects:

| Subject | Overall WR | opens/game | steal-on-leader | vs catanatron-value | vs oracle-ab-d2 |
|---|---:|---:|---:|---:|---:|
| `il_best.pt` (incumbent net) | 15.75% | **0.0** | 0.46 | **1.25%** | 2.5% |
| `balanced-strong` (legal-info persona) | 21.25% | 17.5 | **0.805** | **11.25%** | 21.25% |
| `oracle-ab-d2` (full-info search) | 42.25% | 0.0 | 0.41 | 35% | 25% |

`balanced-strong` uses **only legal information** but explicit belief/threat
heuristics — and it beats our net overall (21 vs 16%) and by ~**9×** on the
information-heavy `catanatron-value` slice (11.25 vs 1.25%), while robbing the
leader 0.805 vs our 0.46. That is a direct **existence proof that legal belief
features close a large part of the gap our actor is missing** — precisely the
features this proposal encodes. The residual `balanced-strong` → `oracle-ab-d2`
(21 → 42%) is the genuinely-hidden + search component, reserved for the
next-cycle belief-conditioned distillation. (Note `oracle-ab-d2` robs the leader
only 0.41 and opens 0.0: "rob/deny the leader" and "propose trades" are *persona*
skills, not AB's — so the belief-feature target for those slices is the persona,
not AB.)

## Hypothesis

The ~17% ceiling is an **actor-encoding limit, not a legality limit** (DR-001).
The baseline **already** gives the *critic* the full 1132-float observation — an
asymmetric privileged critic (Pinto et al. 2017) — and is still capped. So the
privileged-critic lever is spent; the remaining lever is putting *inferable*
hidden state **into the actor**. Most hidden Catan state is deducible from public
events (production from dice+board, spends from fixed build costs, monopoly/YoP
reveals); only robber-steals, discard mixes, and dev-card identity inject
*bounded* uncertainty (DR-001). A cheap per-seat tracker recovers most of the
48-float hidden appendix **legally**. Given those features, the actor should (a)
imitate the full-information AB teacher better and (b) make the belief-dependent
decisions the analyzer shows are broken.

## Predicted benchmark slices to improve (pre-registered)

- **IL top-1 imitation of the native-AB teacher:** plain-actor 0.881 →
  belief-actor materially higher (predict **≥ +2 pts absolute**, CE down). AB
  conditions its chosen action on full state, so a legal belief the plain actor
  lacks should be directly imitation-relevant — this is a *fast* test of the
  ceiling mechanism.
- **Ladder trades-on win rate**, information-heavy rungs: **vs `catanatron-value`
  1.25% → up** and **vs `oracle-ab-d2` 2.5% → up** (CI-separated); overall
  anchor-set **15.75% → up**; **seat-0 0.11 → up**.
- **Behavioral slices (analyzer):** `steal_on_leader_rate` 0.46 → higher, victim
  VP-rank concentrates on the leader; trade-**responder** EV improves (fewer
  accepts that hand a leader their build resource).

## Falsifying result

If the belief-augmented actor, trained by IL on **identical** teacher data and
equal budget, improves teacher top-1 imitation by **< +1.0 pt** (within noise)
**and** yields no CI-separated win-rate gain vs `catanatron-value`/`oracle-ab-d2`
nor improvement in the belief-sensitive behavioral slices — then the ceiling is
**not** an actor-encoding limit at this budget and belief-as-input is rejected.
That verdict is consistent with the strongest counter-evidence in the literature
(DeepNash, SAD): expert imperfect-information play *without* explicit belief.

## Why this is the creative (higher-variance) bet, not conservative feature-tuning

The literature genuinely splits, so falsification is a real outcome, not a
formality: **DeepNash / R-NaD** (2206.15378) reached expert Stratego with *no*
belief and *no* opponent model; **SAD** (1912.02288) showed an *implicit*
representation beating a hand-built Bayesian belief. This is a
**representation-level** change — a new inferential subsystem over the event
stream, ~20–40 belief/strategy features, and a leakage referee — not a two-knob
tweak; it is squarely the creative lane and distinct from the conservative
PPO/curriculum twin. It also directly tests the program's load-bearing thesis
(DR-001).

## Relationship to Principle 6 (search-distilled targets) — sequencing, not dropping

**"To Distill or Decide?"** (arXiv 2510.03207, NeurIPS 2025) shows that
distilling a **full-information-optimal** teacher into a **partial-information**
student is *often worse* than distilling a belief-aware / handicapped teacher,
and degrades as hidden-state stochasticity rises — a plausible mechanism behind
the prior 17% cap. The correct realization of Principle 6 therefore needs a
**card-counted teacher**, which requires the belief representation to exist
first. **This cycle builds and validates that representation; the next cycle
distills a belief-conditioned search teacher (native AB / hybrid MCTS) into it.**
The highest-impact lever is being *unblocked and sequenced*, not skipped.

## Belief / strategy features (finalized in the exec spec)

Mirror the structure of the 48-float hidden appendix but compute each legally,
per opponent: expected resource composition `[5]` (exact where public events
determine it, else expected count = public production − known spends), a
per-resource **uncertainty** scalar, dev-cards-by-type expected counts `[5]`,
hidden-VP probability; plus strategy scalars the game-meta research flagged as
the ones that unlock robber/trade play: `opp_completes_build_resource[p][5]`
(trade-danger mask), `is_vp_leader[p]`, `opp_within_2_of_win[p]`, `hand_count[p]`,
`robber_denies_leader`, `steal_target_expected_useful`,
`marginal_ev_per_resource[5]`, `cheapest_build_gap`, `board_scarcity[5]`. Every
feature must pass the event-stream leakage referee.

## Implementation plan (reuse substrate pipelines; ≤2 h incl. data-gen; IL-first)

1. **Branch** `phase3-creat-x1` from `fdf0684`; edits under `models/` only;
   `scripts/check_frozen.sh` stays green. Belief tracker as new module
   `models/belief/` (research surface). Commit the analyzer + tracker (untracked
   files were wiped by a shared-box git op mid-research — the tool needs a
   committed home).
2. **Belief tracker** consumes the per-step public action/event stream the
   train/eval layer already sees (it drives the env). Maintain per-opponent
   exact-or-distribution counts; steal/discard resolved cards are never exposed
   to non-participants. **Leakage referee** (reuse the `ladder/leakage_referee.py`
   / persona-referee pattern): assert every feature is computable from public
   stream + own private state. **No engine change** (DR-001 §3).
3. **Augment the actor input:** concat belief features to the 1084 obs (new actor
   width); **critic unchanged** (already full-info). Keep the split-MLP; widen
   only the actor LayerNorm + first Linear. Verify param count stays < 8M and the
   critic-only wiring test still shows bit-identical actor features under
   critic perturbation.
4. **Data:** reuse `models/alphazero/il_dataset.py` to regenerate native-AB
   teacher games (board-disjoint train/held-out, as in 2.0), additionally logging
   the public event stream and computing belief features per decision.
5. **Warm-start (primary experiment):** reuse `models/alphazero/il_pretrain.py`
   to IL-train the belief-augmented actor (masked-CE to the AB action + dense
   `vp_margin` critic). **Primary comparison:** belief-actor vs plain-actor
   top-1/CE on the *identical* held-out set. Fast (~9 min / 2 epochs per 2.0).
6. **Confirm on the ladder:** run `models/analysis/action_profile.py` (this tool)
   + the pool smoke (`models/baseline/evaluate.py`) trades-on vs the anchor set;
   log win-rate + behavioral slices to W&B `goodsettler-eval`.
7. **Optional PPO polish** only if the IL lift confirms and budget remains (short,
   from the belief-actor warm start). Budget guard: 2 h wall incl. data-gen; ≤3
   env procs; `PYTHONHASHSEED=0`.
8. Every run (including failures) → results row + `docs/reports/p3-x1-report.md`
   + relay-push via the Mac scratchpad clone; message the orchestrator with the
   verdict and next-cycle proposal.

## Risks

- **Leakage:** the entire idea is legal only if the referee passes — gate hard,
  unit-test with the persona-referee harness.
- **Tracker correctness** vs the true hidden appendix: cross-check belief expected
  counts against `write_obs_full`'s 48 floats offline (judge-only) to bound error;
  this is a diagnostic, not a training input.
- **Distribution shift:** belief features on AB-teacher data reflect AB's play;
  validate held-out under pool opponents too before trusting the lift.
- **Actor widening:** keep < 8M params; re-run the actor/critic disjointness
  wiring test.
- **IL-first de-risks** the PPO instability seen in the 2.0 warm sprint (reward
  drifted −0.36 → −0.92 in the short budget).

## Citations

- DR-001 (this repo) — the information ceiling is an encoding limit, not a
  legality limit; belief features are legal research surface.
- Pinto et al. 2017, *Asymmetric Actor-Critic* — https://arxiv.org/abs/1710.06542
  (privileged critic; already spent on our critic side).
- Baisero & Amato, *Unbiased Asymmetric Actor-Critic* — naive privileged critic
  is biased; validate vs symmetric baseline.
- Perolat et al. 2022, *DeepNash / R-NaD* — https://arxiv.org/abs/2206.15378
  (expert imperfect-info play with **no** belief — our falsifier's basis).
- Hu & Foerster 2020, *SAD* — https://arxiv.org/abs/1912.02288 (implicit ≥
  hand-built belief).
- Hernandez-Leal et al. 2019, *Agent Modeling as Auxiliary Task* —
  https://arxiv.org/abs/1907.09597 (opponent/resource prediction head helps).
- *To Distill or Decide?* 2025 — https://arxiv.org/abs/2510.03207 (full-info→
  partial-info distillation fails; distill a belief-aware teacher — why belief
  precedes Principle-6 search-distillation).
- Anthony et al. 2017 *ExIt* (https://arxiv.org/abs/1705.08439) + Silver et al.
  *AlphaZero* (https://arxiv.org/abs/1712.01815) + Wu 2019 *KataGo*
  (https://arxiv.org/abs/1902.10565) — dense value targets / visit-count policy
  targets for the next-cycle search-distillation.
- Catan game-meta (pip table, robber equity, trade EV, resource marginal value):
  playsettlr.com, settlersboard.com, everythingisagame.com, boardgameanalysis.com,
  catanatron docs — source of the strategy-scalar features.

## Runner-up (noted, not this cycle)

**Trade-competent mixed-expert distillation:** distill p2p-trade compose/open/
respond decisions from the trade-competent personas (`balanced-strong`,
`trade-happy`) while keeping native AB for build/robber/dev — directly fills the
`opens_per_game = 0.0` hole. Lower-variance and narrower; strong candidate if the
gate prefers to attack trade initiation before the information ceiling.
