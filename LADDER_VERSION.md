# Ladder version 1.0-v1

Frozen before calibration. Any roster, tier, seed, scoring, or observation-band
change requires a ladder-version bump.

## Pairing and scoring

- Master seed: `0x6a09e667f3bcc909`.
- Board `i`: fastCatan-compatible SplitMix64 of
  `master_seed XOR (i * 0x9e3779b97f4a7c15)`.
- One rotation block is four games on the same board seed, with the candidate in
  seats 0, 1, 2, and 3 once each.
- Smoke: 256 games/opponent-mode (64 board blocks). Full: 1024
  games/opponent-mode (256 board blocks).
- P2P trading-on and trading-off are both run. Maritime trading remains enabled.
- `MAX_TURNS`/other no-winner outcomes are recorded distinctly and count as a
  loss for all four seats.
- Incumbent: `balanced-strong`.
- Promotion: lower 95% Wilson bound of the pooled, equal-sized, trading-on
  legal-information opponent set. Bridge-bot and oracle bands are excluded.

## Legal-information persona roster

| Name | Opening | Build | Robber | Dev | Propose | Respond | Trade lambda |
|---|---|---|---|---|---|---|---:|
| random-legal | random-legal | random-legal | random | never | none | decline-all | 0.75 |
| weighted-random | production-weighted | weighted-random | random | hold-knights | none | own-gain-threshold | 0.75 |
| port-rusher | production-port-synergy | value-greedy | richest-victim | hold-knights | surplus-dump | own-gain-threshold | 0.75 |
| builder-basic | production-weighted | cheapest-first | richest-victim | hold-knights | none | own-gain-threshold | 0.75 |
| builder-strong | production-port-synergy | value-greedy | leader-blocker | timed-play | targeted-need | gain-minus-leader-boost | 1.25 |
| trade-happy | production-port-synergy | value-greedy | leader-blocker | timed-play | targeted-need | own-gain-threshold | 0.20 |
| trade-averse | production-port-synergy | value-greedy | leader-blocker | timed-play | none | gain-minus-leader-boost | 1.50 |
| leader-blocker | production-weighted | value-greedy | leader-blocker | hold-knights | surplus-dump | gain-minus-leader-boost | 1.25 |
| dev-rusher | production-weighted | value-greedy | richest-victim | timed-play | none | own-gain-threshold | 0.75 |
| balanced-strong | production-port-synergy | value-greedy | leader-blocker | timed-play | targeted-need | gain-minus-leader-boost | 0.90 |

Every module reads only the acting seat's own private accessors and public state
decoded from `write_obs`; none calls `write_obs_full`, `ab_value`, or `ab_decide`.

## Catanatron bridge-bot band (regression anchors)

- `catanatron-weighted-random`: native action-type-weight port.
- `catanatron-value`: native one-action plus `ab_value` port.
- `catanatron-alphabeta-d1`: native `ab_decide(depth=1, chance_mode=1)` port.

These are anchor rows and are excluded from the promotion metric. Their one-time
native-vs-real-bridge cross-check uses at least 200 games and overlapping Wilson
intervals under the no-p2p-trades fidelity protocol.

## Oracle band

Frozen names (implemented in Part D): `oracle-ab-d2`, `oracle-ab-d2-blur`,
`oracle-mcts-abvalue-256`, `oracle-mcts-abvalue-1024`. All are full-information,
logged in full runs, and excluded from promotion.
