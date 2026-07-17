# Ladder results schema (v1)

`results/ladder.parquet` is append-only and contains one row per four-game cyclic
rotation block. Within a block, one board seed is replayed four times and the
candidate occupies each seat exactly once. Every matchup and trading mode uses
the same indexed board-seed stream.

No-winner games (including the engine's `MAX_TURNS=2000` backstop) remain in
`games` and are a loss for all four seats. They are separately counted in
`no_winner`; `no_winner_policy` is frozen to `loss_for_all_seats`.

| Column group | Columns | Meaning |
|---|---|---|
| Identity | `schema_version`, `ladder_version`, `run_id`, `timestamp_utc`, `tier` | Schema/run provenance. |
| Matchup | `candidate`, `candidate_band`, `opponent`, `opponent_band`, `promotion_eligible`, `anchor`, `incumbent`, `mode` | Roster and reporting strata. `mode` is `trading_on` or `trading_off` (p2p only; maritime remains legal). |
| Pairing | `rotation_block`, `master_seed`, `board_seed` | Call-order-independent SplitMix64 board identity. Seeds are fixed-width hex strings to avoid signed parquet coercion. |
| Outcome | `games`, `candidate_wins`, `opponent_wins`, `no_winner`, `no_winner_policy` | `games = candidate_wins + opponent_wins + no_winner`. Win share and Wilson denominators always use `games`. |
| Seat slices | `candidate_seat{0..3}_games`, `candidate_seat{0..3}_wins`, `winner_seats_json` | Candidate exposure/wins by realized seat and the four raw winner seats (`-1` is no winner). |
| Performance | `decisions`, `wall_seconds`, `decisions_per_second` | Policy+engine ladder throughput, excluding result serialization and W&B upload. |
| Runtime | `git_commit`, `hostname`, `python_version` | Reproduction provenance. |

The promotion metric is the lower 95% Wilson bound of pooled candidate wins over
equal-sized, trading-on, legal-information opponent matchups. Oracle-band rows and
any other row with `promotion_eligible=false` are excluded. Trading-off rows are
reported as the protocol delta, never silently mixed into the promotion number.
