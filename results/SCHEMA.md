# Results schema v0 (frozen at `substrate-v1`)

The append-only contract every tournament / checkpoint / kill writes, so a result
that isn't recorded didn't happen and results are never silently compared across
ladder versions. Canonical store: `results/ladder.parquet` (append-only). Human
mirror: `results/ladder.md`. Writer + column list: `results/schema.py`
(`SCHEMA_VERSION = "v0"`).

**Frozen.** Adding, renaming, removing, or retyping a column is a schema change:
bump `SCHEMA_VERSION`, update this file, and — because the schema is eval-visible —
bump the ladder version (`LADDER_VERSION.md`, owned by spec 1.0). `check_frozen.sh`
guards `schema.py` and this doc.

One row per **(ladder_version, candidate, opponent, mode, rotation)**.

| column | type | meaning |
|---|---|---|
| `schema_version` | str | this contract's version (`v0`) |
| `ts` | float | unix seconds when the row was written |
| `ladder_version` | str | frozen roster + seed set the row was measured against |
| `candidate` | str | agent under test — checkpoint id or persona name |
| `opponent` | str | opponent-set label (e.g. `3x random`, `AB-d2`) |
| `mode` | str | `trades_on` \| `trades_off` — **both are always logged** (program requirement) |
| `rotation` | str | `full` (seat-balanced block) or a seat index for a worst-seat slice |
| `games` | int | games in this row |
| `wins` | int | candidate wins |
| `win_rate` | float | `wins / games` |
| `wilson_low` | float | Wilson lower bound (z=1.96) — **the promotion statistic** |
| `wilson_high` | float | Wilson upper bound |
| `no_winner_rate` | float | fraction ending with no winner (MAX_TURNS backstop / stall) — never hidden |
| `seat_wins` | json[4] | wins by seat — the worst-seat slice |
| `trading_delta` | float\|null | `win_rate(trades_on) − win_rate(trades_off)`; set on the `trades_on` row |
| `decisions_per_s` | float | throughput |
| `param_count` | int | candidate parameter count (0 for scripted/random) |
| `commit` | str | engine commit the row was produced on |
| `config_hash` | str | hash of the run config |
| `wandb_url` | str\|null | W&B run, if any |
| `verdict` | str | `pass` \| `fail` \| `baseline` \| free text |
| `notes` | str | free text |

**Promotion metric** (program.md): the Wilson lower bound (`wilson_low`) of the
candidate's win share against the fixed legal-information opponent set (oracles
excluded), aggregated over the full seat-rotation block with paired boards. Always
also logged: the per-opponent rows, `seat_wins` (worst-seat), `no_winner_rate`,
and the `trading_delta`. Regression gates compare a candidate against designated
anchor opponents; a candidate cannot promote by exploiting one opponent.
