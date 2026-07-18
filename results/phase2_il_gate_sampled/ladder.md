# Ladder results (schema v0)

Append-only; one row per (ladder_version, candidate, opponent, mode, rotation). Canonical store is `ladder.parquet`; this table mirrors it.

| ts | ladder | candidate | opponent | mode | games | win_rate | wilson_low | no_winner | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1784339490 | 1.0-v1 | 2.0-il-ab-d1-100k-sampled | builder-basic | trades_on | 64 | 0.750 | 0.632 | 0.000 | checkpoint_smoke |
| 1784339490 | 1.0-v1 | 2.0-il-ab-d1-100k-sampled | builder-basic | trades_off | 64 | 0.750 | 0.632 | 0.000 | checkpoint_smoke |
