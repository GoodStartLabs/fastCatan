# Self-play training on the TU Berlin HPC — step by step

Goal: run the **az-scratch-vpm** self-play arm (pure AlphaZero, dense
`vp_margin` values, zero AlphaBeta data) on a GPU node, chained past
walltime limits, ending with the native AB-d2 ladder number.

Everything below assumes the one-time setup from `scripts/hpc/README.md`
(conda env `catan`, repo at `$HOME/fastCatan`, native build verified on a
compute node). If that's done, this is 4 commands.

## 1. Sync the repo (local machine)

```bash
git push                                   # make sure your remote has HEAD
ssh <tub-login>@gateway.hpc.tu-berlin.de   # [verify hostname]
cd $HOME/fastCatan && git pull
```

## 2. One 30-second smoke (login or interactive node)

```bash
conda activate catan
python -m models.alphazero.batched_selfplay --total-games 8 \
    --num-games 8 --sims 16 --device cpu --value-mode vp_margin \
    --save-dir /tmp/sp_smoke
```

Want: `[done] 8 games ...`. If `fastcatan` import errors → rebuild on a
compute node (README §2, `-march=native` caveat).

## 3. Launch

```bash
cd $HOME/fastCatan
jid=$(sbatch --parsable scripts/hpc/selfplay.sbatch)
echo $jid
```

Defaults: 60 000 games, 256 concurrent games, 128 sims, 512,512,256 net,
**trades OFF**, checkpoints every 2 000 games into
`models/checkpoints/az_scratch_vpm_hpc/`. Override via `--export`:

```bash
sbatch --export=ALL,GAMES=100000,SIMS=256 scripts/hpc/selfplay.sbatch
```

### Trades-ON capacity arm (the current target)

Pure scratch, **p2p trades ON** (`ALLOW_TRADES=1`), 2048-wide net. Net sizes
contain commas, which break `--export=VAR=a,b,c` parsing — so export in the
shell and propagate with `--export=ALL`:

```bash
export ALLOW_TRADES=1 HIDDEN=2048,2048,1024 SIMS=128 \
       GAMES=20000 NUM_GAMES=256 \
       SAVE_DIR=models/checkpoints/az_scratch_vpm_trades2048
jid=$(sbatch --parsable --export=ALL scripts/hpc/selfplay.sbatch)
echo $jid
sbatch --dependency=afterany:$jid --export=ALL scripts/hpc/selfplay.sbatch   # chain
```

**Throughput reality:** trades-on is ~30–40× the decisions/game of trades-off
(smoke: ~3 300 dec/game, 2048 net). On the local 4070 that's ~700 games/hr at
sims-12, ≈30 games/hr at sims-256. So:
- Prefer **SIMS=128** over 256 (2× cheaper, marginal target-quality loss).
- This is a **multi-day chained run**, not an overnight job — `checkpoint-every
  2000` makes every job resumable; chain with `--dependency=afterany`.
- The final AB-d2 ladder only runs when a job *reaches* its `GAMES` target; for
  a walltime-killed job, run it by hand on the newest `az_g*.pt`:
  `python -m models.alphazero.evaluate --ckpt <ckpt> --opponent alphabeta
  --ab-depth 2 --sims 512 --games 200 --device cpu --allow-trades`.

## 4. Chain extra walltime (optional, repeatable)

The script auto-resumes from the newest checkpoint in `SAVE_DIR`, so
chaining is just a dependency:

```bash
sbatch --dependency=afterany:$jid scripts/hpc/selfplay.sbatch
```

Each chained job plays `GAMES` MORE games on top of the loaded net.
(Note: optimizer state + replay buffer restart per job — net weights carry.)

## 5. Watch

```bash
squeue -u $USER
tail -f slurm-selfplay-<jid>.out
```

Healthy log lines:
- `[it  NNN] games=... dec/s=...` — throughput; expect dec/s to dominate
  wall clock, g/s rising as the net stops stalling games.
- `[eval g=N] raw-policy vs-random 0.xx` — raw net strength, should climb
  from ~0.25 toward 0.7+ over the run.
- `[ckpt] .../az_gNNNN.pt` — resume points.

## 6. Read the result

The job ends with the dev ladder (200 games vs native AB-d2 @512 sims):

```
=== AlphaZero vs AlphaBeta(d=2,prune=False) ===
win rate: 0.xxxx  95% CI [...]
```

also saved to `models/checkpoints/az_scratch_vpm_hpc/ladder_abd2.txt`.

How to place the number (2026-06 reference band):

| config | win % vs AB-d2 |
|---|---|
| self-contained learned-judge cells | 16.5–20.0 |
| parity | 25 |
| hybrid (ab_value leaves) reference | 29.5 |

- ≥ the 17–20 band → self-play matched AB-data distillation without any
  AB data — strong thesis arm, consider scaling games/sims.
- ≈ 0–10 → the historical pure-AZ result reproduces even with dense
  values; the OOD argument stands.

## 7. Bring results home

```bash
# from your local machine:
rsync -av <tub-login>@gateway.hpc.tu-berlin.de:fastCatan/models/checkpoints/az_scratch_vpm_hpc/ \
    models/checkpoints/az_scratch_vpm_hpc/
```

Checkpoints are numpy-2.x (venv) compatible; load with
`models.alphazero.net.load_policy_value_net`.
