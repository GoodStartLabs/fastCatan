# Stage-2 probe

## Scale-up note (2026-07-19, 01:39Z) — cores 2 -> 12

The 2-core run (tmux `stage2-probe`, W&B `4obf1qxr`, `stage2-probe-overnight`,
~73k learner dec/s) was SIGINT-checkpointed at `learner_decisions = 534,116,666`
(`final.pt` holds `net_state` + `optimizer_state` + `learner_decisions`) and W&B
synced, then relaunched (W&B `3yrkgfd3`, `stage2-probe-overnight-scaled`, same
group) resumed from that checkpoint via `--init-from`.

### Anneal continuity (exact, not approximate)
The trainer has no `--resume` that restores optimizer/anneal state. `--init-from`
restores network weights only and `__init__` sets `learner_decisions = 0`, which
would snap the annealed anchor beta from its current 0.393 back to the initial
0.5 (`_anchor_beta = anchor_coef + progress*(final-initial)`, progress =
`learner_decisions/total`). To preserve semantics the schedule was reconstructed
exactly:
  - `--anchor-coef 0.393177`  (= beta at the save point)
  - `--anchor-coef-final 0.1`
  - `--total-learner-decisions 1,465,883,334`  (= 2e9 - 534,116,666)
This reproduces the original schedule slope AND endpoint from the resume point,
so beta continues 0.393 -> 0.1 unbroken. Verified post-restart: `anchor_kl`
0.019-0.020 continuous with pre-restart 0.0198; `anchor_beta` annealing down from
0.391; entropy 1.78-1.79 in band; no tripwire. Only discontinuity: AdamW moments
reset (brief transient, no entropy spike observed).

### Threading / envs
`num-envs` 4000 -> 8000; `torch-threads` 1 -> 12; `OMP_NUM_THREADS`/`MKL` 1 -> 12;
`taskset` 2 cores -> 12 cores (0-11). Tripwires unchanged (entropy-max 2.35,
min-throughput 50k, low-logs 6).

### Throughput before/after
73k -> ~90-106k learner dec/s (spikes to 106k post-warmup, settles ~90k over the 15-min window); GPU util ~11% -> 0-13% (mostly idle). The
main python holds ~2.5 cores (250% CPU) regardless of the verified 12-core pin:
the single-process rollout loop is GIL/serial-bound, so freed cores past ~3 give
diminishing returns. The 1.4x gain came from OMP env threads + 8k-env per-step
amortization. Reaching 250k+ dec/s needs multiprocess env workers, not more cores
(spec 3.2 punch-list).

### Semantics
Unchanged: same opponents, anchor-ref, `entropy-coef 0`, tripwires, seed path;
only parallelism width and (exactly-continued) anneal params differ.

### W&B reward metric
Enabled `promotion/win_rate` vs random via existing `--promotion-games 128`
(logged at step 0 and run end); no code change. A continuous rollout win-rate
series would require a code change (not done — 3.2 punch-list).

## Proper `--resume` (trainer capability fix — NOT a semantics change)

Implemented on branch `batched-trainer` (train.py + trainer.py) as a permanent
punch-list fix. `--resume <ckpt>` restores net weights + `optimizer_state` +
`learner_decisions` (+ `total_decisions`/`updates`/`episodes`) and, for
checkpoints saved after this change, numpy/torch/cuda RNG state. Because
`learner_decisions` is restored, `_anchor_beta` recomputes the correct in-flight
beta from the ORIGINAL `--anchor-coef` / `--anchor-coef-final` /
`--total-learner-decisions` — so a resumed relaunch needs NO reparameterization
(unlike this run's exact-anneal workaround, which was necessary only because
`--resume` did not yet exist). A startup log line prints resumed
`learner_decisions` and computed `anchor_beta` for at-a-glance continuity checks.
`save()` was extended to persist the extra fields; backward compatible (older
checkpoints resume net+optimizer+decisions, RNG skipped). Explicitly a trainer
capability fix — it does NOT change training semantics. Verified: resuming
`final.pt` restores decisions=534,116,666, beta=0.393177, optimizer moments
present. Built during the live run without relaunching it.

## Cumulative decision accounting

This scaled run used `--init-from`, so its `learner_decisions` counter RESTARTS
at 0. Cumulative stage-2 decisions = old 2-core run (W&B `4obf1qxr`) + scaled run
(W&B `3yrkgfd3`). The end-of-run promotion eval and any compute-budget accounting
MUST sum BOTH runs; do not read the scaled run's counter alone.
