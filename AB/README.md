# AB/ — M4: Alpha-Beta eval + final-model thesis gate

The thesis claim lives here: **the trained RL agent beats Catanatron's
Alpha-Beta with statistical significance — win rate > 25% over ≥1000 four-player
games, 95% CI** (`PLAN.md` M4, root). 0.25 is the 4-player chance baseline.

The agent plays through `bridge/CatanatronBridge` *inside Catanatron's reference
engine*, so the numbers are directly comparable to Catanatron paper baselines.

## Files

| File | Role |
|---|---|
| `policy.py` | wraps a trained checkpoint as a bridge `PolicyFn` (`obs, mask, rng -> int`). Registry mirrors `models/eval.py`; only `ppo` wired today. Raises on obs/action-dim mismatch. |
| `tournament.py` | the harness: policy-via-bridge vs `AlphaBetaPlayer`/`ValueFunctionPlayer`/`RandomPlayer`. Win rate + 95% Wilson CI + thesis gate → `results/*.json`. |
| `soak.py` | 10⁸-step stability soak (pure fastcatan): finite-obs + mask-integrity + leak checks. |
| `REPRODUCIBILITY.md` | toolchain, build flags, the **two-env** setup, **catanatron git pin**, seeds, train config. |
| `results/` | tournament result JSONs + `validation_1084.md` (pipeline validation). |

## Environment — use anaconda, not `.venv`

The RL interface is **obs 1084 / actions 286**. Two interpreters exist; the
1084 build lives in **anaconda** (see `REPRODUCIBILITY.md` §5):

- **`/home/sinan/anaconda3/bin/python`** — 1084/286 fastcatan + catanatron 3.3.0
  + sb3. **Train and eval M4 here.**
- `./.venv/bin/python` — stale 724/296 build, legacy M2 only. No catanatron.

Catanatron is a **pinned git build, not PyPI** (3.3.0 @ `41ba0db`, "deterministic
discards"); installed editable from `/home/sinan/Desktop/msc/catanatron` and
recorded in root `requirements.txt`. Verified at this commit: 281/281
`bridge/tests` pass (2026-05-27). `soak.py` needs only fastcatan and runs in
either env. See `REPRODUCIBILITY.md` §6 for why the pin must be a commit (newer
builds move `models.tiles` → `models.map` and break bridge import).

## Run (anaconda)

```bash
AP=/home/sinan/anaconda3/bin/python

# 1) Train the M4 model on the 1084/286 interface (768 envs).
$AP -m models.train_ppo --num-envs 768 --total-steps 20_000_000 --run-name ppo_1084_20m
#    ~15-18 min. >=30M for gate margin; 20M is near-convergence.

# 2) Thesis gate (slow — AlphaBeta ~6.4 s/game unpruned, ~1.8 h/1000): vs Alpha-Beta.
PYTHONHASHSEED=0 $AP -m AB.tournament \
    --ckpt models/checkpoints/ppo_1084_20m/ppo_final.zip \
    --games 1000 --opponent alphabeta --ab-depth 2 --ab-prune --seed 42

# Smoke (seconds): vs random.
$AP -m AB.tournament --games 20 --opponent random --ckpt models/checkpoints/ppo_1084_20m/ppo_final.zip

# 10^8 soak (~minutes at ~70k steps/s).
$AP -m AB.soak --steps 100000000 --seed 7
```

The evaluated model is a `--ckpt` flag — swap in any future M3 self-play
checkpoint (must be 1084/286) freely.

## Status (2026-05-27)

- [x] PPO→bridge policy adapter (`policy.py`) — smoke: 30/30 legal picks.
- [x] Tournament harness (`tournament.py`) — win rate + 95% CI + gate + JSON.
- [x] Soak harness (`soak.py`) — smoke: 10k steps, RSS flat (1.00×), STABILITY PASS.
- [x] catanatron pinned to git `41ba0db` (3.3.0, not PyPI), installed editable +
      recorded in root `requirements.txt`. Bridge verified: 281/281 tests pass.
- [x] Found + resolved the obs-interface drift: source was 1084/286 but the build
      + every checkpoint were stale 724/296. fastcatan rebuilt to 1084/286
      (anaconda, `editable.rebuild=true` so it can't go stale again).
- [x] Wiped all 724/296 checkpoints — `models/checkpoints/` is empty; next run
      retrains from scratch on the 1084/286 interface.
- [x] 1084 pipeline validated end-to-end: `test_obs_identity` 5/5 (encoder↔C++
      parity) + uniform-bridge games vs Value/AlphaBeta complete. See
      `results/validation_1084.md`.
- [x] Reproducibility doc.
- [ ] **Train the 1084/286 M4 model** (`ppo_1084_20m`) — command above (in progress).
- [ ] Final model vs Alpha-Beta, ≥1000 games (after the 1084 model exists).
- [ ] Full 10⁸-step soak (smoke green; ~24 min full).
- [ ] Record thesis-gate result.
