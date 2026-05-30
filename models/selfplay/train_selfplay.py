"""M3 iterative self-play with frozen-snapshot rotation.

One MaskablePPO model + one DummyVecEnv of SelfPlayEnv; the opponent pool is
shared and mutated across rounds. Each round:
  1. train the learner for `--steps-per-round` (= the snapshot interval),
  2. freeze the current policy to `snap_<timesteps>.zip` and add it to the pool,
  3. inline-gate the latest snapshot vs the snapshot from `--gate-lag` rounds ago.

Warm-start from an M2 checkpoint loads *weights only* (`set_parameters`) so the
swept hyperparams (lr, ent_coef) still apply. DummyVecEnv only: the pool holds
loaded torch models (not cheaply picklable) and shares across envs in-process.

Run (real schedule, warm-started from the M2 10M checkpoint):
    python -m models.selfplay.train_selfplay \
        --init-from models/checkpoints/ppo_random_10m/ppo_final.zip \
        --seed-pool --num-envs 8 --steps-per-round 1_000_000 --num-rounds 12 \
        --run-name sp_main

Resume a crashed run (continue, NOT restart) — add --resume and re-pass only
--run-name; the schedule/gate flags are restored from the run's run_config.json:
    python -m models.selfplay.train_selfplay --resume --run-name sp_main
It reloads snap_*.zip into the pool (recency window + gate-lag baseline intact),
MaskablePPO.load()s the latest snapshot for full optimizer + num_timesteps state
(so snapshot numbering keeps going), restores the gate log, and picks up at the
next round. --resume on a dir with no snapshots is a no-op (fresh start).

League / PFSP (opt-in via --league): replaces the sliding-window pool with a
bounded archive of the best snapshots, sampled by prioritized fictitious self-
play (opponents you lose to are sampled more). See models/selfplay/league.py.
    python -m models.selfplay.train_selfplay --init-from <ckpt> --seed-pool \
        --no-p2p-trade --league --league-size 32 --league-recent 8 \
        --pfsp hard --pfsp-beta 2 --league-decay 0.9 --num-rounds 40 \
        --run-name sp_league
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from models.env import MAX_TRADE_COMPOSE_PER_TURN
from models.selfplay.gate import gate_result, play_2v2
from models.selfplay.league import League
from models.selfplay.opponents import OpponentPool, PolicyOpponent
from models.selfplay.selfplay_env import SelfPlayEnv


CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def _mask_fn(env):
    return env.action_masks()


def _tb_dir_if_available(tb_dir: Path) -> str | None:
    """SB3 hard-errors if tensorboard_log is set but tensorboard is missing.
    Return the path only when tensorboard imports; else None (SB3 still logs
    to stdout). Avoids coupling self-play runs to an optional dep."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("[selfplay] tensorboard not installed -> stdout logging only")
        return None
    return str(tb_dir)


def _build_vec_env(
    pool: OpponentPool, num_envs: int, base_seed: int, suppress_p2p_trade: bool,
    trade_compose_cap: int = MAX_TRADE_COMPOSE_PER_TURN,
) -> DummyVecEnv:
    def _make(seed: int):
        def _thunk():
            e = SelfPlayEnv(pool, seed=seed, suppress_p2p_trade=suppress_p2p_trade,
                            trade_compose_cap=trade_compose_cap)
            e = ActionMasker(e, _mask_fn)
            return Monitor(e)

        return _thunk

    return DummyVecEnv([_make(base_seed + i) for i in range(num_envs)])


# --- resume support -------------------------------------------------------
# A run writes snap_<num_timesteps>.zip each round, run_config.json once at
# start, and one gate_log.jsonl line per gate. `--resume` reads all three back
# so a crashed run continues instead of degrading to a one-snapshot restart.

_SNAP_RE = re.compile(r"^snap_(\d+)\.zip$")

# Loop/gate/env args the learner's checkpoint does NOT carry (PPO hyperparams
# ARE restored by MaskablePPO.load; these are not). Persisted on a fresh run and
# restored on --resume so a continue can't silently diverge from the original —
# e.g. a mismatched --num-rounds would miscalibrate the linear lr decay, and a
# mismatched --no-p2p-trade would gate under different rules than training.
_RESUME_KEYS = (
    "num_rounds", "steps_per_round", "num_envs",
    "lr", "lr_schedule", "ent_coef", "target_kl", "net_arch",
    "p_random", "pool_window", "opponent_device", "no_p2p_trade",
    "trade_compose_cap",
    "gate_lag", "gate_games", "gate_threshold", "seed",
    "init_from", "seed_pool",
    "league", "league_size", "league_recent", "pfsp", "pfsp_beta", "league_decay",
)


def _discover_snaps(save_dir: Path) -> list[Path]:
    """Existing round snapshots in round order. snap_<N>.zip is named by the
    num_timesteps at that freeze, so sorting by N recovers the round order."""
    found = []
    for p in save_dir.glob("snap_*.zip"):
        m = _SNAP_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def _write_run_config(path: Path, args: argparse.Namespace) -> None:
    path.write_text(json.dumps({k: getattr(args, k) for k in _RESUME_KEYS}, indent=2))


def _restore_run_config(path: Path, args: argparse.Namespace) -> bool:
    """Overwrite the resume-relevant args in place from a prior run's config.
    Returns False if absent (a run predating --resume) so the caller can warn."""
    if not path.exists():
        return False
    cfg = json.loads(path.read_text())
    for k in _RESUME_KEYS:
        if k in cfg:
            setattr(args, k, cfg[k])
    return True


def _append_gate_log(path: Path, entry: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _load_gate_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _save_league_state(path: Path, pool: League) -> None:
    path.write_text(json.dumps(pool.state(), indent=2))


def _load_league_state(path: Path) -> dict | None:
    """The league archive is a bounded, evicted subset + PFSP counts that the
    snap_*.zip set alone can't reconstruct, so --resume restores it from here."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # self-play schedule
    p.add_argument("--init-from", type=str, default=None,
                   help="M2 checkpoint to warm-start weights from (set_parameters).")
    p.add_argument("--seed-pool", action="store_true",
                   help="Add the init checkpoint to the pool so round 0 already "
                        "faces a real opponent (else round 0 is vs random).")
    p.add_argument("--num-rounds", type=int, default=10)
    p.add_argument("--steps-per-round", type=int, default=1_000_000,
                   help="Train steps between snapshots == the snapshot interval.")
    p.add_argument("--num-envs", type=int, default=8)
    # pool sampling
    p.add_argument("--p-random", type=float, default=0.2)
    p.add_argument("--pool-window", type=int, default=5)
    p.add_argument("--opponent-device", type=str, default="cpu",
                   help="Device for frozen opponents. CPU is faster for the "
                        "single-obs forward passes (no GPU launch/transfer "
                        "overhead per call); the learner keeps its own device.")
    p.add_argument("--no-p2p-trade", action="store_true",
                   help="Forbid player-to-player trades in train AND gate. Kills "
                        "the trade-loop stall so self-play games terminate; without "
                        "it strong-vs-strong games stall to the cap and the gate is "
                        "inconclusive. See SelfPlayEnv / models/selfplay/PLAN.md.")
    p.add_argument("--trade-compose-cap", type=int,
                   default=MAX_TRADE_COMPOSE_PER_TURN,
                   help="Per-seat trade-compose actions/turn before the compose "
                        "block is masked, for learner AND opponents AND the gate. "
                        "Lets trades stay ON without the ADD/CANCEL stall (keeps "
                        "real trades, kills only churn). Lower = games terminate "
                        "faster (more likely decidable within the step budget).")
    # PPO hyperparams (mirror models/train_ppo.py defaults)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--lr-schedule", choices=["constant", "linear"],
                   default="constant",
                   help="'linear' decays lr from --lr toward 0 across the whole "
                        "run (per-round stepwise: lr*(1-round/num_rounds)). Avoids "
                        "the late-round oscillation that flattens self-play gains.")
    p.add_argument("--target-kl", type=float, default=None,
                   help="PPO early-stops an update once policy KL exceeds this "
                        "(e.g. 0.03). Keeps steps conservative while the opponent "
                        "pool shifts each round; default None = off.")
    p.add_argument("--net-arch", type=str, default=None,
                   help="Hidden layer sizes for the pi & vf MLPs, comma-separated "
                        "(e.g. '256,256'). Default: SB3 default [64,64]. NOTE: an "
                        "arch differing from --init-from's checkpoint cannot "
                        "warm-start (shape mismatch) -> that run trains from "
                        "scratch (logged). Seed-pool opponents keep their own arch.")
    # gate
    p.add_argument("--gate-lag", type=int, default=2,
                   help="Compare latest snapshot vs the one this many rounds ago.")
    p.add_argument("--gate-games", type=int, default=200)
    p.add_argument("--gate-threshold", type=float, default=0.55)
    # league (PFSP) — opt-in alternative to the sliding-window pool
    p.add_argument("--league", action="store_true",
                   help="Use a bounded PFSP league of the best snapshots instead "
                        "of the sliding-window pool: opponents are sampled by how "
                        "often they currently beat the learner (prioritized "
                        "fictitious self-play). See models/selfplay/league.py.")
    p.add_argument("--league-size", type=int, default=32,
                   help="Max snapshots kept in the league archive.")
    p.add_argument("--league-recent", type=int, default=8,
                   help="Always keep the most-recent K snapshots; fill the rest "
                        "with the hardest-for-the-learner, evicting the easiest "
                        "non-recent member when the archive overflows.")
    p.add_argument("--pfsp", choices=["hard", "even"], default="hard",
                   help="PFSP weight over p=learner win-rate vs the opponent: "
                        "'hard' (1-p)^beta favors opponents you lose to; 'even' "
                        "p(1-p) favors evenly-matched.")
    p.add_argument("--pfsp-beta", type=float, default=2.0,
                   help="Exponent for --pfsp hard (higher = sharper focus on the "
                        "hardest opponents).")
    p.add_argument("--league-decay", type=float, default=1.0,
                   help="Per-round multiplicative decay on the win/loss counts "
                        "(e.g. 0.9) so PFSP tracks the improving learner; 1.0=off.")
    # bookkeeping
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR))
    p.add_argument("--run-name", type=str, default="selfplay")
    p.add_argument("--progress", action="store_true", help="SB3 progress bar.")
    p.add_argument("--resume", action="store_true",
                   help="Continue a crashed run in --run-name's dir: reload "
                        "snap_*.zip into the pool, MaskablePPO.load() the latest "
                        "(optimizer + num_timesteps, so numbering continues), "
                        "restore run_config.json + gate_log.jsonl, and start at "
                        "the next round. Just re-pass --run-name; the schedule/"
                        "gate flags are restored from run_config.json. No-op "
                        "(fresh start) if the dir has no snapshots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = save_dir / "tb"
    cfg_path = save_dir / "run_config.json"
    gate_log_path = save_dir / "gate_log.jsonl"
    league_state_path = save_dir / "league_state.json"

    # Resume discovery FIRST: a found snapshot set restores the schedule/gate
    # flags (so a continue can't silently drift from the original run) before we
    # build the pool, env, and model that read them.
    resume_snaps = _discover_snaps(save_dir) if args.resume else []
    if args.resume and not resume_snaps:
        print(f"[selfplay] --resume: no snap_*.zip in {save_dir} -> fresh start.")
    if resume_snaps:
        if _restore_run_config(cfg_path, args):
            print(f"[selfplay] resume: restored schedule/gate config "
                  f"from {cfg_path.name}")
        else:
            print("[selfplay] resume: no run_config.json (run predates --resume?) "
                  "-> using CLI flags; re-pass the SAME flags the run started with.")

    start_round = len(resume_snaps)
    snap_paths: list[Path] = list(resume_snaps)
    gate_log: list[dict] = _load_gate_log(gate_log_path) if resume_snaps else []

    if args.league:
        pool = League(
            seats=[1, 2, 3], capacity=args.league_size, recent=args.league_recent,
            p_random=args.p_random, pfsp=args.pfsp, beta=args.pfsp_beta,
            seed=args.seed,
        )
        print(f"[selfplay] league: size={args.league_size} "
              f"recent={args.league_recent} pfsp={args.pfsp} "
              f"beta={args.pfsp_beta} decay={args.league_decay}")
    else:
        pool = OpponentPool(
            seats=[1, 2, 3], seed=args.seed,
            p_random=args.p_random, window=args.pool_window,
        )
    env = _build_vec_env(pool, args.num_envs, args.seed, args.no_p2p_trade,
                         args.trade_compose_cap)

    if resume_snaps:
        if args.league:
            # The archive is a bounded, evicted subset + PFSP counts that the
            # snap set alone can't reconstruct, so restore it from league_state.
            # Missing (older/partial run) -> rebuild from the most-recent snaps
            # with reset stats (membership approximate; PFSP relearns quickly).
            st = _load_league_state(league_state_path)
            if st:
                pool.load_state(st, device=args.opponent_device)
                print(f"[selfplay] resume: restored league ({len(pool)} members) "
                      f"from {league_state_path.name}")
            else:
                for sp in resume_snaps[-args.league_size:]:
                    pool.add_candidate(PolicyOpponent.load(
                        sp, name=sp.stem, device=args.opponent_device), sp)
                print(f"[selfplay] resume: no {league_state_path.name}; rebuilt "
                      f"league from last {len(pool)} snaps (PFSP stats reset).")
        else:
            # Window pool: append init (if seeded) then every snap in round order,
            # so the recency window is whole again and the gate's snap_paths
            # indexing still aligns. The pool never evicts.
            if args.init_from and args.seed_pool:
                pool.add_candidate(PolicyOpponent.load(
                    args.init_from, name="init", device=args.opponent_device),
                    args.init_from)
                print("[selfplay] resume: re-seeded pool with init checkpoint")
            for i, sp in enumerate(resume_snaps):
                pool.add_candidate(PolicyOpponent.load(
                    sp, name=f"snap{i}", device=args.opponent_device), sp)
        # Load the latest snapshot for FULL state — weights, optimizer, AND
        # num_timesteps — so snapshot numbering continues instead of restarting.
        model = MaskablePPO.load(resume_snaps[-1], env=env)
        print(f"[selfplay] RESUMED from {resume_snaps[-1].name}: round "
              f"{start_round}/{args.num_rounds}, num_timesteps="
              f"{model.num_timesteps}, pool={len(pool)}, "
              f"gate entries restored={len(gate_log)}")
    else:
        policy_kwargs = None
        if args.net_arch:
            hidden = [int(x) for x in args.net_arch.split(",") if x.strip()]
            policy_kwargs = {"net_arch": hidden}
            print(f"[selfplay] net_arch = {hidden}")

        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            seed=args.seed,
            tensorboard_log=_tb_dir_if_available(tb_dir),
            verbose=1,
        )

        if args.init_from:
            # Weights only — keeps our swept lr/ent_coef. Arch must match the ckpt;
            # a swept --net-arch that differs will mismatch -> fall back to scratch.
            try:
                model.set_parameters(args.init_from, device=model.device)
                print(f"[selfplay] warm-started weights from {args.init_from}")
            except (ValueError, RuntimeError, KeyError) as e:
                print(f"[selfplay] WARM-START SKIPPED (arch mismatch with "
                      f"{args.init_from}?): {type(e).__name__}: {str(e)[:140]} "
                      f"-> training from scratch")
            if args.seed_pool:
                pool.add_candidate(PolicyOpponent.load(
                    args.init_from, name="init", device=args.opponent_device),
                    args.init_from)
                print("[selfplay] seeded pool with init checkpoint")

        # Persist resume-relevant config + start clean gate/league logs. Fresh
        # runs only: a resume keeps these so it can restore from / append to them.
        # (league_state is rewritten with the full pool at round 0's freeze; a
        # crash before that leaves 0 snaps -> resume starts fresh anyway.)
        _write_run_config(cfg_path, args)
        gate_log_path.unlink(missing_ok=True)
        league_state_path.unlink(missing_ok=True)

    for rnd in range(start_round, args.num_rounds):
        if args.league and args.league_decay < 1.0 and rnd > 0:
            # Fade prior rounds' results so PFSP weights track the improving
            # learner. rnd>0 only: round 0 has no games yet, and on resume the
            # restored counts are pre-decay-for-this-round (saved at end of rnd-1).
            pool.decay_stats(args.league_decay)
        if args.lr_schedule == "linear":
            # Decay across the WHOLE run, stepwise per round. We call learn() once
            # per round, so SB3's built-in progress_remaining resets to 1.0 each
            # round (would sawtooth lr); overriding lr_schedule here decays it
            # globally: round 0 -> --lr, last round -> ~--lr/num_rounds.
            round_lr = args.lr * (1.0 - rnd / args.num_rounds)
            model.lr_schedule = lambda _progress, _lr=round_lr: _lr
            print(f"[selfplay] round {rnd}: lr -> {round_lr:.2e}")
        model.learn(
            total_timesteps=args.steps_per_round,
            reset_num_timesteps=(rnd == 0),
            progress_bar=args.progress,
        )

        snap = save_dir / f"snap_{model.num_timesteps}.zip"
        model.save(snap)
        snap_paths.append(snap)
        pool.add_candidate(PolicyOpponent.load(
            snap, name=f"snap{rnd}", device=args.opponent_device), snap)
        if args.league:
            _save_league_state(league_state_path, pool)
        print(f"[selfplay] round {rnd}: saved {snap.name}, pool={len(pool)}")

        # Inline gate: latest vs N-rounds-ago. Load both from snap_paths (the
        # append-only on-disk history), not the pool, so it's correct whether the
        # pool is a window or a league that may have evicted the N-ago snapshot,
        # and under --resume (snap_paths is restored). Cheap vs a round of training.
        if len(snap_paths) > args.gate_lag:
            latest = PolicyOpponent.load(
                snap_paths[-1], name=snap_paths[-1].stem,
                device=args.opponent_device)
            nago = PolicyOpponent.load(
                snap_paths[-1 - args.gate_lag],
                name=snap_paths[-1 - args.gate_lag].stem,
                device=args.opponent_device)
            latest_wins, decided, no_winner = play_2v2(
                latest, nago, args.gate_games, seed=args.seed + rnd,
                suppress_p2p=args.no_p2p_trade,
                trade_compose_cap=args.trade_compose_cap)
            r = gate_result(latest_wins, decided, no_winner, args.gate_threshold)
            entry = {
                "round": rnd, "timesteps": model.num_timesteps,
                "latest": snap_paths[-1].name,
                "nago": snap_paths[-1 - args.gate_lag].name, **r,
            }
            gate_log.append(entry)
            # Persist each gate as it's computed so --resume appends to the log
            # instead of starting it over — a crash keeps every gate so far.
            _append_gate_log(gate_log_path, entry)
            tag = "PASS" if r["pass"] else ("inconclusive" if not r["conclusive"] else "fail")
            print(f"[selfplay] gate r{rnd}: latest vs -{args.gate_lag} "
                  f"= {r['win_rate']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}] "
                  f"(no-winner {r['no_winner_rate']:.0%}) {tag}")

    final = save_dir / "selfplay_final.zip"
    model.save(final)

    last_gate = gate_log[-1] if gate_log else None
    summary = {
        "run_name": args.run_name,
        "config": {
            "lr": args.lr, "lr_schedule": args.lr_schedule,
            "ent_coef": args.ent_coef, "target_kl": args.target_kl,
            "steps_per_round": args.steps_per_round,
            "num_rounds": args.num_rounds, "num_envs": args.num_envs,
            "p_random": args.p_random, "pool_window": args.pool_window,
            "net_arch": args.net_arch,
            "trade_compose_cap": args.trade_compose_cap,
            "gate_lag": args.gate_lag, "init_from": args.init_from,
        },
        "final_ckpt": str(final),
        "gate_log": gate_log,
        "final_gate_rate": last_gate["win_rate"] if last_gate else None,
        "final_gate_pass": last_gate["pass"] if last_gate else None,
    }
    if args.league:
        summary["league"] = {
            "size": args.league_size, "recent": args.league_recent,
            "pfsp": args.pfsp, "beta": args.pfsp_beta, "decay": args.league_decay,
            "archive": pool.state()["members"],
        }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[selfplay] done -> {final}")
    print(f"[selfplay] summary -> {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
