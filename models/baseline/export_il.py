"""Export supervised Phase-2 weights as a tournament-loadable PPO checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from models.baseline.env import Phase2CatanEnv
from models.baseline.policy import Phase2Policy, load_il_weights, parameter_counts
from models.ckpt import write_stamp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--il-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    def factory():
        env = Phase2CatanEnv(seed=args.seed)
        return Monitor(ActionMasker(env, lambda wrapped: wrapped.action_masks()))

    env = DummyVecEnv([factory])
    model = MaskablePPO(
        Phase2Policy,
        env,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=0.999,
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    state = load_il_weights(model.policy, args.il_checkpoint)
    counts = parameter_counts(model.policy)
    if counts["total"] > 8_000_000:
        raise ValueError(f"parameter budget exceeded: {counts}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    # SB3 appends .zip when the caller omits it.
    actual = out if out.suffix == ".zip" else Path(str(out) + ".zip")
    write_stamp(actual)
    print(
        f"[export] {actual} actor={counts['actor']:,} critic={counts['critic']:,} "
        f"total={counts['total']:,} il_top1={state.get('val_top1')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
