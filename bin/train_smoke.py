"""One-command PPO wiring smoke (spec 0.3 Task 5).

Runs the *shipped* MaskablePPO trainer construction (`models.train_ppo._build_vec_env`
+ MaskablePPO) for a short budget, then asserts the pipeline actually learns and
moves data: policy loss is finite and throughput clears a floor. Creates a W&B
run (project `goodsettler`, name `wiring-smoke`) so "one command trains and it is
recorded" is provable. Not a training run — a wiring proof.

Env knobs: SMOKE_STEPS (default 60000), FPS_FLOOR (default 300), WANDB_MODE.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback

from models.train_ppo import _build_vec_env

SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "300000"))
FPS_FLOOR = float(os.environ.get("FPS_FLOOR", "800"))


class _SmokeCB(BaseCallback):
    def __init__(self, wandb_run):
        super().__init__()
        self.run = wandb_run
        self.last_loss = None
        self.last_fps = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        v = self.model.logger.name_to_value
        if "train/loss" in v:
            self.last_loss = float(v["train/loss"])
        if "time/fps" in v:
            self.last_fps = float(v["time/fps"])
        if self.run is not None and self.last_loss is not None:
            self.run.log({"train/loss": self.last_loss,
                          "time/fps": self.last_fps or 0.0,
                          "num_timesteps": int(self.num_timesteps)})


def main() -> None:
    import wandb

    run = wandb.init(project="goodsettler", name="wiring-smoke",
                     config={"smoke_steps": SMOKE_STEPS, "fps_floor": FPS_FLOOR},
                     reinit=True)

    env = _build_vec_env(
        num_envs=8, base_seed=42, use_subproc=False, shaped=False,
        shaping_coef=0.1, gamma=0.999, opponent="random", ab_depth=2,
        ab_prune=False, suppress_p2p_trade=False,
    )
    model = MaskablePPO(
        MaskableActorCriticPolicy, env,
        policy_kwargs=dict(net_arch=[64, 64]),
        n_steps=256, batch_size=1024, n_epochs=3, learning_rate=3e-4,
        gamma=0.999, seed=42, verbose=0,
    )
    cb = _SmokeCB(run)
    t0 = time.time()
    model.learn(total_timesteps=SMOKE_STEPS, callback=cb, progress_bar=False)
    elapsed = time.time() - t0
    fps = SMOKE_STEPS / elapsed if elapsed else 0.0

    ok_loss = cb.last_loss is not None and math.isfinite(cb.last_loss)
    ok_fps = fps >= FPS_FLOOR
    if run is not None:
        run.summary["final_loss"] = cb.last_loss
        run.summary["fps"] = fps
        run.summary["elapsed_s"] = elapsed
        run.summary["wiring_smoke_pass"] = bool(ok_loss and ok_fps)
    url = getattr(run, "url", None)
    if run is not None:
        run.finish()

    print(f"[train_smoke] loss={cb.last_loss} finite={ok_loss} "
          f"fps={fps:.0f} floor={FPS_FLOOR} elapsed={elapsed:.0f}s steps={SMOKE_STEPS}")
    print(f"[train_smoke] WANDB_URL={url}")
    assert ok_loss, f"policy loss not finite: {cb.last_loss}"
    assert ok_fps, f"throughput {fps:.0f} fps below floor {FPS_FLOOR}"
    print("[train_smoke] PASS")


if __name__ == "__main__":
    main()
