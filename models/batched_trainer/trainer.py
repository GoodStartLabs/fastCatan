"""End-to-end batched rollout and update loop for fastCatan.

There are no Python per-environment policy calls in the decision loop.  State
signatures, masks, POV observations, actions, and steps are each handled as one
batch.  Learner transitions are event-based: the transition after a learner
decision ends at its next decision or at the episode terminal, so opponent
turns never become fake learner samples.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import fastcatan as fc
from models.alphazero.net import PolicyValueNet, load_policy_value_net
from models.batched_trainer.losses import (
    DistillationBatch,
    DistillationLoss,
    PPOBatch,
    PPOLoss,
)
from models.ckpt import write_stamp

MASK_BYTES = (fc.NUM_ACTIONS + 7) // 8
NO_PLAYER = 255


def terminal_rewards(winners: np.ndarray, learner_seats: np.ndarray) -> np.ndarray:
    """Terminal reward from each environment's learner-seat perspective.

    A no-winner/MAX_TURNS terminal is a loss (-1), as required by spec 3.2.
    """
    winners = np.asarray(winners)
    learner_seats = np.asarray(learner_seats)
    if winners.shape != learner_seats.shape:
        raise ValueError("winner/learner-seat shape mismatch")
    return np.where(winners == learner_seats, 1.0, -1.0).astype(np.float32)


def compute_gae(
    env_ids: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    num_envs: int,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """GAE over chronological, interleaved per-environment transitions."""
    advantages = np.empty(env_ids.size, dtype=np.float32)
    last_adv = np.zeros(num_envs, dtype=np.float32)
    for i in range(env_ids.size - 1, -1, -1):
        env_id = env_ids[i]
        continuation = 1.0 - float(dones[i])
        delta = (float(rewards[i])
                 + gamma * float(next_values[i]) * continuation
                 - float(values[i]))
        adv = (delta + gamma * gae_lambda * continuation
               * last_adv[env_id])
        advantages[i] = adv
        # A terminal ignores later transitions (possibly the next episode),
        # then becomes the successor advantage for earlier transitions in its
        # own episode as the reverse scan continues.
        last_adv[env_id] = adv
    return advantages


@dataclass
class TrainerConfig:
    num_envs: int = 4000
    seed: int = 0
    device: str = "cuda"
    hidden: tuple[int, ...] = (512, 512, 256)
    opponents: tuple[str, str, str] = ("random", "random", "random")
    init_from: str = ""
    rollout_decisions: int = 262_144
    batch_size: int = 65_536
    update_epochs: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    amp: bool = True
    anchor_ref: str = ""
    anchor_coef: float = 0.0
    anchor_coef_final: float = 0.0
    total_learner_decisions: int = 100_000_000


class SeatAssignments:
    """Per-env permutation of policy slots 0..3; slot 0 is the learner."""

    def __init__(self, n: int, rng: np.random.Generator) -> None:
        self.n = n
        self.rng = rng
        self.seat_to_policy = np.empty((n, fc.NUM_PLAYERS), dtype=np.uint8)
        self.reshuffle(np.arange(n, dtype=np.int64))

    def reshuffle(self, env_indices: np.ndarray) -> None:
        env_indices = np.asarray(env_indices, dtype=np.int64)
        if not env_indices.size:
            return
        keys = self.rng.random((env_indices.size, fc.NUM_PLAYERS))
        self.seat_to_policy[env_indices] = np.argsort(keys, axis=1).astype(np.uint8)

    @property
    def learner_seats(self) -> np.ndarray:
        return np.argmax(self.seat_to_policy == 0, axis=1).astype(np.uint8)

    def current_policies(self, current_players: np.ndarray) -> np.ndarray:
        rows = np.arange(self.n)
        return self.seat_to_policy[rows, current_players]


def _load_net(path: str, device: str) -> PolicyValueNet:
    state = torch.load(path, map_location=device, weights_only=False)
    return load_policy_value_net(state, device)


def _sample_random_legal(
    legal: np.ndarray, rows: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One uniform legal action per selected row without a Python row loop."""
    if not rows.size:
        return np.empty(0, dtype=np.uint32)
    local_rows, cols = np.nonzero(legal[rows])
    counts = np.bincount(local_rows, minlength=rows.size)
    if np.any(counts == 0):
        raise RuntimeError("empty legal-action mask")
    starts = np.cumsum(counts) - counts
    ranks = (rng.random(rows.size) * counts).astype(np.int64)
    return cols[starts + ranks].astype(np.uint32, copy=False)


class BatchedTrainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        if config.init_from:
            self.net = _load_net(config.init_from, config.device)
        else:
            self.net = PolicyValueNet(hidden=config.hidden).to(self.device)
        self.net.eval()
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=config.learning_rate, eps=1e-5
        )

        self.anchor = None
        if config.anchor_ref:
            self.anchor = _load_net(config.anchor_ref, config.device)
            self.anchor.eval()
            for parameter in self.anchor.parameters():
                parameter.requires_grad_(False)

        self.opponent_specs = self._normalize_opponents(config.opponents)
        self.opponent_nets: dict[str, torch.nn.Module] = {}
        for spec in set(self.opponent_specs):
            if spec.startswith("checkpoint:"):
                path = spec.split(":", 1)[1]
                net = _load_net(path, config.device)
                net.eval()
                for parameter in net.parameters():
                    parameter.requires_grad_(False)
                self.opponent_nets[spec] = net

        n = config.num_envs
        self.env = fc.BatchedEnv(n, config.seed)
        self.env.reset()
        self.assignments = SeatAssignments(n, self.rng)

        self.sigs = np.empty((n, fc.SIG_INTS), dtype=np.int32)
        self.povs = np.empty(n, dtype=np.uint8)
        self.masks = np.empty((n, fc.MASK_WORDS), dtype=np.uint64)
        self.obs = np.empty((n, fc.OBS_SIZE), dtype=np.float32)
        self.actions = np.empty(n, dtype=np.uint32)
        self.rewards = np.empty(n, dtype=np.float32)
        self.dones = np.empty(n, dtype=np.uint8)
        self.ab_actions = np.empty(n, dtype=np.uint32)
        self._stage_obs_t = None
        self._stage_mask_t = None
        self._stage_obs_np = None
        self._stage_mask_np = None
        if self.device.type == "cuda":
            # Reused pinned staging avoids a fresh pageable advanced-index
            # allocation for every policy forward.
            self._stage_obs_t = torch.empty(
                (n, fc.OBS_SIZE), dtype=torch.float32, pin_memory=True
            )
            self._stage_mask_t = torch.empty(
                (n, fc.NUM_ACTIONS), dtype=torch.bool, pin_memory=True
            )
            self._stage_obs_np = self._stage_obs_t.numpy()
            self._stage_mask_np = self._stage_mask_t.numpy()

        # A learner action is written once into a preallocated rollout slot;
        # reward/bootstrap fields are filled at its next decision or terminal.
        # Allocation is lazy so benchmark-only runs do not reserve ~1.2 GiB.
        self.pending_slot = np.full(n, -1, dtype=np.int64)
        self._rollout: dict[str, np.ndarray] | None = None
        self._rollout_size = 0
        self._completed = 0

        self.total_decisions = 0
        self.learner_decisions = 0
        self.updates = 0
        self.episodes = 0
        self.entropy_sum = 0.0
        self.entropy_count = 0
        self.last_update_metrics: dict[str, float] = {}
        self.times = {name: 0.0 for name in
                      ("env_step", "obs_encode", "forward", "rollout_store",
                       "update")}

    @staticmethod
    def _normalize_opponents(specs: tuple[str, ...]) -> tuple[str, str, str]:
        if len(specs) == 1:
            specs = specs * 3
        if len(specs) != 3:
            raise ValueError("opponents must contain one spec or exactly three")
        allowed = {"random", "self", "ab1", "ab2"}
        for spec in specs:
            if spec not in allowed and not spec.startswith("checkpoint:"):
                raise ValueError(f"unsupported vectorized opponent: {spec}")
        return tuple(specs)  # type: ignore[return-value]

    def _anchor_beta(self) -> float:
        if not self.anchor:
            return 0.0
        progress = min(1.0, self.learner_decisions
                       / max(1, self.config.total_learner_decisions))
        return (self.config.anchor_coef
                + progress * (self.config.anchor_coef_final
                              - self.config.anchor_coef))

    def _encode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current players, policy slots, and legal bool masks."""
        self.env.write_sigs(self.sigs)
        np.copyto(self.povs, self.sigs[:, 0], casting="unsafe")
        self.env.write_masks(self.masks)
        # Explicit POV batch is the actor path; never write_obs_full.
        self.env.write_obs_pov_batch(self.povs, self.obs)
        legal = np.unpackbits(
            self.masks.view(np.uint8), axis=1, bitorder="little"
        )[:, :fc.NUM_ACTIONS].astype(bool, copy=False)
        if not bool(legal.any(axis=1).all()):
            raise RuntimeError("empty legal-action mask")
        policies = self.assignments.current_policies(self.povs)
        return self.povs, policies, legal

    @torch.no_grad()
    def _act_net(
        self, net: torch.nn.Module, rows: np.ndarray, legal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not rows.size:
            empty_f = np.empty(0, dtype=np.float32)
            return np.empty(0, dtype=np.uint32), empty_f, empty_f, empty_f
        if self.device.type == "cuda":
            assert self._stage_obs_np is not None
            assert self._stage_mask_np is not None
            assert self._stage_obs_t is not None
            assert self._stage_mask_t is not None
            np.take(self.obs, rows, axis=0,
                    out=self._stage_obs_np[:rows.size])
            np.take(legal, rows, axis=0,
                    out=self._stage_mask_np[:rows.size])
            obs_t = self._stage_obs_t[:rows.size].to(
                self.device, non_blocking=True
            )
            mask_t = self._stage_mask_t[:rows.size].to(
                self.device, non_blocking=True
            )
        else:
            obs_t = torch.from_numpy(self.obs[rows])
            mask_t = torch.from_numpy(legal[rows])
        amp = self.config.amp and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type,
                            dtype=torch.bfloat16, enabled=amp):
            logits, values = net(obs_t)
        logp_all = torch.log_softmax(
            logits.float().masked_fill(~mask_t, -1e9), dim=1
        )
        probs = logp_all.exp()
        # Exact categorical sampling via the Gumbel-max trick.  This is faster
        # than torch.multinomial for the many small legal supports in Catan.
        exponential = torch.empty_like(logp_all).exponential_()
        actions = (logp_all - exponential.log()).argmax(dim=1)
        logp = logp_all.gather(1, actions[:, None]).squeeze(1)
        entropy = -(probs * logp_all).sum(dim=1)
        # One device->host synchronization for all outputs.
        packed = torch.stack(
            (actions.float(), logp, values.float(), entropy), dim=1
        ).cpu().numpy()
        return (
            packed[:, 0].astype(np.uint32, copy=False),
            packed[:, 1].astype(np.float32, copy=False),
            packed[:, 2].astype(np.float32, copy=False),
            packed[:, 3].astype(np.float32, copy=False),
        )

    @torch.no_grad()
    def _act_random(self, rows: np.ndarray, legal: np.ndarray) -> np.ndarray:
        """Uniform-random legal actions, vectorized on the idle GPU."""
        if not rows.size:
            return np.empty(0, dtype=np.uint32)
        if self.device.type == "cpu":
            return _sample_random_legal(legal, rows, self.rng)
        assert self._stage_mask_np is not None
        assert self._stage_mask_t is not None
        np.take(legal, rows, axis=0,
                out=self._stage_mask_np[:rows.size])
        mask = self._stage_mask_t[:rows.size].to(
            self.device, non_blocking=True
        )
        scores = torch.rand(mask.shape, device=self.device)
        actions = scores.masked_fill(~mask, -1.0).argmax(dim=1)
        return actions.cpu().numpy().astype(np.uint32, copy=False)

    def _ensure_rollout(self) -> dict[str, np.ndarray]:
        if self._rollout is None:
            capacity = self.config.rollout_decisions + 4 * self.config.num_envs
            self._rollout = {
                "env": np.empty(capacity, dtype=np.int32),
                # float32 avoids an expensive CPU conversion in the hot loop.
                "obs": np.empty((capacity, fc.OBS_SIZE), dtype=np.float32),
                "mask": np.empty((capacity, MASK_BYTES), dtype=np.uint8),
                "action": np.empty(capacity, dtype=np.uint16),
                "logp": np.empty(capacity, dtype=np.float32),
                "value": np.empty(capacity, dtype=np.float32),
                "reward": np.empty(capacity, dtype=np.float32),
                "done": np.empty(capacity, dtype=np.uint8),
                "next_value": np.empty(capacity, dtype=np.float32),
                "complete": np.zeros(capacity, dtype=bool),
            }
        return self._rollout

    def _complete_pending(
        self,
        rows: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_values: np.ndarray,
    ) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if not rows.size:
            return
        slots = self.pending_slot[rows]
        valid = slots >= 0
        rows = rows[valid]
        if not rows.size:
            return
        slots = slots[valid]
        rollout = self._ensure_rollout()
        rollout["reward"][slots] = np.asarray(rewards, dtype=np.float32)[valid]
        rollout["done"][slots] = np.asarray(dones, dtype=np.uint8)[valid]
        rollout["next_value"][slots] = np.asarray(
            next_values, dtype=np.float32
        )[valid]
        rollout["complete"][slots] = True
        self._completed += slots.size

    def _set_pending(
        self,
        rows: np.ndarray,
        legal: np.ndarray,
        actions: np.ndarray,
        logp: np.ndarray,
        values: np.ndarray,
    ) -> None:
        rollout = self._ensure_rollout()
        end = self._rollout_size + rows.size
        if end > rollout["env"].size:
            raise RuntimeError("rollout capacity exhausted before update")
        slots = np.arange(self._rollout_size, end, dtype=np.int64)
        rollout["env"][slots] = rows
        rollout["obs"][slots] = self.obs[rows]
        rollout["mask"][slots] = np.packbits(
            legal[rows], axis=1, bitorder="little"
        )
        rollout["action"][slots] = actions
        rollout["logp"][slots] = logp
        rollout["value"][slots] = values
        rollout["complete"][slots] = False
        self.pending_slot[rows] = slots
        self._rollout_size = end

    def collect_step(self, store_rollout: bool = True) -> int:
        t0 = time.perf_counter()
        _players, policies, legal = self._encode()
        self.times["obs_encode"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        self.actions.fill(fc.SKIP_ACTION)
        learner_rows = np.flatnonzero(policies == 0)

        # Learner and self-play opponent rows share one GPU forward.
        learner_or_self = policies == 0
        for slot, spec in enumerate(self.opponent_specs, start=1):
            if spec == "self":
                learner_or_self |= policies == slot
        shared_rows = np.flatnonzero(learner_or_self)
        shared_actions, shared_logp, shared_values, shared_entropy = self._act_net(
            self.net, shared_rows, legal
        )
        self.actions[shared_rows] = shared_actions

        # Locate learner outputs inside the sorted shared-row vector.
        learner_pos = np.searchsorted(shared_rows, learner_rows)
        learner_values = shared_values[learner_pos]
        learner_actions = shared_actions[learner_pos]
        learner_logp = shared_logp[learner_pos]
        learner_entropy = shared_entropy[learner_pos]

        # One forward per distinct frozen checkpoint policy.
        for spec, net in self.opponent_nets.items():
            slots = [i + 1 for i, value in enumerate(self.opponent_specs)
                     if value == spec]
            rows = np.flatnonzero(np.isin(policies, slots))
            actions, _lp, _v, _ent = self._act_net(net, rows, legal)
            self.actions[rows] = actions

        # Random-legal opponents are sampled as a vectorized ragged batch.
        random_slots = [i + 1 for i, value in enumerate(self.opponent_specs)
                        if value == "random"]
        if random_slots:
            rows = np.flatnonzero(np.isin(policies, random_slots))
            self.actions[rows] = self._act_random(rows, legal)

        # Native AB is itself a batched OpenMP call.  Identical AB policies
        # share the call; inactive rows are ignored when copying its output.
        for spec, depth in (("ab1", 1), ("ab2", 2)):
            slots = [i + 1 for i, value in enumerate(self.opponent_specs)
                     if value == spec]
            if slots:
                self.env.ab_decide_batch(depth, False, self.ab_actions)
                rows = np.flatnonzero(np.isin(policies, slots))
                self.actions[rows] = self.ab_actions[rows]

        if bool((self.actions == fc.SKIP_ACTION).any()):
            raise RuntimeError("policy failed to fill an action")
        self.times["forward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if store_rollout:
            self._complete_pending(
                learner_rows,
                np.zeros(learner_rows.size, dtype=np.float32),
                np.zeros(learner_rows.size, dtype=np.uint8),
                learner_values,
            )
            self._set_pending(
                learner_rows, legal, learner_actions, learner_logp, learner_values
            )
        self.times["rollout_store"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        self.env.step(self.actions, self.rewards, self.dones)
        done_rows = np.flatnonzero(self.dones)
        if done_rows.size:
            winners = np.fromiter(
                (self.env.last_winner(int(i)) for i in done_rows),
                dtype=np.uint8,
                count=done_rows.size,
            )
            if store_rollout:
                rewards = terminal_rewards(
                    winners, self.assignments.learner_seats[done_rows]
                )
                self._complete_pending(
                    done_rows,
                    rewards,
                    np.ones(done_rows.size, dtype=np.uint8),
                    np.zeros(done_rows.size, dtype=np.float32),
                )
                self.pending_slot[done_rows] = -1
            self.assignments.reshuffle(done_rows)
            self.episodes += done_rows.size
        self.times["env_step"] += time.perf_counter() - t0

        count = learner_rows.size
        self.total_decisions += self.config.num_envs
        self.learner_decisions += count
        self.entropy_sum += float(learner_entropy.sum())
        self.entropy_count += count
        return count

    def _rollout_arrays(self) -> dict[str, np.ndarray]:
        if self._rollout is None or not self._completed:
            raise RuntimeError("empty rollout")
        complete = np.flatnonzero(
            self._rollout["complete"][:self._rollout_size]
        )
        keys = ("env", "obs", "mask", "action", "logp", "value", "reward",
                "done", "next_value")
        data = {key: self._rollout[key][complete] for key in keys}
        n = data["env"].size
        advantages = compute_gae(
            data["env"], data["reward"], data["done"], data["value"],
            data["next_value"], self.config.num_envs, self.config.gamma,
            self.config.gae_lambda,
        )
        data["advantages"] = advantages
        data["returns"] = advantages + data["value"]
        return data

    def update_ppo(self) -> dict[str, float]:
        t0 = time.perf_counter()
        data = self._rollout_arrays()
        n = data["env"].size
        advantages = data["advantages"]
        advantages = ((advantages - advantages.mean())
                      / max(float(advantages.std()), 1e-8)).astype(np.float32)

        objective = PPOLoss(
            clip_coef=self.config.clip_coef,
            value_coef=self.config.value_coef,
            entropy_coef=self.config.entropy_coef,
            anchor_ref=self.anchor,
            anchor_coef=self._anchor_beta(),
        )
        # Stage the rollout once.  Per-minibatch NumPy advanced indexing and
        # host->device copies left the A100 idle and dominated early probes.
        obs_all = torch.from_numpy(data["obs"]).to(self.device)
        mask_np = np.unpackbits(
            data["mask"], axis=1, bitorder="little"
        )[:, :fc.NUM_ACTIONS].astype(bool, copy=False)
        legal_all = torch.from_numpy(mask_np).to(self.device)
        actions_all = torch.from_numpy(data["action"].astype(np.int64)).to(self.device)
        logp_all = torch.from_numpy(data["logp"]).to(self.device)
        advantages_all = torch.from_numpy(advantages).to(self.device)
        returns_all = torch.from_numpy(data["returns"]).to(self.device)

        sums: dict[str, float] = {}
        batches = 0
        self.net.train()
        for _epoch in range(self.config.update_epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, self.config.batch_size):
                idx = order[start:start + self.config.batch_size]
                batch = PPOBatch(
                    obs=obs_all[idx],
                    legal=legal_all[idx],
                    actions=actions_all[idx],
                    old_logp=logp_all[idx],
                    advantages=advantages_all[idx],
                    returns=returns_all[idx],
                )
                self.optimizer.zero_grad(set_to_none=True)
                amp = self.config.amp and self.device.type == "cuda"
                with torch.autocast(device_type=self.device.type,
                                    dtype=torch.bfloat16, enabled=amp):
                    loss, metrics = objective(self.net, batch)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite PPO loss: {loss}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                               self.config.max_grad_norm)
                self.optimizer.step()
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                batches += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.net.eval()
        if self._rollout is not None:
            self._rollout["complete"][:self._rollout_size] = False
        self._rollout_size = 0
        self._completed = 0
        # Avoid carrying an action sampled by the pre-update policy into the
        # next on-policy segment.
        self.pending_slot.fill(-1)
        self.updates += 1
        out = {key: value / batches for key, value in sums.items()}
        out.update({"anchor_beta": self._anchor_beta(),
                    "rollout_samples": float(n)})
        self.last_update_metrics = out
        self.times["update"] += time.perf_counter() - t0
        return out

    def update_distillation(
        self,
        obs: np.ndarray,
        legal: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        value_coef: float = 1.0,
    ) -> dict[str, float]:
        """Run one optimizer step on stage3-style search targets."""
        batch = DistillationBatch(
            obs=torch.from_numpy(np.asarray(obs, np.float32)).to(self.device),
            legal=torch.from_numpy(np.asarray(legal, bool)).to(self.device),
            policy_targets=torch.from_numpy(np.asarray(policy_targets)).to(self.device),
            value_targets=torch.from_numpy(np.asarray(value_targets, np.float32)).to(self.device),
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = DistillationLoss(value_coef)(self.net, batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite distillation loss: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return {key: float(value) for key, value in metrics.items()}

    def reset_measurements(self) -> None:
        self.total_decisions = 0
        self.learner_decisions = 0
        self.entropy_sum = 0.0
        self.entropy_count = 0
        self.episodes = 0
        for key in self.times:
            self.times[key] = 0.0

    def summary(self, wall_seconds: float) -> dict[str, Any]:
        measured = sum(self.times.values())
        summary: dict[str, Any] = {
            "num_envs": self.config.num_envs,
            "wall_seconds": wall_seconds,
            "learner_decisions": self.learner_decisions,
            "total_decisions": self.total_decisions,
            "learner_decisions_per_s": self.learner_decisions / max(wall_seconds, 1e-9),
            "total_decisions_per_s": self.total_decisions / max(wall_seconds, 1e-9),
            "episodes": self.episodes,
            "updates": self.updates,
            "entropy": self.entropy_sum / max(self.entropy_count, 1),
            "time_s": dict(self.times),
            "time_ms_per_step": {
                key: value / max(self.total_decisions / self.config.num_envs, 1) * 1000
                for key, value in self.times.items()
            },
            "unattributed_s": max(0.0, wall_seconds - measured),
            "last_update": self.last_update_metrics,
        }
        if self.device.type == "cuda":
            summary["gpu_mem_allocated_mb"] = (
                torch.cuda.max_memory_allocated(self.device) / 2**20
            )
            try:
                raw = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ], text=True, timeout=2).strip().splitlines()[0]
                util, memory = [float(x.strip()) for x in raw.split(",")]
                summary["gpu_util_percent_sample"] = util
                summary["gpu_memory_used_mb_sample"] = memory
            except Exception:
                pass
        return summary

    def run(
        self,
        duration_seconds: float = 0.0,
        max_learner_decisions: int = 0,
        benchmark_only: bool = False,
        warmup_steps: int = 20,
        log_interval_seconds: float = 10.0,
        log_callback=None,
    ) -> dict[str, Any]:
        for _ in range(warmup_steps):
            self.collect_step(store_rollout=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.pending_slot.fill(-1)
        if self._rollout is not None:
            self._rollout["complete"][:self._rollout_size] = False
        self._rollout_size = 0
        self._completed = 0
        self.reset_measurements()

        start = time.perf_counter()
        last_log = start
        while True:
            self.collect_step(store_rollout=not benchmark_only)
            if not benchmark_only and self._completed >= self.config.rollout_decisions:
                self.update_ppo()
            now = time.perf_counter()
            if log_callback is not None and now - last_log >= log_interval_seconds:
                log_callback(self.summary(now - start))
                last_log = now
            if duration_seconds and now - start >= duration_seconds:
                break
            if max_learner_decisions and self.learner_decisions >= max_learner_decisions:
                break
            if not duration_seconds and not max_learner_decisions:
                raise ValueError("duration_seconds or max_learner_decisions is required")

        if (not benchmark_only and self._completed >= self.config.batch_size
                and (not duration_seconds or time.perf_counter() - start < duration_seconds * 1.05)):
            self.update_ppo()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - start
        return self.summary(wall)

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "net_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "learner_decisions": self.learner_decisions,
            "extra": extra or {},
        }
        torch.save(payload, path)
        write_stamp(path)

    @staticmethod
    def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
        Path(path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
