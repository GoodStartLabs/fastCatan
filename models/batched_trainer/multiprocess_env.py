"""Shared-memory process shards for :class:`fastcatan.BatchedEnv`.

Each worker owns one native ``BatchedEnv`` and writes its global row slice into
shared NumPy buffers.  Pipes carry only small barrier commands; observations,
masks, actions, rewards, and terminal metadata are never pickled.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import torch

import fastcatan as fc

NO_PLAYER = 255


def _array(buffer: Any, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(buffer, dtype=dtype).reshape(shape)


def _worker_main(
    connection: Connection,
    buffers: dict[str, Any],
    num_envs: int,
    start: int,
    stop: int,
    seed: int,
    cpu_affinity: tuple[int, ...],
) -> None:
    """Own one shard and service shared-memory barrier commands."""
    try:
        if cpu_affinity and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpu_affinity)
        # ``torch.set_num_threads`` sets the OpenMP width used by the native
        # simulator too.  One native thread per process prevents N x N
        # oversubscription when the parent is pinned to a fixed core budget.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        count = stop - start
        env = fc.BatchedEnv(count, seed)
        env.reset()

        sigs = _array(
            buffers["sigs"], np.dtype(np.int32), (num_envs, fc.SIG_INTS)
        )[start:stop]
        povs = _array(
            buffers["povs"], np.dtype(np.uint8), (num_envs,)
        )[start:stop]
        masks = _array(
            buffers["masks"], np.dtype(np.uint64),
            (num_envs, fc.MASK_WORDS),
        )[start:stop]
        legal = _array(
            buffers["legal"], np.dtype(np.bool_),
            (num_envs, fc.NUM_ACTIONS),
        )[start:stop]
        obs = _array(
            buffers["obs"], np.dtype(np.float32),
            (num_envs, fc.OBS_SIZE),
        )[start:stop]
        actions = _array(
            buffers["actions"], np.dtype(np.uint32), (num_envs,)
        )[start:stop]
        rewards = _array(
            buffers["rewards"], np.dtype(np.float32), (num_envs,)
        )[start:stop]
        dones = _array(
            buffers["dones"], np.dtype(np.uint8), (num_envs,)
        )[start:stop]
        winners = _array(
            buffers["winners"], np.dtype(np.uint8), (num_envs,)
        )[start:stop]
        ab_actions = _array(
            buffers["ab_actions"], np.dtype(np.uint32), (num_envs,)
        )[start:stop]
        snapshots = _array(
            buffers["snapshots"], np.dtype(np.uint8),
            (num_envs, fc.SNAPSHOT_BYTES),
        )[start:stop]

        while True:
            command = connection.recv()
            op = command[0]
            if op == "close":
                connection.send(("ok",))
                return
            if op == "observe":
                env.write_sigs(sigs)
                np.copyto(povs, sigs[:, 0], casting="unsafe")
                env.write_masks(masks)
                legal[:] = np.unpackbits(
                    masks.view(np.uint8), axis=1, bitorder="little"
                )[:, :fc.NUM_ACTIONS]
                env.write_obs_pov_batch(povs, obs)
            elif op == "sigs":
                env.write_sigs(sigs)
            elif op == "masks":
                env.write_masks(masks)
                legal[:] = np.unpackbits(
                    masks.view(np.uint8), axis=1, bitorder="little"
                )[:, :fc.NUM_ACTIONS]
            elif op == "obs":
                env.write_obs_pov_batch(povs, obs)
            elif op == "step":
                env.step(actions, rewards, dones)
                for row in np.flatnonzero(dones):
                    winners[row] = env.last_winner(int(row))
            elif op == "ab":
                env.ab_decide_batch(int(command[1]), bool(command[2]), ab_actions)
            elif op == "save":
                env.save_snapshots(snapshots)
            elif op == "load":
                env.load_snapshots(snapshots)
            elif op == "reset":
                env.reset()
                winners.fill(NO_PLAYER)
            else:
                raise ValueError(f"unknown worker command: {op}")
            connection.send(("ok",))
    except (EOFError, BrokenPipeError):
        return
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class ProcessBatchedEnv:
    """A row-compatible ``BatchedEnv`` split across worker processes."""

    _SPECS = {
        "sigs": (np.dtype(np.int32), lambda n: (n, fc.SIG_INTS)),
        "povs": (np.dtype(np.uint8), lambda n: (n,)),
        "masks": (np.dtype(np.uint64), lambda n: (n, fc.MASK_WORDS)),
        "legal": (np.dtype(np.bool_), lambda n: (n, fc.NUM_ACTIONS)),
        "obs": (np.dtype(np.float32), lambda n: (n, fc.OBS_SIZE)),
        "actions": (np.dtype(np.uint32), lambda n: (n,)),
        "rewards": (np.dtype(np.float32), lambda n: (n,)),
        "dones": (np.dtype(np.uint8), lambda n: (n,)),
        "winners": (np.dtype(np.uint8), lambda n: (n,)),
        "ab_actions": (np.dtype(np.uint32), lambda n: (n,)),
        "snapshots": (
            np.dtype(np.uint8), lambda n: (n, fc.SNAPSHOT_BYTES)
        ),
    }

    def __init__(
        self,
        num_envs: int,
        seed: int,
        worker_count: int,
        start_method: str = "spawn",
        split_affinity: bool = False,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if worker_count < 1 or worker_count > num_envs:
            raise ValueError("worker_count must be in [1, num_envs]")
        self.num_envs = num_envs
        self.worker_count = worker_count
        self._closed = False
        self._pending_command = False
        self._context = mp.get_context(start_method)
        self._original_affinity: tuple[int, ...] = ()
        worker_affinity: tuple[int, ...] = ()
        main_affinity: tuple[int, ...] = ()
        if split_affinity and hasattr(os, "sched_getaffinity"):
            self._original_affinity = tuple(sorted(os.sched_getaffinity(0)))
            if len(self._original_affinity) < 2:
                raise ValueError("split affinity requires at least two CPUs")
            worker_affinity = self._original_affinity[:-1]
            main_affinity = self._original_affinity[-1:]
        self._buffers: dict[str, Any] = {}
        for name, (dtype, make_shape) in self._SPECS.items():
            shape = make_shape(num_envs)
            size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            self._buffers[name] = self._context.RawArray("b", size)
            setattr(self, name, _array(self._buffers[name], dtype, shape))
        self.winners.fill(NO_PLAYER)

        # Contiguous, exhaustive shards keep all global trainer row IDs stable.
        bounds = np.linspace(
            0, num_envs, worker_count + 1, dtype=np.int64
        )
        self.shards = tuple(
            (int(bounds[i]), int(bounds[i + 1]))
            for i in range(worker_count)
        )
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        for worker_id, (start, stop) in enumerate(self.shards):
            parent, child = self._context.Pipe(duplex=True)
            # Stable, non-overlapping master streams per shard.  The exact
            # trajectories are deterministic for a fixed worker topology.
            worker_seed = (
                int(seed)
                ^ ((worker_id + 1) * 0x9E3779B97F4A7C15)
            ) & ((1 << 64) - 1)
            process = self._context.Process(
                target=_worker_main,
                args=(
                    child, self._buffers, num_envs, start, stop, worker_seed,
                    worker_affinity,
                ),
                name=f"fastcatan-env-{worker_id}",
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        if main_affinity:
            os.sched_setaffinity(0, main_affinity)
        # Force startup errors to surface in the constructor.
        self.observe()

    def _send(self, command: tuple[Any, ...]) -> None:
        if self._closed:
            raise RuntimeError("multiprocess environment is closed")
        if self._pending_command:
            raise RuntimeError("worker command already pending")
        for connection in self._connections:
            connection.send(command)
        self._pending_command = True

    def _receive(self) -> None:
        if not self._pending_command:
            raise RuntimeError("no worker command is pending")
        errors = []
        for worker_id, connection in enumerate(self._connections):
            try:
                reply = connection.recv()
            except (EOFError, BrokenPipeError) as exc:
                errors.append(f"worker {worker_id} disconnected: {exc}")
                continue
            if not reply or reply[0] != "ok":
                detail = reply[1] if len(reply) > 1 else repr(reply)
                errors.append(f"worker {worker_id} failed:\n{detail}")
        self._pending_command = False
        if errors:
            raise RuntimeError("\n".join(errors))

    def _barrier(self, command: tuple[Any, ...]) -> None:
        self._send(command)
        self._receive()

    def observe(self) -> None:
        self._barrier(("observe",))

    def reset(self) -> None:
        self._barrier(("reset",))

    def write_sigs(self, out: np.ndarray) -> None:
        self._barrier(("sigs",))
        if out is not self.sigs:
            np.copyto(out, self.sigs)

    def write_masks(self, out: np.ndarray) -> None:
        self._barrier(("masks",))
        if out is not self.masks:
            np.copyto(out, self.masks)

    def write_obs_pov_batch(
        self, povs: np.ndarray, out: np.ndarray
    ) -> None:
        if povs is not self.povs:
            np.copyto(self.povs, povs)
        self._barrier(("obs",))
        if out is not self.obs:
            np.copyto(out, self.obs)

    def step(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        if actions is not self.actions:
            np.copyto(self.actions, actions)
        self._barrier(("step",))
        if rewards is not self.rewards:
            np.copyto(rewards, self.rewards)
        if dones is not self.dones:
            np.copyto(dones, self.dones)

    def ab_decide_batch(
        self, depth: int, prune: bool, out: np.ndarray
    ) -> None:
        self._barrier(("ab", depth, prune))
        if out is not self.ab_actions:
            np.copyto(out, self.ab_actions)

    def save_snapshots(self, out: np.ndarray) -> None:
        self._barrier(("save",))
        if out is not self.snapshots:
            np.copyto(out, self.snapshots)

    def load_snapshots(self, snapshots: np.ndarray) -> None:
        if snapshots is not self.snapshots:
            np.copyto(self.snapshots, snapshots)
        self._barrier(("load",))

    def last_winner(self, env_idx: int) -> int:
        return int(self.winners[env_idx])

    def close(self) -> None:
        if self._closed:
            return
        if self._pending_command:
            try:
                self._receive()
            except Exception:
                pass
        for connection in self._connections:
            try:
                connection.send(("close",))
            except (EOFError, BrokenPipeError):
                pass
        for connection in self._connections:
            try:
                if connection.poll(1.0):
                    connection.recv()
            except (EOFError, BrokenPipeError):
                pass
            connection.close()
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        if self._original_affinity and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, self._original_affinity)
        self._closed = True

    def __enter__(self) -> "ProcessBatchedEnv":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
