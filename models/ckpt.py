"""Checkpoint env stamping + numpy-compat guard (cross-device safety net).

Two cheap, additive guards so checkpoints move between devices safely (see
[[device-env-canonical]] / requirements.txt for the canonical env):

  1. write_stamp(ckpt)  — after every save, drop a `<ckpt>.env.json` sidecar
     recording the interpreter that pickled it (python / numpy / torch / git-sha /
     obs+action dims). Turns a checkpoint from a black box into something you can
     inspect before trusting on another machine.

  2. verify_stamp(ckpt) — before every load, read that sidecar and refuse on the
     real pickle-breaker: a numpy MAJOR-version mismatch (numpy 1.x <-> 2.x pickles
     are incompatible — the numpy._core rename + BitGenerator / gym-Space pickle).
     One clear line instead of a cryptic unpickle stacktrace deep in SB3.

Both are best-effort: legacy checkpoints with no sidecar only warn (we can't know
what pickled them). Stamps live beside the checkpoint and are git-ignorable noise.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

STAMP_SUFFIX = ".env.json"


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - git absent / not a repo
        return "?"


def _fingerprint() -> dict:
    fp: dict = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "git_sha": _git_sha(),
    }
    try:
        import torch

        fp["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import fastcatan

        fp["obs_size"] = int(fastcatan.OBS_SIZE)
        fp["num_actions"] = int(fastcatan.NUM_ACTIONS)
    except Exception:  # noqa: BLE001
        pass
    return fp


def stamp_path(ckpt: str | Path) -> Path:
    """The sidecar path for a checkpoint: `<ckpt>.env.json`."""
    return Path(str(ckpt) + STAMP_SUFFIX)


def write_stamp(ckpt: str | Path) -> Path:
    """Write the env fingerprint beside a just-saved checkpoint. Never raises."""
    p = stamp_path(ckpt)
    try:
        p.write_text(json.dumps(_fingerprint(), indent=2) + "\n")
    except Exception as e:  # noqa: BLE001 - a stamp failure must not fail training
        warnings.warn(f"[ckpt] could not write env stamp {p.name}: {e}")
    return p


def verify_stamp(ckpt: str | Path, *, strict: bool = True) -> None:
    """Check the env that pickled `ckpt` can load it here, BEFORE unpickling.

    numpy MAJOR mismatch raises RuntimeError when ``strict`` (SB3 .zip: real
    pickle-breaker), else warns (torch .pt: pure tensors survive a numpy major
    bump). Missing/unreadable sidecar, or python/torch/obs drift, only warn.
    """
    name = Path(ckpt).name
    sp = stamp_path(ckpt)
    if not sp.exists():
        warnings.warn(
            f"[ckpt] {name}: no env stamp ({sp.name}) — cannot verify numpy compat "
            f"(pickled before stamping). Load may crash if numpy major differs."
        )
        return
    try:
        saved = json.loads(sp.read_text())
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"[ckpt] {name}: unreadable env stamp ({e}); skipping check.")
        return

    saved_np = str(saved.get("numpy", ""))
    saved_major = saved_np.split(".", 1)[0]
    cur_major = np.__version__.split(".", 1)[0]
    if saved_major and saved_major != cur_major:
        msg = (
            f"[ckpt] {name} was pickled under numpy {saved_np}, but this env has "
            f"numpy {np.__version__}. numpy {saved_major}.x<->{cur_major}.x pickles are "
            f"INCOMPATIBLE. Load it in the matching env or re-save it. "
            f"Canonical env: conda `catan` (numpy 1.26.4) — see requirements.txt."
        )
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg)
        return

    # Soft drift: obs/action layout determines net I/O dims — a mismatch means the
    # checkpoint predates an obs-format change and won't load cleanly anyway.
    cur = _fingerprint()
    for k in ("obs_size", "num_actions"):
        if k in saved and k in cur and saved[k] != cur[k]:
            warnings.warn(
                f"[ckpt] {name}: {k} mismatch (stamp {saved[k]} vs env {cur[k]}) — "
                f"obs/action layout changed since this was trained."
            )
