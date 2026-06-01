"""Game-log file format: gzipped JSONL.

Layout
------
Line 1   : header           {"kind":"header", v, engine, obs, mask, nactions,
                              seed, players, snap_every}
Line 2+  : step records     {"kind":"step", i, t, ph, fl, cp, dr, a, r, d,
                              snap (optional b64)}
Last line: final            {"kind":"final", winner, vps, steps, turns}

`snap` is base64 of `env.snapshot()`. Recorded every `snap_every` steps and
always on the final step. Replayer seeks by walking from the nearest snapshot
and stepping forward.

Versioning: bump `v` on any breaking schema change. Readers MUST check.
"""

from __future__ import annotations

import base64
import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Iterator

LOG_VERSION = 1


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class LogHeader:
    v: int
    engine: str
    obs: int
    mask: int
    nactions: int
    seed: int
    players: list[str]
    snap_every: int
    kind: str = "header"


@dataclass
class StepRecord:
    i: int        # step index (0-based)
    t: int        # turn_count at start of step
    ph: int       # phase
    fl: int       # flag
    cp: int       # current_player (absolute seat)
    dr: int       # dice_roll (0 if not rolled yet)
    a: int        # action ID applied this step
    r: float      # reward returned by env.step
    d: int        # done flag (0/1)
    snap: str | None = None  # base64 snapshot, may be None
    kind: str = "step"


@dataclass
class FinalRecord:
    winner: int          # -1 if no winner inside step cap
    vps: list[int]
    steps: int
    turns: int
    kind: str = "final"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class LogWriter:
    """Streaming JSONL.gz writer. Use as a context manager."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh: IO[bytes] | None = None

    def __enter__(self) -> "LogWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = gzip.open(self.path, "wb")
        return self

    def __exit__(self, *_exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _emit(self, obj: dict) -> None:
        assert self._fh is not None, "writer not open"
        self._fh.write(json.dumps(obj, separators=(",", ":")).encode())
        self._fh.write(b"\n")

    def write_header(self, h: LogHeader) -> None:
        self._emit(asdict(h))

    def write_step(self, s: StepRecord) -> None:
        d = asdict(s)
        if d["snap"] is None:
            d.pop("snap")
        self._emit(d)

    def write_final(self, f: FinalRecord) -> None:
        self._emit(asdict(f))


def encode_snap(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def decode_snap(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

@dataclass
class GameLog:
    header: LogHeader
    steps: list[StepRecord] = field(default_factory=list)
    final: FinalRecord | None = None

    def nearest_snapshot(self, step_idx: int) -> tuple[int, bytes] | None:
        """Find the latest snapshot at or before `step_idx`. Returns
        (snap_step_idx, raw_bytes) or None if no snapshot precedes it."""
        snap_step = -1
        snap_raw: bytes | None = None
        for s in self.steps[: step_idx + 1]:
            if s.snap is not None:
                snap_step = s.i
                snap_raw = decode_snap(s.snap)
        if snap_raw is None:
            return None
        return snap_step, snap_raw


def _parse_record(obj: dict) -> LogHeader | StepRecord | FinalRecord:
    k = obj.get("kind")
    if k == "header":
        return LogHeader(**{kk: vv for kk, vv in obj.items() if kk != "kind"},
                         kind="header")
    if k == "step":
        return StepRecord(
            i=obj["i"], t=obj["t"], ph=obj["ph"], fl=obj["fl"],
            cp=obj["cp"], dr=obj["dr"], a=obj["a"],
            r=obj["r"], d=obj["d"],
            snap=obj.get("snap"),
        )
    if k == "final":
        return FinalRecord(
            winner=obj["winner"], vps=list(obj["vps"]),
            steps=obj["steps"], turns=obj["turns"],
        )
    raise ValueError(f"unknown log record kind: {k!r}")


def iter_log(path: str | Path) -> Iterator[LogHeader | StepRecord | FinalRecord]:
    """Stream-read a log without loading the full game into memory."""
    with gzip.open(Path(path), "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield _parse_record(json.loads(line))


def read_log(path: str | Path) -> GameLog:
    """Eager-load an entire game log into memory."""
    header: LogHeader | None = None
    steps: list[StepRecord] = []
    final: FinalRecord | None = None
    for rec in iter_log(path):
        if isinstance(rec, LogHeader):
            if header is not None:
                raise ValueError("duplicate header in log")
            header = rec
        elif isinstance(rec, StepRecord):
            steps.append(rec)
        elif isinstance(rec, FinalRecord):
            final = rec
    if header is None:
        raise ValueError("log missing header")
    if header.v != LOG_VERSION:
        raise ValueError(
            f"log version {header.v} != expected {LOG_VERSION} — "
            "update ui.log_format or re-record"
        )
    return GameLog(header=header, steps=steps, final=final)
