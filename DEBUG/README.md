# DEBUG/ — game-log viewer + benchmarks

Debug/inspection tooling, kept out of the core tree. Nothing here is imported by
the engine, training, or eval at runtime (except the bridge tests, which import
`ui.obs_layout` / `ui.obs_decoder` to check obs parity).

| Path | Role |
|---|---|
| `ui/` | game-log viewer: `replay.py` (interactive), `recorder.py` (record a game), board/state/mask renderers, obs decoder. Package name stays `ui`. |
| `bench/` | throughput benchmarks — C++ floor (`bench_step.cpp`, `bench_batched.cpp`, built by the standalone CMake build) + Python harnesses (`bench_throughput.py`, `bench_comprehensive.py`). |
| `logs/` | **drop folder** — put `*.jsonl.gz` game logs here; `replay.py` resolves a bare filename against it. |

`import ui` resolves because `DEBUG` is on the pytest `pythonpath` (see
`pyproject.toml`); for the CLIs, run from the repo root with `PYTHONPATH=DEBUG`.

## Workflow — record, drop, view

```bash
# record a random-vs-random game into the drop folder
PYTHONPATH=DEBUG python3 -m ui.recorder --seed 42 --out DEBUG/logs/game.jsonl.gz

# view it (bare name resolves against DEBUG/logs/)
PYTHONPATH=DEBUG python3 -m ui.replay game.jsonl.gz
PYTHONPATH=DEBUG python3 -m ui.replay game.jsonl.gz --auto --delay 0.2   # autoplay
PYTHONPATH=DEBUG python3 -m ui.replay game.jsonl.gz --out frames/        # dump PNGs

# just drag any .jsonl.gz into DEBUG/logs/ then replay it by name.
```

## Benchmarks

```bash
# Python throughput (needs fastcatan + catanatron + bridge under EVAL/):
PYTHONPATH=DEBUG python3 DEBUG/bench/bench_throughput.py

# C++ floor (standalone CMake build only, not the pip wheel):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
build/bench_step
build/bench_batched 4096 5000
```
