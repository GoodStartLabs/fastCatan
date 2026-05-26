# `ui/` — Visual Debugger & Replay

Visual layer for fastcatan. Purpose: human-readable rendering of board, obs
vector, and legal-action mask. Used for debugging the engine, debugging the
bridge, and (later) explaining AI decisions.

Non-goal: a real-time playable client. This is a *post-hoc* replayer driven by
recorded game logs, plus a thin live-step mode for ad-hoc inspection.

---

## Inputs / data sources

All data comes from the existing engine surface — no new C++ needed.

- `fastcatan.Env`: `phase`, `flag`, `current_player`, `dice_roll`, `turn_count`,
  per-seat accessors (`player_vp`, `player_handsize`, `player_road_length`,
  `player_ports`, `player_resource`, `bank`, `longest_road_owner`,
  `largest_army_owner`).
- `env.write_obs(pov, buf)` → float32[`OBS_SIZE`]. POV-relative encoding.
  Layout documented in `src/catan/obs.cpp` and mirrored in
  `bridge/obs_encoder.py`. Count fields are **normalized** by structural Catan
  maxima (`obs.cpp` `namespace norm`); `obs_decoder.py` de-normalizes them on
  decode (its `N_*` divisors MUST match obs.cpp / obs_encoder.py).
- `env.action_mask(buf)` → uint64[`MASK_WORDS`]. Bit `i` of word `w` = action
  `w*64 + i` legal.
- `env.snapshot()` / `load_snapshot(bytes)` → exact state round-trip. Used by
  the replayer to fast-seek without replaying every step.
- `fastcatan.action.*`: flat-ID layout (`SETTLE_BASE`, `ROAD_BASE`, …).
- Topology geometry: reuse `visual/viz_topology.py` (parses
  `include/topology.hpp`, computes hex/node/edge positions).

The board state needed for drawing (settlements, cities, roads, robber, hex
resources, hex numbers, ports) is **fully recoverable from the obs vector**
of any POV — no new C++ accessors required. The obs decoder (M2) extracts it.

---

## Components (in `ui/`)

```
ui/
  PLAN.md              ← this file
  obs_layout.py        ← M2: named slice ranges over the float32 obs vector
  obs_decoder.py       ← M2: obs → structured dict (board, hands, meta)
  action_names.py      ← M6: pretty-printer for flat action IDs
  mask_view.py         ← M4: mask → list[(id, name)] + per-category indices
  board_render.py      ← M3: matplotlib draw of board + state overlay
  state_panel.py       ← M3/M4: side panel (hands, VP, dev deck, bank, mask)
  log_format.py        ← M1: schema for game-log files (read+write)
  recorder.py          ← M1: drop-in wrapper around the Env step loop that
                         writes a log
  replay.py            ← M5: CLI entry — load a log, walk it interactively or
                         autoplay; emits per-step rendering
  __init__.py
```

No subdirs unless we add a JS/web frontend later (out of scope for v1).

---

## Milestones

> Status: M0–M6 implementation landed (`recorder.py`, `obs_layout.py`,
> `obs_decoder.py`, `board_render.py`, `state_panel.py`, `mask_view.py`,
> `replay.py`, `action_names.py`, `log_format.py`, `geometry.py` all
> present). Remaining work is **debugging** the renderer + replayer against
> recorded games and verifying obs-decoder ↔ engine field parity.

### M0 — Plan (this commit) — DONE
- `PLAN.md` written. No code.

### M1 — Log format + recorder — DONE
- **Format**: gzipped JSONL or compressed `.npz`. Decide via prototype; start
  with **JSONL+gzip** (human-diffable, easy to grep, no schema lock-in). One
  record per env step. Header record first with seed, engine version, mask
  size, obs size, action layout snapshot.

  Per-step record (all keys flat to keep the JSONL small):
  ```
  {
    "i":   step_idx,
    "t":   turn_count,
    "ph":  phase, "fl": flag,
    "cp":  current_player,
    "dr":  dice_roll,
    "a":   action_id,
    "r":   reward,
    "d":   done,
    "snap": "<base64 of env.snapshot()>"   // every K steps; or every step
  }
  ```
  Obs vectors and masks are **not** stored — they are derivable from
  `snap` (cheap to recompute). Storing snapshots every step is ~2× cheaper
  than storing obs+mask and lets the replayer seek by index.

  Alt: store `snap` only every K=10 steps + replay forward from the
  nearest snapshot. Keep configurable.

- **`recorder.py`**: a function `record_game(seed, players, out_path, ...)`
  mirroring `examples/random_player_test.py::play_one` but emitting the log.
  Should be a near-drop-in so the existing test harness can opt in via flag.

### M2 — Obs decoder — DONE (needs cross-check vs `bridge/obs_encoder.py`)
- **`obs_layout.py`**: constants for every named slice of the obs vector
  (`PLAYER_BLOCKS = slice(0, 4*15)`, `SELF_PRIVATE = …`, `NODES = …`,
  `EDGES = …`, `HEXES = …`, `HEX_NUMS = …`, `PORTS = …`, `ROBBER = …`,
  `PHASE = …`, `FLAG = …`, `LAST_ROLL = …`, `BANK = …`, `DEV_DECK = …`,
  `LR_OWNER`, `LA_OWNER`, `START_PLAYER`, `FREE_ROADS`, `TRADE_SCRATCH = …`).
  Single source of truth for both decoder and any obs-slot viewer.

  Cross-check against `bridge/obs_encoder.py` order (it is the canonical
  Python mirror) and `src/catan/obs.cpp` (the C++ writer). Add an assertion
  test that the sum of slice widths equals `fastcatan.OBS_SIZE`.

- **`obs_decoder.py`**: `decode(obs: np.ndarray, pov: int) -> BoardView`
  where `BoardView` is a dataclass holding:
  - `nodes[54]`: `(owner_relseat or None, "settlement"/"city" or None)`
  - `edges[72]`: `owner_relseat or None`
  - `hex_resources[19]`, `hex_numbers[19]`, `port_types[9]`, `robber_hex`
  - `phase`, `flag`, `last_roll`, `bank[5]`, `dev_deck[5]`,
    `lr_owner_rel`, `la_owner_rel`, `start_player_rel`, `free_roads`
  - `self_hand[5]`, `self_dev_playable[5]`, `self_dev_pending[5]`,
    `self_dev_played_flag`
  - `player_blocks[4]`: `vp, handsize, total_dev, knights, road_len,
    settle_left, city_left, road_left, ports_bits[6], discard_left,
    is_current`
  - `trade`: `proposer_rel`, `give[5]`, `want[5]`, `responses[3]`

  Relseats are kept (POV-relative). The renderer can map back to absolute
  seats via the recorded `current_player` from the log.

### M3 — Board renderer — DONE (visual debug pass outstanding)
- **`board_render.py`**: a function `draw_board(ax, view: BoardView,
  *, options) -> None` that paints onto a matplotlib axis:
  - Hex polygons coloured by resource (brick=red, lumber=green, wool=light
    green, grain=yellow, ore=grey, desert=tan). Number token in the centre,
    bold for 6/8.
  - Settlements as small filled triangles at node positions; cities as small
    filled squares. Coloured by absolute seat (mapped from relseat at draw
    time). Empty nodes faint dots.
  - Roads as thick line segments along their edge midpoints. Coloured by
    seat.
  - Robber as a black hex outline (or small black disk) overlaid on its hex.
  - Ports as wedges on coastal nodes labelled `2:1 wool` / `3:1`.
  - Toggle: ID labels on/off (reuse `visual/viz_topology.py` style).

- **`state_panel.py`**: text panel beside the board with per-player VP /
  hand / road length / knights / ports / discard-owed, plus bank,
  dev deck, last roll, phase/flag.

### M4 — Mask viewer — DONE
- **`mask_view.py`**: `decode_mask(mask: np.ndarray) -> dict[str, list[int]]`
  bucketing legal action IDs by category (`settle`, `city`, `road`,
  `move_robber`, `steal`, `discard`, `trade_bank`, `trade_p2p`, `dev_play`,
  `dev_buy`, `roll`, `end_turn`).
- The renderer highlights legal *spatial* actions on the board:
  legal settle nodes get a green halo, legal road edges a green tint, legal
  robber hexes a red outline. Non-spatial legal actions appear as a chip
  list in the side panel ("ROLL", "END_TURN", "PLAY_KNIGHT", …).

### M5 — Replayer CLI — DONE (frames in `logs/frames/`, still debugging)
- **`replay.py`**: entry-point.
  ```
  python -m ui.replay path/to/game.jsonl.gz
      --start N           # seek to step N
      --auto              # autoplay
      --delay 0.5         # seconds between frames in --auto
      --pov 0             # which player's POV obs to decode (default = step's cp)
      --out frames/       # write PNGs instead of opening a window
      --no-mask           # skip mask overlay
  ```
- Interactive (when no `--out`): matplotlib window with key bindings —
  `Right`/`Space` next, `Left` prev, `Home`/`End` start/end, `a` toggle auto,
  `m` toggle mask overlay, `i` toggle ID labels, `1-4` switch POV, `q` quit.
  Implementation: matplotlib `key_press_event` on the figure. No new deps.

### M6 — Action pretty-printer — DONE
- **`action_names.py`**: `name(action_id: int) -> str` and
  `describe(action_id) -> (category, payload)`. Examples: `36` →
  `"SETTLE @ node 0x24"`, `198` → `"ROAD @ edge 0x06"`. Drives:
  - per-step log header ("step 142 — P1 ROAD @ edge 0x12, r=0, d=0")
  - mask side-panel chip labels.
- Symmetric to `bridge/action_codec.py::encode_to_fast_ids` but for human
  consumption only.

---

## Open questions to revisit after M1

- **Replay seeking strategy**: snapshot-every-step (~few KB/step) vs. every
  K + replay-forward. JSONL size matters for `bench/results/*` — many games
  will be logged. Start with every step; profile; switch if needed.
- **Catanatron-side recording**: same log format from the bridge? The bridge
  has its own action stream; we'd record fastcatan IDs from `rep_map` and
  reuse the same renderer. Defer to M5+.
- **Action IDs in mask but illegal at runtime**: the engine guarantees these
  don't exist, but the renderer should fail loudly if `mask` and the
  recorded action disagree at replay time.

---

## Order of work

M0 → M1 → M2 → M3 → (M6 in parallel with M3, trivial) → M4 → M5.

Each milestone leaves a usable artifact:
- After M1: a `.jsonl.gz` of a real game.
- After M2: dump structured board view to stdout.
- After M3: static PNG of a single step.
- After M4: PNG with mask overlay.
- After M5: interactive replay.
