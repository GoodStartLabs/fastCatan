# bridge/ — Catanatron interop + cross-engine correctness

Catanatron is the **ground-truth oracle**. This package (a) lets a fastcatan-
trained policy play inside Catanatron for thesis eval (M4), and (b) proves
fastcatan plays Catan identically to Catanatron (the M1 correctness gate).

Resource order differs between engines and is permuted everywhere:
fastcatan `[brick, lumber, wool, grain, ore]` vs Catanatron `RESOURCES`
`[WOOD, BRICK, SHEEP, WHEAT, ORE]`. Catanatron also **shuffles seat order** —
`state.color_to_index` is a per-game permutation, not player-list order; index
per-player arrays by fixed `Color`.

## Files

| File | Role |
|------|------|
| `topology_map.py` | node/edge/hex/port ID maps fastcatan ↔ Catanatron (bijections, tested) |
| `action_codec.py` | bidirectional action translation; `encode_to_fast_ids(cat_action)` → fastcatan ID(s) |
| `obs_encoder.py` | Catanatron state → fastcatan obs vector (the **eval** encoder). Mirrors `src/catan/obs.cpp` layout AND normalization (`N_*` divisors) |
| `catanatron_bridge.py` | a Catanatron `Player` that encodes obs → runs a policy → decodes the action |
| `run_eval.py` | eval driver (bridge vs Catanatron bots) |
| `state_mirror.py` | byte-exact ctypes mirror of the 384-B C++ `GameState`+`BoardLayout` |
| `state_inject.py` | serialize a Catanatron state into a fastcatan `GameState` (for `load_snapshot`) |
| `rng_force.py` | bit-exact xoshiro128++ replica; find an RNG state forcing a target `bounded(N)` (dice/dev/steal) |

## Differential test (the M1 gate)

`tests/test_differential.py` co-steps **both** engines through the same action
stream and asserts full state parity every ply:

```
for each Catanatron ply:
    inject Catanatron PRE-state into fastcatan (state_inject)
    translate the action (action_codec) + force RNG (rng_force) so a
        stochastic draw matches Catanatron's outcome
    env.step(...)                       # fastcatan
    assert fast post-state == Catanatron post-state (seat-absolute fields)
```

`tests/test_obs_identity.py` then checks the two obs encoders agree: inject a
state, C++ `write_obs` vs `obs_encoder.encode_obs`, bit-for-bit (from the
current-player POV — the only POV ever consumed). This is what `test_obs_*`
referred to as the "obs-identity check once the state mirror is wired up".

**Reproducibility:** pin `random.seed`, `np.random.seed`, AND `PYTHONHASHSEED`.
Catanatron's `RandomPlayer` (`random.choice`) and set-iteration order all
matter; without all three the corpus differs run-to-run.

## Bugs this harness found + fixed (in the C++ core)

1. Longest-road title off-by-one — first player to reach exactly 5 roads got no
   title (+2 VP).
2. Production bank-shortage — pay nobody when total demand for a resource
   exceeds the bank (no partial to a sole recipient).
3. Road buildability through an enemy-occupied node the player reached first.
4. History-dependent longest-road membership → added `road_node_member` to
   `GameState`.
5. Obs `trade_responses` no-trade mismatch (PENDING vs N/A).

## Known parity bound

Catanatron's longest road is internally inconsistent on **road cuts** (cached
lengths + history-dependent component membership + lowest-seat tie-break).
fastcatan is rule-correct; the ≤1–2% residual (random games, cuts only) is
exempted in `test_differential` (`_exempt_lr_tie_quirk`, and the road-length
exemption that fires only when fast disagrees with a *fresh* Catanatron
recompute). Porting Catanatron's component state machine for bit-parity is not
worth it.

## Gotcha: obs normalization must stay in sync

`obs_encoder.py` `N_*` divisors MUST equal `src/catan/obs.cpp` `namespace norm`
and `ui/obs_decoder.py` `N_*`. `test_obs_identity` guards this.
