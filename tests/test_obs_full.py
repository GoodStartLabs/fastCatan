"""write_obs_full: POV prefix bit-identical to write_obs; appendix consistent.

The appendix is the hidden-enemy state the POV obs masks out — per opponent
(relseat +1..+3): resources(5)/19, dev_playable(5)/10, dev_bought(5)/10,
hidden dev VP /10. The learned judge trains and evaluates on it; POV
consumers slice the [:OBS_SIZE] prefix, so prefix identity is load-bearing.
"""
from __future__ import annotations

import numpy as np
import pytest

import fastcatan
from tests.conftest import play_random_game

RES_NORM = 19.0


def test_full_obs_size_constant():
    assert fastcatan.OBS_FULL_SIZE == fastcatan.OBS_SIZE + 3 * 16


@pytest.mark.parametrize("seed", [3, 99, 777])
def test_prefix_identical_and_appendix_consistent(seed):
    obs = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)
    full = np.zeros(fastcatan.OBS_FULL_SIZE, dtype=np.float32)
    checked = {"n": 0, "nonzero": 0}

    def on_step(env, mask_buf, action, step_idx):
        if step_idx % 50:
            return
        for pov in range(4):
            env.write_obs(pov, obs)
            env.write_obs_full(pov, full)
            assert np.array_equal(obs, full[: fastcatan.OBS_SIZE])
            app = full[fastcatan.OBS_SIZE:].reshape(3, 16)
            for rel in range(3):
                pl = (pov + rel + 1) % 4
                # appendix resource sum must equal the PUBLIC handsize
                assert abs(app[rel, :5].sum() * RES_NORM
                           - env.player_handsize(pl)) < 1e-3
            checked["n"] += 1
            if full[fastcatan.OBS_SIZE:].any():
                checked["nonzero"] += 1

    play_random_game(seed, on_step=on_step)
    assert checked["n"] > 0
    assert checked["nonzero"] > 0      # hidden state actually flows through
