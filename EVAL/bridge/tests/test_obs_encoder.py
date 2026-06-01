"""Basic structural tests for obs_encoder. Full obs-identity check
(encoder vs fastcatan mirror) lives in test_obs_identity.py once the
state-mirror is wired up in Phase 4."""

from __future__ import annotations

import numpy as np
import pytest

from catanatron import Color
from catanatron.game import Game
from catanatron.models.player import RandomPlayer

import fastcatan
from bridge.obs_encoder import encode_obs


COLORS = [Color.RED, Color.WHITE, Color.BLUE, Color.ORANGE]


@pytest.fixture
def fresh_game():
    return Game([RandomPlayer(c) for c in COLORS], seed=42)


@pytest.mark.parametrize("color", COLORS)
def test_shape(fresh_game, color):
    obs = encode_obs(fresh_game, color)
    assert obs.shape == (fastcatan.OBS_SIZE,)
    assert obs.dtype == np.float32


def test_no_nan_inf(fresh_game):
    obs = encode_obs(fresh_game, Color.RED)
    assert np.isfinite(obs).all()


def test_initial_state_has_known_values(fresh_game):
    obs = encode_obs(fresh_game, Color.RED)
    # Obs count fields are normalized by structural maxima (see obs_encoder
    # N_* / src/catan/obs.cpp norm::). At game start settle/city/road are at
    # their maxima, so each normalizes to exactly 1.0.
    # Self block: index 0 = VP (0 at start, 0/10 = 0)
    assert obs[0] == 0.0
    # Self block: index 5 = settle_left (5/5 = 1.0)
    assert obs[5] == 1.0
    # Self block: index 6 = city_left (4/4 = 1.0)
    assert obs[6] == 1.0
    # Self block: index 7 = road_left (15/15 = 1.0)
    assert obs[7] == 1.0
    # Self block: discard_remaining (0), is_current (1 if RED's turn first)
    # Self block layout: [vp, hand, dev, knights, road_len, settle, city, road,
    #                     ports*6, discard, is_current]
    is_current_idx = 0 + 8 + 6 + 1  # offsets in self block
    # RED is first in catanatron's default Game; on RED's POV, is_current=1
    assert obs[is_current_idx] == 1.0


def test_runs_through_many_steps(fresh_game):
    """Encoder must survive a partial game's worth of states without error."""
    for _ in range(200):
        for c in COLORS:
            obs = encode_obs(fresh_game, c)
            assert obs.shape == (fastcatan.OBS_SIZE,)
            assert np.isfinite(obs).all()
        fresh_game.play_tick()


def test_pov_relative_changes_with_pov(fresh_game):
    """Same state should produce different obs from different POVs."""
    # Play a bit to differentiate players
    for _ in range(30):
        fresh_game.play_tick()
    obs_red = encode_obs(fresh_game, Color.RED)
    obs_blue = encode_obs(fresh_game, Color.BLUE)
    # They should differ in at least the per-player block region (first 64
    # floats), since POV reorders the block sequence.
    assert not np.array_equal(obs_red[:64], obs_blue[:64])


def test_static_board_blocks_match_across_pov(fresh_game):
    """Hex resources / numbers / port types are absolute (not POV-relative),
    so they must match across different POVs.
    Layout offsets:
      per-player: 64
      self private: 16
      nodes: 432
      edges: 288
      hex res start: 64 + 16 + 432 + 288 = 800
      hex num start: 800 + 6*19 = 914
      port types start: 914 + 19 = 933
      port types end: 933 + 9*6 = 987
    """
    obs_red = encode_obs(fresh_game, Color.RED)
    obs_blue = encode_obs(fresh_game, Color.BLUE)
    # Hex resources + numbers + ports (absolute) should match.
    assert np.array_equal(obs_red[800:987], obs_blue[800:987])
