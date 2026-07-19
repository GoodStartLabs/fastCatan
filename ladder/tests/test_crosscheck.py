from ladder.crosscheck_bridge import intervals_overlap


def test_interval_overlap_includes_touching_bounds() -> None:
    assert intervals_overlap((0.1, 0.2), (0.2, 0.4))
    assert not intervals_overlap((0.1, 0.19), (0.2, 0.4))
