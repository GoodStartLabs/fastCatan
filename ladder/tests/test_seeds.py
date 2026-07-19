from ladder.seeds import derive_board_seed


def test_board_seed_is_stable_and_indexed() -> None:
    master = 0x6A09E667F3BCC909
    seeds = [derive_board_seed(master, index) for index in range(8)]
    assert seeds == [derive_board_seed(master, index) for index in range(8)]
    assert len(set(seeds)) == len(seeds)
    assert seeds[:3] == [
        0x63CFC62A2B097592,
        0x0FB1000633E9EC55,
        0xF94F589A714B0DA3,
    ]


def test_board_seed_rejects_negative_index() -> None:
    try:
        derive_board_seed(0, -1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative game index was accepted")
