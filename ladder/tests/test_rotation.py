from ladder.match import rotate


def test_four_cyclic_rotations_put_candidate_in_each_seat() -> None:
    base = ["candidate", "opponent", "opponent", "opponent"]
    candidate_seats = [rotate(base, shift).index("candidate") for shift in range(4)]
    assert candidate_seats == [0, 1, 2, 3]
