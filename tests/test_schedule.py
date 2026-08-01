import pytest

from pimf import make_window_schedule


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (200, [101, 71, 51, 37, 31]),
        (500, [251, 177, 125, 89, 63, 45, 31]),
        (1000, [501, 355, 251, 177, 125, 89, 63, 45, 31]),
        (2000, [1001, 707, 499, 353, 249, 177, 125, 89, 63, 45, 31]),
    ],
)
def test_research_fixtures(n, expected):
    assert make_window_schedule(n) == expected


@pytest.mark.parametrize("n", [50, 200, 777, 1000, 4096])
def test_invariants(n):
    sizes = make_window_schedule(n)
    assert all(size % 2 == 1 for size in sizes)
    assert all(a > b for a, b in zip(sizes, sizes[1:], strict=False))
    assert sizes[0] <= n


def test_tiny_signal_returns_single_window():
    assert make_window_schedule(10) == [5]


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        make_window_schedule(0)
    with pytest.raises(ValueError):
        make_window_schedule(1000, factor=1.0)
