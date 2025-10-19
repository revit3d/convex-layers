import pytest
import numpy as np

from convex_layers_naive import NaiveHullBuilder
from convex_layers_tree import HullTree
from geometry import Point


def check_layers_equal(points: list[Point]):
    layers_naive = NaiveHullBuilder().compute_layers(points)
    layers_fast = HullTree().compute_layers(points)

    layers_naive = [sorted(layer) for layer in layers_naive]
    layers_fast = [sorted(layer) for layer in layers_fast]

    assert len(layers_naive) == len(layers_fast), (
        f"Number of layers differ: {len(layers_naive)} != {len(layers_fast)}\n"
        f"{layers_naive}\n"
        f"{layers_fast}"
    )
    for i in range(len(layers_naive)):
        for pt_naive, pt_fast in zip(layers_naive[i], layers_fast[i]):
            assert np.isclose(pt_naive.x, pt_fast.x) and np.isclose(pt_naive.y, pt_fast.y), (
                f"Layer {i} differs: {layers_naive[i]} != {layers_fast[i]}"
            )


@pytest.fixture
def distribution_gen_func():
    return {
        "uniform": lambda low, high, s: np.random.rand(s) * high + low,
        "normal": lambda low, high, s: np.random.randn(s) * high + low,
        "uniform_int": np.random.randint,
    }


@pytest.fixture
def n_trials():
    # n_points -> n_trials
    return {
        10: 100_000,
        30: 10_000,
        100: 10_000,
        1000: 1000,
        10_000: 100,
    }


@pytest.mark.parametrize("n_points", [10, 30, 100, 1000, 10_000])
@pytest.mark.parametrize("distribution_type", ["uniform_int", "uniform", "normal"])
@pytest.mark.parametrize("limits", [(0, 100), (0, 10**17), (-100, 100), (-10**17, 10**17)])
def test_convex_layers_algorithms(n_points, distribution_type, limits, distribution_gen_func, n_trials):
    np.random.seed(42)

    seeds = np.random.randint(0, 100_000, size=n_trials[n_points])
    for seed in seeds:
        np.random.seed(seed)

        gen_func = distribution_gen_func[distribution_type]
        low, high = limits
        xs = gen_func(low, high, n_points).astype(float)
        ys = gen_func(low, high, n_points).astype(float)
        points = [Point(xs[i], ys[i]) for i in range(n_points)]

        check_layers_equal(points)
