import numpy as np
from numpy.typing import ArrayLike


def uniform_weights(winrates: ArrayLike) -> np.ndarray:
    weights = np.ones_like(winrates)
    weights /= weights.sum()
    return weights


def calculate_mean(winrates: ArrayLike, weights: np.ndarray | None = None):
    p_x = np.array(winrates)

    if weights is None:
        weights = uniform_weights(winrates)

    if not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must sum to 1")

    win_rate = (weights * p_x).sum()
    return win_rate


def calculate_var(winrates: ArrayLike, shots: int, weights: np.ndarray | None = None):
    p_x = np.array(winrates)

    if weights is None:
        weights = uniform_weights(winrates)

    if not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must sum to 1")

    win_rate_var = (weights**2 * p_x * (1 - p_x) / shots).sum()
    return win_rate_var


def calculate_ci(
    winrates: ArrayLike,
    shots: int,
    d: float = 0.05,
    weights: np.ndarray | None = None,
    var: float | None = None,
):
    if weights is None:
        weights = uniform_weights(winrates)

    if not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must sum to 1")

    constant = 1 / shots

    if var is None:
        var = calculate_var(winrates, shots, weights)

    term1 = 2 * constant * np.log(2 / d) / 3
    term2 = 2 * np.log(2 / d)
    return term1 + np.sqrt(var * term2)


def calculate_p_value(
    winrates: ArrayLike,
    shots: int,
    omega_c: float,
    weights: np.ndarray | None = None,
    var: float | None = None,
):
    if weights is None:
        weights = uniform_weights(winrates)

    if var is None:
        var = calculate_var(winrates, shots, weights)

    constant = 1 / shots
    mean_wr = calculate_mean(winrates, weights)
    eps_c = mean_wr - omega_c

    if eps_c < 0:
        return 1

    numerator = 0.5 * eps_c**2
    denominator = var + constant * eps_c / 3
    return np.exp(-numerator / denominator)
