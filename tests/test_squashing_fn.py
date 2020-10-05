import unittest
from typing import Callable, Optional, Tuple
import numpy as np
from scipy.stats import random_correlation, multivariate_normal

from pilco.controllers import squash_sin


class TestSquash(unittest.TestCase):
    def test_sin_squash(self):
        is_correct = squashing_fn(squash_sin, sin_squash)
        self.assertTrue(is_correct)


def squashing_fn(
        squashing_fn: Callable[[np.ndarray, np.ndarray, Optional[float]], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        individual_squashing_fn: Callable[[np.ndarray], np.ndarray]):
    # TODO: Test V variable as well.
    dims = 3
    N = 10000
    means = np.random.uniform(-1, 1, (1, dims))
    cor_eigenval = np.random.uniform(1, 20, (dims))
    cor_eigenval = cor_eigenval * dims / np.sum(cor_eigenval)  # Scaling to sum to dim
    covariance = random_correlation.rvs(tuple(cor_eigenval))
    M, S, V = squashing_fn(means, covariance)
    sampled_inputs = np.random.multivariate_normal(means.reshape(-1), covariance, N)
    sample_output = [individual_squashing_fn(sample_input) for sample_input in sampled_inputs]
    sample_output = np.stack(sample_output)
    sampled_means = np.mean(sample_output, axis=0)
    sampled_covariance = np.cov(sample_output.T)
    mean_and_covariance_correct = np.allclose(M, sampled_means, atol=.03) and np.allclose(S, sampled_covariance,
                                                                                          atol=.03)

    return mean_and_covariance_correct


def sin_squash(input: np.ndarray) -> np.ndarray:
    return np.sin(input)


def cum_normal_squash(input: np.ndarray) -> np.ndarray:
    return multivariate_normal.cdf(input)


if __name__ == '__main__':
    unittest.main()
