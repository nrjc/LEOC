import unittest
from typing import Callable, Optional, Tuple
import numpy as np
import tensorflow as tf
from scipy.stats import random_correlation, multivariate_normal
from pilco.controllers import squash_sin, squash_cum_normal


class TestSquash(unittest.TestCase):
    def test_sin_squash(self):
        is_correct = squashing_fn(squash_sin, sin_squash)
        self.assertTrue(is_correct)
        print('sin_squash passes test')

    def test_cum_normal_squash(self):
        is_correct = squashing_fn(squash_cum_normal, cum_normal_squash)
        self.assertTrue(is_correct)
        print('cum_normal_squash passes test')

def squashing_fn(
        squashing_fn: Callable[[np.ndarray, np.ndarray, Optional[float]], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        individual_squashing_fn: Callable[[np.ndarray], np.ndarray]):
    dims = 1
    N = 10000
    means = np.random.uniform(-1, 1, (1, dims))
    if dims > 1:
        cor_eigenval = np.random.uniform(1.0, 20.0, (dims))
        cor_eigenval = cor_eigenval * dims / np.sum(cor_eigenval)  # Scaling to sum to dim
        covariance = random_correlation.rvs(tuple(cor_eigenval))
    else:
        covariance = np.random.uniform(0.0, 10.0, size=(dims, dims))
    M, S, V = squashing_fn(means, covariance)
    sampled_inputs = np.random.multivariate_normal(means.reshape(-1), covariance, N)
    sample_output = [individual_squashing_fn(sample_input) for sample_input in sampled_inputs]
    sample_output = np.stack(sample_output)
    sampled_means = np.mean(sample_output, axis=0, keepdims=True)
    sampled_covariance = np.cov(sample_output.T)
    sampled_input_output_covariance = calculate_covariance(sample_output, sampled_means, sampled_inputs, means, N, dims)
    mean_and_covariance_correct = np.allclose(M, sampled_means, atol=.03) and \
                                  np.allclose(S, sampled_covariance, atol=.03) and \
                                  np.allclose(V, sampled_input_output_covariance, atol=0.03)
    return mean_and_covariance_correct


def calculate_covariance(x, ux, y, uy, N, dims):
    acc = np.zeros((dims, dims), dtype=np.float)
    for i in range(N):
        x_minus_mean, y_minus_mean = x[[i], :] - ux, y[i, :] - uy
        acc += np.matmul(x_minus_mean.T, y_minus_mean)
    return acc / N


def sin_squash(input: np.ndarray) -> np.ndarray:
    return np.sin(input)

def cum_normal_squash(input: np.ndarray) -> np.ndarray:
    return [multivariate_normal.cdf(input)]


if __name__ == '__main__':
    unittest.main()
