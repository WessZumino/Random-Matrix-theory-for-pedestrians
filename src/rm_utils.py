"""
MIT License

Copyright (c) 2020 Mirco Milletarì

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


set of utility functions for random matrix evaluations
Author: Mirco Milletari' <milletari@gmail.com>
"""
import numpy as np
from numpy.linalg import eigvalsh


def normal(z, s):
    """Gaussian distribution centered in 0
       with variance s.

       Args:
           z (np.array, float): an array of float values. 
           s (float): the variance of the distribution.

       Returns:
           the normal distribution N(0,s)
    """

    return 1 / (s * (2 * np.pi) ** (0.5)) * np.exp(-(z ** 2) / (2 * s ** 2))


def rm_sampling(n_samples, N, rng):
    """Sample a symmetric RM from the normal
       distribution N(0, sqrt(N)).

    Args:
        n_samples (int): number of sampling steps.
        N (int): the size of the RM.
        rng (np ufunc): random number generator
    """

    lambdas = []
    sigma = N ** (-0.5)

    for _ in range(n_samples):
        # get an instance of X
        X = rng.normal(scale=sigma, size=(N, N))
        # symmetrise X
        Xs = (X + X.T) / 2 ** (0.5)
        # record the eigenvalues
        lambdas.append(eigvalsh(Xs).astype(np.float32))

    lambdas = np.array(lambdas)

    # return the flattened array
    return lambdas.flatten()


def wigner(l):
    """Generate the Wigner semicirlce law
       for the eigenvalue distribution rho(l)
       in the Gaussian Orthogonal Ensemble.

    Args:
        l (np array, float32): value range of
            the eigenvalues.
    Returns:
        rho (np array, float32): eigenvalue
            distribution.
    """

    rho = np.sqrt(4 - l ** 2) / (2 * np.pi)

    return rho.astype(np.float32)


def rmse(numerical, expected):
    """Evaluate the Root Mean square Error

    Args:
       numerical (np array, float32): numerical
           value of eigenvalue distribution.
        expected (np array, float32): values
            from the wigner semicircle.

    Returns:
        rmse (float32): mean squareroot error.
    """

    err = np.square(numerical - expected).mean()

    return np.sqrt(err)


def get_bulk_edge_values(counts, bins):
    """Separate the bulk from the edge values
       The bulk is defined according to the semicircle
       law: lambda in [-2, 2].

    Args:
        counts (np array, float32): eigenvalue density.
        bins (np array, float32): eigenvalues.       

    Returns:
        bulk_rho (np array, float32): bulk eigenvalue density.
        tails_tho (np array, float32): tails eigenvalue density.
        bulk_lambdas (np array, float32): bulk eigenvalues.
        tails_lambdas (np array, float32): tails eigenvalues.
    """
    # get the right edge position
    right = bins[:-1] <= 2.0
    # get the left edge position
    left = bins[:-1] >= -2.0

    # Get the bulk and tails mask
    bulk_mask = right * left
    tails_mask = ~bulk_mask

    # get the bulk and edge values

    bulk_rho = counts[bulk_mask]
    tails_rho = counts[tails_mask]

    bulk_lambdas = bins[:-1][bulk_mask]
    tails_lambdas = bins[:-1][tails_mask]

    return bulk_rho, bulk_lambdas, tails_rho, tails_lambdas
