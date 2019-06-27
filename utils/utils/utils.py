# Copyright 2019 Alex Judge
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Data analysis utilities.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression



def interp_ds(ds, xyi):
    """
    Interpolate data set containing one or more 2D regular lon-lat data grids.

    Parameters
    ----------
    ds : Dataset
        Data set with p data variables.
    xyi : array, shape=(N, 2)
        Interpolation points in lon-lat coordinates with longitude in
        [-180, 180).

    Returns
    -------
    zi : array, shape=(N, p)
        Data values at interpolation points.
    """
    # Extract coordinates and variable grids.
    x, y = [ds.coords[k].values for k in ds.coords]
    var_keys = list(ds.data_vars.keys())
    grids = [ds.data_vars[k].values for k in var_keys]
    # Interpolate grids.
    zi = np.vstack([interp_grid(x, y, z, xyi) for z in grids]).T
    return zi, var_keys


def interp_grid(x, y, z, xyi):
    """
    Interpolate 2D regular lon-lat grid data.

    Parameters
    ----------
    x : array, shape=(m,)
        X-axis grid coordinate values.
    y : array, shape=(n,)
        Y-axis grid coordinate values.
    z : array, shape=(n, m)
        Grid values.
    xyi : array, shape=(N, 2)
        Interpolation points in lon-lat coordinates with longitude in
        [-180, 180).

    Returns
    -------
    zi : array, shape=(N,)
        Data values at interpolation points.

    """
    # Make sure the coordinates are ordered.
    if (np.sort(x) != x).any():
        ix = np.argsort(x)
        x = x[ix]
        z = z[:, ix]
    # Shift longitude to [-180, 180) if necessary.
    if x.max() > 180:
        if x.min() == 0 and x.max() == 360:
            x = x[:-1]
            z = z[:, :-1]
        mask = x >= 180
        x[mask] = x[mask] - 360
        x = np.roll(x, mask.sum())
        z = np.roll(z, mask.sum(), axis=1)
    # Hack to deal with null values on grid edges.
    x = x[1:-1]
    z = z[:, 1:-1]
    # Interpolate grid.
    interp_func = RegularGridInterpolator(
        (y, x), z, method='nearest', bounds_error=False, fill_value=None
    )
    zi = interp_func(np.fliplr(xyi))
    return zi


def fib_lattice(n):
    """
    Generate a Fibonacci lattice on the sphere.

    Parameters
    ----------
    n : int
        Lattice size. Must be odd number.

    Returns
    -------
    xy : array, shape=(n, 2)
        Lattice points in lon-lat coordinates with longitude in [-180, 180).
    """
    # Setup.
    assert np.mod(n, 2) == 1, 'Lattice size must be odd.'
    m = int((n - 1) / 2)
    i = np.arange(-m, m + 1)
    golden = (1 + np.sqrt(5)) / 2
    # Generate longitude array.
    x = np.mod(i / golden, 1)
    mask = x >= .5
    x[mask] = x[mask] - 1
    x = 360 * x
    # Generate latitude array.
    y = 180 * np.arcsin(2 * i / n) / np.pi
    xy = np.vstack([x, y]).T
    return xy


class EBLogisticRegression:
    """
    Empirical Bayes logistic regression.

    Parameters
    ----------
    tol : float
        Convergence tolerance shrinkage parameter in evidence maximisation.
    max_iter : int
        Maximum number of iterations in evidence maximisation.
    solver_tol : float
        Convergence tolerance for MAP solver.
    solver_max_iter : int
        Maximum number of MAP solver iterations.
    alpha0 : float
        Initial shrinkage parameter value.
    intercept : bool
        Whether to add an intercept to the data.
    """
    def __init__(self, tol=1e-3, max_iter=100, solver_tol=1e-5,
        solver_max_iter=1000, alpha0=1., intercept=True):
        self.tol = tol
        self.max_iter = max_iter
        self.solver_tol = solver_tol
        self.solver_max_iter = solver_max_iter
        self.alpha0 = alpha0
        self.intercept=intercept
        self._map = LogisticRegression(
            tol=self.solver_tol,
            solver='lbfgs',
            max_iter=self.solver_max_iter
        )

    def _hess(self, beta):
        hess = np.eye(self.ndims + 1) * self.alpha
        hess[0, 0] = np.finfo(np.float16).eps
        p = expit(self._X.dot(beta))[:, np.newaxis]
        hess += self._X.T.dot(p * (1 - p) * self._X)
        return hess

    def _beta(self):
        self._map.set_params(C=1 / self.alpha)
        self._map.fit(self.X, self.y)
        beta = np.hstack((self._map.intercept_[0], self._map.coef_[0]))
        return beta

    def fit(self, X, y):
        """
        Fit model.

        Parameters
        ----------
        X : array, size=(n_samples, n_features)
            Feature samples.
        y : array, size=(n_samples,)
            Target samples.

        Returns
        -------
        self
        """
        self.alpha = self.alpha0
        self.X = X.copy()
        self.y = y.copy()
        self.nsamps, self.ndims = self.X.shape
        if self.intercept:
            self._X = np.hstack((np.ones((self.nsamps, 1)), self.X))
        else:
            self._X = self.X.copy()
        for i in range(self.max_iter):
            beta = self._beta()
            hess = self._hess(beta)
            tr_hess_inv = np.sum(1 / np.linalg.eigvalsh(hess))
            alpha = (self.ndims + 1) / (np.dot(beta, beta) + tr_hess_inv)
            if abs(alpha - self.alpha) < self.tol:
                break
            elif i == self.max_iter - 1:
                raise RuntimeWarning(
                    'Fit failed to converge after {} iterations.'.format(
                        self.max_iter
                    )
                )
            self.alpha = alpha
        self.alpha = alpha
        self.beta = self._beta()
        self.hess = self._hess(self.beta)
        return self

    def prob_rvs(self, X, n):
        """
        Calculate positive class probability for multiple draws from the
        posterior over the coefficients.

        Parameters
        ----------
        X : array, size=(n_samples, n_features)
            Feature samples.
        n : int
            Number of draws.

        Returns
        -------
        prob : array, size=(n_samples, n)
            Samples of positive class probabilities.
        """
        if self.intercept:
            _X = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            _X = X.copy()
        beta = np.random.multivariate_normal(
            self.beta, np.linalg.inv(self.hess), n
        ).T
        prob = expit(np.dot(_X, beta))
        return prob

    def prob(self, X, bound=None):
        """
        Calculate positive class probabilites.

        Parameters
        ----------
        X : array, size=(n_samples, n_features)
            Feature samples.
        bound : float
            Upper probability bound at which to evaluate the posterior. If
            `None` then expected value of the probability is returned.

        Returns
        -------
        probs : array, size=(n_samples,)
            Positive class probabilities.
        """
        if self.intercept:
            _X = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            _X = X.copy()
        m = _X.dot(self.beta).ravel()
        s = np.einsum('ij,jk,ik->i', _X, np.linalg.inv(self.hess), _X)
        if bound is None:
            k = 1 / np.sqrt(1 + np.pi * s / 8)
            p = expit(k * m)
        else:
            p = expit(norm.ppf(bound, m, np.sqrt(s)))
        return p
