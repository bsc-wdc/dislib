"""
ADMM Lasso

@Authors: Aleksandar Armacki and Lidija Fodor
@Affiliation: Faculty of Sciences, University of Novi Sad, Serbia

This work is supported by the I-BiDaaS project, funded by the European
Commission under Grant Agreement No. 780787.
"""
import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator

import dislib as ds
from dislib.optimization import ADMM


class Lasso(BaseEstimator):
    """Lasso represents the Least Absolute Shrinkage and Selection Operator
    (Lasso) for
    regression analysis, solved in a distributed manner. 

    :param n: The number of agents used to solve the problem
    :param max_iter: The maximum number of iterations before the algorithm
    stops automatically
    :param lmbd: The regularization parameter for Lasso regression

    """

    def __init__(self, max_iter=500, lmbd=1e-3, rho=1, abstol=1e-4,
                 reltol=1e-2, warm_start=False, verbose=False):
        self.max_iter = max_iter
        self.lmbd = lmbd
        self.rho = rho
        self.abstol = abstol
        self.reltol = reltol
        self.warm_start = warm_start
        self.n_iter_ = 0
        self._verbose = verbose

    @staticmethod
    def _loss_fn(x, y, w):
        return 1 / 2 * cp.norm(cp.matmul(x, w) - y, p=2) ** 2

    def fit(self, x, y):
        if x._reg_shape != x._top_left_shape:
            raise ValueError("x must be a regular ds-array")

        n_samples, n_features = x.shape

        # initialization
        z = np.zeros(n_features)

        # u has one row per each row-block in x
        u = ds.zeros((x._n_blocks[0], n_features), (1, x._reg_shape[1]))
        soft_thres = self.lmbd / self.rho

        admm = ADMM(z, u, self.rho, soft_thres, self.abstol,
                    self.reltol, self.warm_start, Lasso._loss_fn)

        while not admm.converged_ and self.n_iter_ < self.max_iter:
            admm.step(x, y)
            self.n_iter_ += 1

            if self._verbose:
                print("Iteration ", self.n_iter_)

        self.coef_ = admm.z
        return self

    def predict(self, x):
        return np.dot(x.collect(), self.coef_)

    def fit_predict(self, x):
        self.fit()
        return self.predict(x)
