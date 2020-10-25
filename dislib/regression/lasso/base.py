"""
ADMM Lasso

@Authors: Aleksandar Armacki and Lidija Fodor
@Affiliation: Faculty of Sciences, University of Novi Sad, Serbia

This work is supported by the I-BiDaaS project, funded by the European
Commission under Grant Agreement No. 780787.
"""
try:
    import cvxpy as cp
except ImportError:
    import warnings
    warnings.warn('Cannot import cvxpy module. Lasso estimator will not work.')
from sklearn.base import BaseEstimator
from dislib.optimization import ADMM


class Lasso(BaseEstimator):
    """ Lasso represents the Least Absolute Shrinkage and Selection Operator
    (Lasso) for regression analysis, solved in a distributed manner with ADMM.

    Parameters
    ----------
    lmbd : float, optional (default=1e-3)
        The regularization parameter for Lasso regression.
    rho : float, optional (default=1)
        The penalty parameter for constraint violation.
    max_iter : int, optional (default=100)
        The maximum number of iterations of ADMM.
    atol : float, optional (default=1e-4)
        The absolute tolerance used to calculate the early stop criterion
        for ADMM.
    rtol : float, optional (default=1e-2)
        The relative tolerance used to calculate the early stop criterion
        for ADMM.
    verbose : boolean, optional (default=False)
        Whether to print information about the optimization process.

    Attributes
    ----------
    coef_ : ds-array, shape=(1, n_features)
        Parameter vector.
    n_iter_ : int
        Number of iterations run by ADMM.
    converged_ : boolean
        Whether ADMM converged.

    See also
    --------
    ADMM
    """

    def __init__(self, lmbd=1e-3, rho=1, max_iter=100, atol=1e-4, rtol=1e-2,
                 verbose=False):
        self.max_iter = max_iter
        self.lmbd = lmbd
        self.rho = rho
        self.atol = atol
        self.rtol = rtol
        self.verbose = verbose

    @staticmethod
    def _loss_fn(x, y, w):
        return 1 / 2 * cp.norm(cp.matmul(x, w) - y, p=2) ** 2

    def fit(self, x, y):
        """ Fits the model with training data. Optimization is carried out
        using ADMM.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Training samples.
        y : ds-array, shape=(n_samples, 1)
            Class labels of x.

        Returns
        -------
        self :  Lasso
        """
        k = self.lmbd / self.rho

        admm = ADMM(Lasso._loss_fn, k, self.rho, max_iter=self.max_iter,
                    rtol=self.rtol, atol=self.atol, verbose=self.verbose)
        admm.fit(x, y)

        self.n_iter_ = admm.n_iter_
        self.converged_ = admm.converged_
        self.coef_ = admm.z_

        return self

    def predict(self, x):
        """ Predict using the linear model.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Samples.

        Returns
        -------
        y : ds-array, shape=(n_samples, 1)
            Predicted values.
        """
        coef = self.coef_.T

        # this rechunk can be removed as soon as matmul supports multiplying
        # ds-arrays with different block shapes
        if coef._reg_shape[0] != x._reg_shape[1]:
            coef = coef.rechunk(x._reg_shape)

        return x @ coef

    def fit_predict(self, x):
        """ Fits the model and predicts using the same data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Training samples.

        Returns
        -------
        y : ds-array, shape=(n_samples, 1)
            Predicted values.
        """
        return self.fit(x).predict(x)
