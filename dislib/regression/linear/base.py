import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.task import task
from sklearn.base import BaseEstimator
from sklearn.utils import validation

from dislib.data.array import Array


class LinearRegression(BaseEstimator):
    """
    Multivariate linear regression using ordinary least squares.

    The model is: y = alpha + beta*X + err, where alpha is the intercept and
    beta is a vector of coefficients of shape (n_features,).

    The goal is to choose alpha and beta that minimize the sum of the squared
    errors. These optimal parameters are computed using linear algebra.

    Parameters
    ----------
    fit_intercept : bool, optional (default=True)
        Whether to calculate the intercept parameter for this model.
        If set to False, no intercept will be used in calculations
        (self.intercept_ will be 0).
    arity : int, optional (default=50)
        Arity of the reductions.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients (beta) in the linear model.
    intercept_ : float
        Estimated independent term (alpha) in the linear model.

    Examples
    --------
    >>> import numpy as np
    >>> import dislib as ds
    >>> from dislib.regression import LinearRegression
    >>> from pycompss.api.api import compss_wait_on
    >>> x_data = np.array([[1, 2], [2, 0], [3, 1], [4, 4], [5, 3]])
    >>> y_data = np.array([2, 1, 1, 2, 4.5]).reshape(-1, 1)
    >>> bn, bm = 2, 2
    >>> x = ds.array(x=x_data, block_size=(bn, bm))
    >>> y = ds.array(x=y_data, block_size=(bn, bm))
    >>> reg = LinearRegression()
    >>> reg.fit(x, y)
    >>> reg.coef_
    array([0.421875, 0.296875])
    >>> reg.intercept_
    0.240625
    >>> x_test = np.array([[3, 2], [4, 4]])
    >>> test_data = ds.array(x=x_test, block_size=(bn, bm))
    >>> pred = reg.predict(test_data).collect()
    >>> pred
    array([2.1, 3.115625])
    """

    def __init__(self, fit_intercept=True, arity=50):
        self.fit_intercept = fit_intercept
        self.arity = arity

    def fit(self, x, y):
        """
        Fit the linear model.

        Parameters
        ----------
        x : ds-array
            Explanatory variables
        y : ds-array
            Response variable

        Raises
        ------
        NotImplementedError
            If x or y are sparse arrays.

        """
        if x._sparse or y._sparse:
            raise NotImplementedError('Sparse data is not supported.')
        ztz = _compute_ztz(x, self.fit_intercept, self.arity)
        zty = _compute_zty(x, y, self.fit_intercept, self.arity)
        params = compss_wait_on(_compute_model_parameters(ztz, zty))
        if self.fit_intercept:
            self._intercept = params[0]
            self._coef = params[1:]
        else:
            self._intercept = 0
            self._coef = params

    def predict(self, x):
        """
        Predict using the linear model.

        Parameters
        ----------
        x : ds-array
            Samples to be predicted: x.shape (n_samples, 1).

        Returns
        -------
        y : ds-array
            Predicted values

        Raises
        ------
        NotImplementedError
            If x is a sparse array.

        """
        if x._sparse:
            raise NotImplementedError('Sparse data is not supported.')

        blocks = [list()]

        for r_block in x._iterator(axis='rows'):
            blocks[0].append(
                _predict(r_block._blocks, self._coef, self._intercept))

        return Array(blocks=blocks, top_left_shape=(x._top_left_shape[0], 1),
                     reg_shape=(x._reg_shape[0], 1), shape=(x.shape[0], 1),
                     sparse=x._sparse)

    @property
    def coef_(self):
        validation.check_is_fitted(self, '_coef')
        self._coef = compss_wait_on(self._coef)
        return self._coef

    @property
    def intercept_(self):
        validation.check_is_fitted(self, '_intercept')
        self._intercept = compss_wait_on(self._intercept)
        return self._intercept


def _compute_ztz(x, fit_intercept, arity):
    """Compute z.T@z, where z is x extended with an additional ones column
    if fit_intercept is set"""
    partials = []
    for row_block in x._iterator('rows'):
        partials.append(_partial_ztz(row_block._blocks, fit_intercept))
    return _reduce(_sum_arrays, partials, arity)


@task(x={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _partial_ztz(x, fit_intercept):
    z = Array._merge_blocks(x)
    if fit_intercept:
        z = np.hstack((np.ones((z.shape[0],)).reshape(-1, 1), z))
    return z.T@z


@task(returns=1)
def _sum_arrays(*arrays):
    return np.add.reduce(arrays)


def _reduce(func, partials, arity):
    while len(partials) > 1:
        partials_chunk = partials[:arity]
        partials = partials[arity:]
        partials.append(func(*partials_chunk))
    return partials[0]


def _compute_zty(x, y, fit_intercept, arity):
    """Compute z.T@y, where z is x extended with an additional ones column
    if fit_intercept is set"""
    x_it, y_it = x._iterator('rows'), y._iterator('rows')
    partials = []
    for bx, by in zip(x_it, y_it):
        partials.append(_partial_zty(bx._blocks, by._blocks, fit_intercept))
    return _reduce(_sum_arrays, partials, arity)


@task(x={Type: COLLECTION_IN, Depth: 2}, y={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def _partial_zty(x, y, fit_intercept):
    z = Array._merge_blocks(x)
    if fit_intercept:
        z = np.hstack((np.ones((z.shape[0],)).reshape(-1, 1), z))
    y = Array._merge_blocks(y)
    return (z.T@y).flatten()


@task(returns=1)
def _compute_model_parameters(ztz, zty):
    """Compute the model parameters, inv(z.T@z)@z.T@y, by solving a linear
    system"""
    return np.linalg.solve(ztz, zty)


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _predict(blocks, coef, intercept):
    x = Array._merge_blocks(blocks)
    return x@coef + intercept
