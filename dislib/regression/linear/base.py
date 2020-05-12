import numpy as np
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
    coef_ : ds-array, shape (n_features, n_targets)
        Estimated coefficients (beta) in the linear model.
    intercept_ : ds-array, shape (1, n_targets)
        Estimated independent term (alpha) in the linear model.

    Examples
    --------
    >>> import numpy as np
    >>> import dislib as ds
    >>> from dislib.regression import LinearRegression
    >>> from pycompss.api.api import compss_wait_on
    >>> x_data = np.array([[1, 2], [2, 0], [3, 1], [4, 4], [5, 3]])
    >>> y_data = np.array([2, 1, 1, 2, 4.5])
    >>> bn, bm = 2, 2
    >>> x = ds.array(x=x_data, block_size=(bn, bm))
    >>> y = ds.array(x=y_data, block_size=(bn, 1))
    >>> reg = LinearRegression()
    >>> reg.fit(x, y)
    >>> reg.coef_.collect()
    array([0.421875, 0.296875])
    >>> reg.intercept_.collect()
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
        x : ds-array, shape (n_samples, n_features)
            Explanatory variables.
        y : ds-array, shape (n_samples, n_targets)
            Response variables.

        Raises
        ------
        NotImplementedError
            If x or y are sparse arrays.

        """
        if x._sparse or y._sparse:
            raise NotImplementedError('Sparse data is not supported.')
        self._n_features = x.shape[1]  # Number of explanatory variables
        self._n_targets = y.shape[1]  # Number of response variables
        ztz = _compute_ztz(x, self.fit_intercept, self.arity)
        zty = _compute_zty(x, y, self.fit_intercept, self.arity)
        params = _compute_model_parameters(ztz, zty, self.fit_intercept)
        self._intercept = _to_dsarray(params[0], (1, self._n_targets))
        self._coef = _to_dsarray(params[1],
                                 (self._n_features, self._n_targets))

    def predict(self, x):
        """
        Predict using the linear model.

        Parameters
        ----------
        x : ds-array, shape (n_samples_predict, n_features)
            Samples to be predicted.

        Returns
        -------
        y : ds-array, shape (n_samples_predict, n_targets)
            Predicted values.

        Raises
        ------
        NotImplementedError
            If x is a sparse array.

        """
        if x._sparse:
            raise NotImplementedError('Sparse data is not supported.')

        blocks = []

        for r_block in x._iterator(axis='rows'):
            blocks.append([_predict(r_block._blocks,
                                    self._coef._blocks[0][0],
                                    self._intercept._blocks[0][0])])
        return Array(blocks=blocks,
                     top_left_shape=(x._top_left_shape[0], self._n_targets),
                     reg_shape=(x._reg_shape[0], self._n_targets),
                     shape=(x.shape[0], self._n_targets), sparse=x._sparse)

    @property
    def coef_(self):
        validation.check_is_fitted(self, '_coef')
        return self._coef

    @property
    def intercept_(self):
        validation.check_is_fitted(self, '_intercept')
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
    return z.T@y


@task(returns=2)
def _compute_model_parameters(ztz, zty, fit_intercept):
    """Compute the model parameters, inv(z.T@z)@z.T@y, by solving a linear
    system"""
    params = np.linalg.solve(ztz, zty)
    if fit_intercept:
        return params[0], params[1:]
    else:
        return np.zeros((1, zty.shape[1])), params


def _to_dsarray(np_array, shape):
    """Takes an unsynchronized numpy 2-d array and its shape, and creates the
    corresponding dsarray with a single block."""
    return Array(blocks=[[np_array]], top_left_shape=shape, reg_shape=shape,
                 shape=shape, sparse=False)


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _predict(blocks, coef, intercept):
    x = Array._merge_blocks(blocks)
    return x@coef + intercept
