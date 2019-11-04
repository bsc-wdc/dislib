import numpy as np
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.task import task
from sklearn.base import BaseEstimator

from dislib.data.array import Array


class LinearRegression(BaseEstimator):
    """
    Simple linear regression using ordinary least squares.

    model: y1 = alpha + beta*x_i + epsilon_i

    goal: y = alpha + beta*x

    Parameters
    ----------
    arity : int, optional (default=50)
        Arity of the reductions.

    Attributes
    ----------
    coef_ : array, shape (n_features, )
        Estimated coefficient (beta) in the linear model.
    intercept_ : float
        Estimated independent term (alpha) in the linear model.

    Examples
    --------
    >>> import numpy as np
    >>> import dislib as ds
    >>> from dislib.regression import LinearRegression
    >>> from pycompss.api.api import compss_wait_on
    >>> x_data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    >>> y_data = np.array([2, 1, 1, 2, 4.5]).reshape(-1, 1)
    >>> bn, bm = 2, 2
    >>> x = ds.array(x=x_data, block_size=(bn, bm))
    >>> y = ds.array(x=y_data, block_size=(bn, bm))
    >>> reg = LinearRegression()
    >>> reg.fit(x, y)
    >>> # y = 0.6 * x + 0.3
    >>> reg.coef_
    0.6
    >>> reg.intercept_
    0.3
    >>> x_test = np.array([3, 5]).reshape(-1, 1)
    >>> test_data = ds.array(x=x_test, block_size=(bn, bm))
    >>> pred = reg.predict(test_data).collect()
    >>> pred
    array([2.1, 3.3])
    """

    def __init__(self, arity=50):
        self.arity = arity

    def fit(self, x, y):
        """
        Fit the linear model.

        Parameters
        ----------
        x : ds-array
            Samples to be used to fit the model
        y : ds-array
            Labels of the samples

        Raises
        ------
        NotImplementedError
            If x is a sparse array.

        """
        if x._sparse or y._sparse:
            raise NotImplementedError('Sparse data is not supported.')
        mean_x, mean_y = _variable_means(x, y, self.arity)
        beta, alpha = _compute_regression(x, y, mean_x, mean_y, self.arity)
        self.coef_ = beta
        self.intercept_ = alpha

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
                _predict(r_block._blocks, self.coef_, self.intercept_))

        return Array(blocks=blocks, top_left_shape=(x._top_left_shape[0], 1),
                     reg_shape=(x._reg_shape[0], 1), shape=(x.shape[0], 1),
                     sparse=x._sparse)


def _reduce(func, partials, arity):
    while len(partials) > 1:
        partials_chunk = partials[:arity]
        partials = partials[arity:]
        partials.append(func(*partials_chunk))
    return partials[0]


def _variable_means(x, y, arity):
    partials = []
    x_it, y_it = x._iterator('rows'), y._iterator('rows')
    for i in range(x._n_blocks[0]):
        bx, by = next(x_it), next(y_it)
        partials.append(_sum_and_count_samples(bx._blocks, by._blocks))
    total_sums_and_count = _reduce(_sum_params, partials, arity)
    mean_x, mean_y = _divide_sums_by_count(total_sums_and_count)
    return mean_x, mean_y


@task(x={Type: COLLECTION_IN, Depth: 2}, y={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def _sum_and_count_samples(x, y):
    x, y = Array._merge_blocks(x), Array._merge_blocks(y)
    partial_sum_x = np.sum(x[:, 0])  # the 0 is because only 1D LR is supported
    partial_sum_y = np.sum(y)
    return partial_sum_x, partial_sum_y, len(x)


@task(returns=1)
def _sum_params(*partials):
    return tuple(sum(p) for p in zip(*partials))


@task(returns=2)
def _divide_sums_by_count(sums_and_count):
    sum_x, sum_y, count = sums_and_count
    return sum_x / count, sum_y / count


def _compute_regression(x, y, mean_x, mean_y, arity):
    partials = []
    x_it, y_it = x._iterator('rows'), y._iterator('rows')

    for i in range(x._n_blocks[0]):
        bx, by = next(x_it), next(y_it)
        partials.append(
            _partial_variability_params(bx._blocks, by._blocks, mean_x,
                                        mean_y))
    variability_params = _reduce(_sum_params, partials, arity)
    return _calculate_coefficients(mean_x, mean_y, variability_params)


@task(x={Type: COLLECTION_IN, Depth: 2}, y={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def _partial_variability_params(x, y, mean_x, mean_y):
    x, y = Array._merge_blocks(x), Array._merge_blocks(y)

    normalized_x = x[:, 0] - mean_x  # the 0 is because only 1D LR is supported
    normalized_y = y - mean_y
    normalized_xy_dot = np.dot(normalized_x, normalized_y)
    normalized_xx_dot = np.dot(normalized_x, normalized_x)
    return normalized_xy_dot, normalized_xx_dot


@task(returns=2)
def _calculate_coefficients(mean_x, mean_y, variability_params):
    dot_xy, dot_xx = variability_params
    beta = dot_xy / dot_xx
    alpha = mean_y - beta * mean_x
    return beta, alpha


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _predict(blocks, coef, intercept):
    x = Array._merge_blocks(blocks)
    y = coef * x[:, 0] + intercept  # the 0 is because only 1D LR is supported

    return y
