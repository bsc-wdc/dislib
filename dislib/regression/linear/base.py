import numpy as np
from pycompss.api.parameter import INOUT

from pycompss.api.task import task


class LinearRegression:
    """
    Simple linear regression using ordinary least squares.

    model: y1 = alpha + beta*x_i + epsilon_i
    goal: y = alpha + beta*x

    Parameters
    ----------
    arity : int
        Arity of the reductions.

    Attributes
    ----------
    coef_ : array, shape (n_features, )
        Estimated coefficients (beta) for the linear model.
    intercept_ : float
        Estimated independent term (alpha) in the linear model.

    Examples
    --------
    >>> import numpy as np
    >>> train_x = np.array([1, 2, 3, 4, 5])[:, np.newaxis]
    >>> train_y = np.array([2, 1, 1, 2, 4.5])
    >>> from dislib.data import load_data
    >>> train_dataset = load_data(x=train_x, y=train_y, subset_size=2)
    >>> from dislib.regression import LinearRegression
    >>> reg = LinearRegression()
    >>> reg.fit(train_dataset)
    >>> # y = 0.6 * x + 0.3
    >>> reg.coef_
    0.6
    >>> reg.intercept_
    0.3
    >>> test_x = np.array([3, 5])[:, np.newaxis]
    >>> test_dataset = load_data(x=test_x, subset_size=2)
    >>> reg.predict(test_dataset)
    >>> test_dataset.labels
    array([2.1, 3.3])
    """

    def __init__(self, arity=50):
        self._arity = arity

    def fit(self, dataset):
        """
        Fit the linear model.

        Parameters
        ----------
        dataset : Dataset
            Training dataset: x.shape (n_samples, 1), y.shape (n_samples, ).
        """
        mean_x, mean_y = _variable_means(dataset, self._arity)
        beta, alpha = _compute_regression(dataset, mean_x, mean_y, self._arity)
        self.coef_ = beta
        self.intercept_ = alpha

    def predict(self, dataset):
        """
        Predict using the linear model.

        Parameters
        ----------
        dataset : Dataset
            Dataset with samples: x.shape (n_samples, 1). Predicted values are
            populated in the labels attribute.
        """
        for subset in dataset:
            _predict(subset, self.coef_, self.intercept_)


def _reduce(func, partials, arity):
    while len(partials) > 1:
        partials_chunk = partials[:arity]
        partials = partials[arity:]
        partials.append(func(*partials_chunk))
    return partials[0]


def _variable_means(dataset, arity):
    partials = [_sum_and_count_samples(subset) for subset in dataset]
    total_sums_and_count = _reduce(_sum_params, partials, arity)
    mean_x, mean_y = _divide_sums_by_count(total_sums_and_count)
    return mean_x, mean_y


@task(returns=1)
def _sum_and_count_samples(subset):
    partial_sum_x = np.sum(subset.samples[:, 0])
    partial_sum_y = np.sum(subset.labels)
    return partial_sum_x, partial_sum_y, subset.samples.shape[0]


@task(returns=1)
def _sum_params(*partials):
    return tuple(sum(p) for p in zip(*partials))


@task(returns=2)
def _divide_sums_by_count(sums_and_count):
    sum_x, sum_y, count = sums_and_count
    return sum_x/count, sum_y/count


def _compute_regression(dataset, mean_x, mean_y, arity):
    partials = []
    for subset in dataset:
        partials.append(_partial_variability_params(subset, mean_x, mean_y))
    variability_params = _reduce(_sum_params, partials, arity)
    return _calculate_coefficients(mean_x, mean_y, variability_params)


@task(returns=1)
def _partial_variability_params(subset, mean_x, mean_y):
    normalized_x = subset.samples[:, 0] - mean_x
    normalized_y = subset.labels - mean_y
    normalized_xy_dot = np.dot(normalized_x, normalized_y)
    normalized_xx_dot = np.dot(normalized_x, normalized_x)
    return normalized_xy_dot, normalized_xx_dot


@task(returns=2)
def _calculate_coefficients(mean_x, mean_y, variability_params):
    dot_xy, dot_xx = variability_params
    beta = dot_xy / dot_xx
    alpha = mean_y - beta*mean_x
    return beta, alpha


@task(subset=INOUT)
def _predict(subset, coef, intercept):
    subset.labels = coef*subset.samples[:, 0] + intercept
    return subset
