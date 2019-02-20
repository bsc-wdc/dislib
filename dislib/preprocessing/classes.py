import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import INOUT
from pycompss.api.task import task


class StandardScaler(object):
    """ Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in the training set. Mean and standard
    deviation are then stored to be used on later data using the transform
    method.

    Parameters
    ----------
    arity : int, optional (default=50)
        Arity of the reduction phase carried out to compute the mean and
        variance of the input dataset.

    Attributes
    ----------
    mean_ : ndarray, shape (n_features,)
        The mean value for each feature in the training set.
    var_ : ndarray, shape (n_features,)
        The variance for each feature in the training set.
    """

    def __init__(self, arity=50):
        self._mean = None
        self._var = None
        self._arity = arity

    @property
    def mean_(self):
        if self._mean is not None:
            self._mean = compss_wait_on(self._mean)
            return self._mean

        return None

    @property
    def var_(self):
        if self._var is not None:
            self._var = compss_wait_on(self._var)
            return self._var

        return None

    def fit(self, dataset):
        """ Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        dataset : Dataset
        """
        partial_sums = []

        for subset in dataset:
            partial_sums.append(_partial_sum(subset))

        total_sum = self._merge_partial_sums(partial_sums)

        partial_vars = []

        for subset in dataset:
            partial_vars.append(_partial_var(subset, total_sum))

        total_var = self._merge_partial_vars(partial_vars)

        self._mean, self._var = _compute_stats(total_sum, total_var)

    def fit_transform(self, dataset):
        """ Fit to data, then transform it.

        Parameters
        ----------
        dataset : Dataset
        """
        self.fit(dataset)
        self.transform(dataset)

    def transform(self, dataset):
        """
        Standarize data.

        Parameters
        ----------
        dataset : Dataset
        """
        if self._mean is None or self._var is None:
            raise Exception("Model has not been initialized.")

        for subset in dataset:
            _transform(subset, self._mean, self._var)

    def _merge_partial_sums(self, partials):
        while len(partials) > 1:
            partials_subset = partials[:self._arity]
            partials = partials[self._arity:]
            partials.append(_merge_sums(*partials_subset))

        return partials[0]

    def _merge_partial_vars(self, partials):
        while len(partials) > 1:
            partials_subset = partials[:self._arity]
            partials = partials[self._arity:]
            partials.append(_merge_vars(*partials_subset))

        return partials[0]


@task(returns=tuple)
def _partial_sum(subset):
    return np.sum(subset.samples, axis=0), subset.samples.shape[0]


@task(returns=tuple)
def _merge_sums(*partials):
    sum_ = sum(par[0] for par in partials)
    size_ = sum(par[1] for par in partials)
    return sum_, size_


@task(returns=np.array)
def _partial_var(subset, sum):
    mean = sum[0] / sum[1]
    return np.sum((subset.samples - mean) ** 2, axis=0)


@task(returns=np.array)
def _merge_vars(*partials):
    return np.sum(partials, axis=0)


@task(returns=2)
def _compute_stats(sum, var):
    mean_ = sum[0] / sum[1]
    var_ = var / sum[1]
    return mean_, var_


@task(subset=INOUT)
def _transform(subset, mean, var):
    scaled_samples = (subset.samples - mean) / np.sqrt(var)
    subset.samples = scaled_samples
