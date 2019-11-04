from copy import copy

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import COLLECTION_IN, Depth, Type, COLLECTION_INOUT
from pycompss.api.task import task
from sklearn.base import BaseEstimator
from sklearn.utils import validation

from dislib.data.array import Array


class PCA(BaseEstimator):
    """ Principal component analysis (PCA) using the covariance method.

    Performs a full eigendecomposition of the covariance matrix.

    Parameters
    ----------
    n_components : int or None, optional (default=None)
        Number of components to keep. If None, all components are kept.
    arity : int, optional (default=50)
        Arity of the reductions.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum
        variance in the data. The components are sorted by explained_variance_.

        Equal to the n_components eigenvectors of the covariance matrix with
        greater eigenvalues.
    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.

        Equal to the first n_components largest eigenvalues of the covariance
        matrix.
    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

    Examples
    --------
    >>> from dislib.decomposition import PCA
    >>> import numpy as np
    >>> import dislib as ds
    >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> bn, bm = 2, 2
    >>> data = ds.array(x=x, block_size=(bn, bm))
    >>> pca = PCA()
    >>> transformed_data = pca.fit_transform(data)
    >>> print(transformed_data)
    >>> print(pca.components_)
    >>> print(pca.explained_variance_)
    """

    def __init__(self, n_components=None, arity=50):
        self.n_components = n_components
        self.arity = arity

    @property
    def components_(self):
        validation.check_is_fitted(self, '_components')
        self._components = compss_wait_on(self._components)
        return self._components

    @property
    def explained_variance_(self):
        validation.check_is_fitted(self, '_variance')
        self._variance = compss_wait_on(self._variance)
        return self._variance

    def fit(self, x):
        """ Fit the model with the dataset.

        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : PCA
        """

        n_samples = x.shape[0]
        self.mean_ = _features_mean(x, self.arity, n_samples)
        norm_blocks = []
        for rows in x._iterator('rows'):
            aux_rows = [object() for _ in range(x._n_blocks[1])]
            _normalize(rows._blocks, aux_rows, self.mean_)
            norm_blocks.append(aux_rows)

        # we shallow copy the original to create a normalized darray
        norm_x = copy(x)
        # shallow copy is enough to avoid modifying original darray x when
        # changing the blocks
        norm_x._blocks = norm_blocks

        scatter_matrix = _scatter_matrix(norm_x, self.arity)
        covariance_matrix = _estimate_covariance(scatter_matrix, n_samples)
        eig_val, eig_vec = _decompose(covariance_matrix, self.n_components)

        self._components = eig_vec
        self._variance = eig_val

        return self

    def fit_transform(self, x):
        """ Fit the model with the dataset and apply the dimensionality
        reduction to it.

        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        transformed_darray : ds-array, shape (n_samples, n_components)
        """
        return self.fit(x).transform(x)

    def transform(self, x):
        """
        Apply dimensionality reduction to ds-array.

        The given dataset is projected on the first principal components
        previously extracted from a training ds-array.

        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            New ds-array, with the same n_features as the training dataset.

        Returns
        -------
        transformed_darray : ds-array, shape (n_samples, n_components)
        """
        return _transform(x, self.mean_, self.components_)


def _features_mean(x, arity, n_samples):
    partials = []
    for rows in x._iterator('rows'):
        partials.append(_subset_feature_sum(rows._blocks))
    return _reduce_features_mean(partials, arity, n_samples)


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _subset_feature_sum(blocks):
    block = Array._merge_blocks(blocks)
    return block.sum(axis=0)


def _reduce_features_mean(partials, arity, n_samples):
    while len(partials) > 1:
        partials_chunk = partials[:arity]
        partials = partials[arity:]
        partials.append(_merge_features_sum(*partials_chunk))
    return _finalize_features_mean(partials[0], n_samples)


@task(returns=1)
def _merge_features_sum(*partials):
    return sum(partials)


@task(returns=1)
def _finalize_features_mean(feature_sums, n_samples):
    return feature_sums / n_samples


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_INOUT, Depth: 1})
def _normalize(blocks, out_blocks, means):
    data = Array._merge_blocks(blocks)
    data = np.array(data - means)

    bn, bm = blocks[0][0].shape

    for j in range(len(blocks[0])):
        out_blocks[j] = data[:, j * bm:(j + 1) * bm]


def _scatter_matrix(x, arity):
    partials = []
    for rows in x._iterator('rows'):
        partials.append(_subset_scatter_matrix(rows._blocks))
    return _reduce_scatter_matrix(partials, arity)


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _subset_scatter_matrix(blocks):
    data = Array._merge_blocks(blocks)
    return np.dot(data.T, data)


def _reduce_scatter_matrix(partials, arity):
    while len(partials) > 1:
        partials_chunk = partials[:arity]
        partials = partials[arity:]
        partials.append(_merge_partial_scatter_matrix(*partials_chunk))
    return partials[0]


@task(returns=1)
def _merge_partial_scatter_matrix(*partials):
    return sum(partials)


@task(returns=1)
def _estimate_covariance(scatter_matrix, n_samples):
    return scatter_matrix / (n_samples - 1)


@task(returns=2)
def _decompose(covariance_matrix, n_components):
    eig_val, eig_vec = np.linalg.eigh(covariance_matrix)

    if n_components is None:
        n_components = len(eig_val)

    # first n_components eigenvalues in descending order:
    eig_val = eig_val[::-1][:n_components]

    # first n_components eigenvectors in rows, with the corresponding order:
    eig_vec = eig_vec.T[::-1][:n_components]

    # normalize eigenvectors sign to ensure deterministic output
    max_abs_cols = np.argmax(np.abs(eig_vec), axis=1)
    signs = np.sign(eig_vec[range(len(eig_vec)), max_abs_cols])
    eig_vec *= signs[:, np.newaxis]

    return eig_val, eig_vec


def _transform(x, mean, components):
    new_blocks = []
    for rows in x._iterator('rows'):
        out_blocks = [object() for _ in range(rows._n_blocks[1])]
        _subset_transform(rows._blocks, out_blocks, mean, components)
        new_blocks.append(out_blocks)

    return Array(blocks=new_blocks, top_left_shape=x._top_left_shape,
                 reg_shape=x._reg_shape,
                 shape=(x.shape[0], components.shape[1]), sparse=x._sparse)


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_INOUT, Depth: 1})
def _subset_transform(blocks, out_blocks, mean, components):
    data = Array._merge_blocks(blocks)
    bn, bm = blocks[0][0].shape

    res = (np.matmul(data - mean, components.T))

    for j in range(0, len(blocks[0])):
        out_blocks[j] = res[:, j * bm:(j + 1) * bm]
