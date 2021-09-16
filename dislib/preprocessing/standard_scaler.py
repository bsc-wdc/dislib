import numpy as np
from pycompss.api.parameter import Depth, Type, COLLECTION_IN, COLLECTION_OUT
from pycompss.api.task import task
from scipy.sparse import csr_matrix, issparse

from dislib.data.array import Array
import dislib as ds


class StandardScaler(object):
    """ Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in the training set. Mean and standard
    deviation are then stored to be used on later data using the transform
    method.

    Attributes
    ----------
    mean_ : ds-array, shape (1, n_features)
        The mean value for each feature in the training set.
    var_ : ds-array, shape (1, n_features)
        The variance for each feature in the training set.
    """

    def __init__(self):
        self.mean_ = None
        self.var_ = None

    def fit(self, x):
        """ Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)

        Returns
        -------
        self : StandardScaler
        """
        self.mean_ = ds.apply_along_axis(np.mean, 0, x)
        var_blocks = [[]]

        for row, m_row in zip(x._iterator(1), self.mean_._iterator(1)):
            var_blocks[0].append(_compute_var(row._blocks, m_row._blocks))

        self.var_ = Array(var_blocks,
                          top_left_shape=self.mean_._top_left_shape,
                          reg_shape=self.mean_._reg_shape,
                          shape=self.mean_.shape, sparse=x._sparse)

        return self

    def fit_transform(self, x):
        """ Fit to data, then transform it.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)

        Returns
        -------
        x_new : ds-array, shape=(n_samples, n_features)
            Scaled data.
        """
        return self.fit(x).transform(x)

    def transform(self, x):
        """
        Standarize data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)

        Returns
        -------
        x_new : ds-array, shape=(n_samples, n_features)
            Scaled data.
        """
        if self.mean_ is None or self.var_ is None:
            raise Exception("Model has not been initialized.")

        n_blocks = x._n_blocks[1]
        blocks = []
        m_blocks = self.mean_._blocks
        v_blocks = self.var_._blocks

        for row in x._iterator(axis=0):
            out_blocks = [object() for _ in range(n_blocks)]
            _transform(row._blocks, m_blocks, v_blocks, out_blocks)
            blocks.append(out_blocks)

        return Array(blocks, top_left_shape=x._top_left_shape,
                     reg_shape=x._reg_shape, shape=x.shape,
                     sparse=x._sparse)


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      m_blocks={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def _compute_var(blocks, m_blocks):
    x = Array._merge_blocks(blocks)
    mean = Array._merge_blocks(m_blocks)
    sparse = issparse(x)

    if sparse:
        x = x.toarray()
        mean = mean.toarray()

    var = np.mean(np.array(x - mean) ** 2, axis=0)

    if sparse:
        return csr_matrix(var)
    else:
        return var


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      m_blocks={Type: COLLECTION_IN, Depth: 2},
      v_blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks=COLLECTION_OUT)
def _transform(blocks, m_blocks, v_blocks, out_blocks):
    x = Array._merge_blocks(blocks)
    mean = Array._merge_blocks(m_blocks)
    var = Array._merge_blocks(v_blocks)
    sparse = issparse(x)

    if sparse:
        x = x.toarray()
        mean = mean.toarray()
        var = var.toarray()

    scaled_x = (x - mean) / np.sqrt(var)

    constructor_func = np.array if not sparse else csr_matrix
    start, end = 0, 0

    for i, block in enumerate(blocks[0]):
        end += block.shape[1]
        out_blocks[i] = constructor_func(scaled_x[:, start:end])
