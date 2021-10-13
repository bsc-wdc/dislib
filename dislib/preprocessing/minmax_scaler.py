import numpy as np
from pycompss.api.constraint import constraint
from pycompss.api.parameter import Depth, Type, COLLECTION_IN, COLLECTION_OUT
from pycompss.api.task import task
from scipy.sparse import csr_matrix, issparse

from dislib.data.array import Array
import dislib as ds


class MinMaxScaler(object):
    """ Standardize features by rescaling them to the provided range

    Scaling happen independently on each feature by computing the relevant
    statistics on the samples in the training set. Minimum and Maximum
    values are then stored to be used on later data using the transform method.

    Attributes
    ----------
    feature_range : tuple
        The desired range of values in the ds-array.
    """

    def __init__(self, feature_range=(0, 1)):
        self._feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, x):
        """ Compute the min and max values for later scaling.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)

        Returns
        -------
        self : MinMaxScaler
        """

        self.data_min_ = ds.apply_along_axis(np.min, 0, x)
        self.data_max_ = ds.apply_along_axis(np.max, 0, x)

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
        Scale data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)

        Returns
        -------
        x_new : ds-array, shape=(n_samples, n_features)
            Scaled data.
        """
        if self.data_min_ is None or self.data_max_ is None:
            raise Exception("Model has not been initialized.")

        n_blocks = x._n_blocks[1]
        blocks = []
        min_blocks = self.data_min_._blocks
        max_blocks = self.data_max_._blocks

        for row in x._iterator(axis=0):
            out_blocks = [object() for _ in range(n_blocks)]
            _transform(row._blocks, min_blocks, max_blocks, out_blocks,
                       self._feature_range[0], self._feature_range[1])
            blocks.append(out_blocks)

        return Array(blocks, top_left_shape=x._top_left_shape,
                     reg_shape=x._reg_shape, shape=x.shape,
                     sparse=x._sparse)


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2},
      min_blocks={Type: COLLECTION_IN, Depth: 2},
      max_blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks=COLLECTION_OUT)
def _transform(blocks, min_blocks, max_blocks, out_blocks,
               range_min, range_max):
    x = Array._merge_blocks(blocks)
    min_val = Array._merge_blocks(min_blocks)
    max_val = Array._merge_blocks(max_blocks)
    sparse = issparse(x)

    if sparse:
        x = x.toarray()
        min_val = min_val.toarray()
        max_val = max_val.toarray()

    std_x = (x - min_val) / (max_val - min_val)
    scaled_x = std_x * (range_max - range_min) + range_min

    constructor_func = np.array if not sparse else csr_matrix
    start, end = 0, 0

    for i, block in enumerate(blocks[0]):
        end += block.shape[1]
        out_blocks[i] = constructor_func(scaled_x[:, start:end])
