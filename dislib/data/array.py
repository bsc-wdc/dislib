from math import ceil

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.parameter import COLLECTION_INOUT
from pycompss.api.task import task
from scipy import sparse as sp
from scipy.sparse import issparse


def array(x, block_size):
    # TODO: documentation ain't true
    """
    Loads data into a Distributed Array.

    Parameters
    ----------
    x : ndarray, shape=[n_samples, n_features]
        Array of samples.
    block_size : (int, int)
        Block sizes in number of samples.

    Returns
    -------
    darray : Array
        A distributed representation of the data divided in blocks.
    """
    x_size, y_size = block_size

    blocks = []
    for i in range(0, x.shape[0], x_size):
        row = [x[i: i + x_size, j: j + y_size] for j in
               range(0, x.shape[1], y_size)]
        blocks.append(row)

    sparse = issparse(x)
    darray = Array(blocks=blocks, block_size=block_size, shape=x.shape,
                   sparse=sparse)

    return darray


def load_svmlight_file(path, block_size, n_features, store_sparse):
    """ Loads a LibSVM file into a Dataset.

     Parameters
    ----------
    path : string
        File path.
    block_size : (int, int)
        Block size for the output darray.
    n_features : int
        Number of features.
    store_sparse : boolean
        Whether to use scipy.sparse data structures to store data. If False,
        numpy.array is used instead.

    Returns
    -------
    x, y : (darray, darray)
        A distributed representation (darray) of the X and y.
    """
    n, m = block_size
    lines = []
    x_blocks, y_blocks = [], []

    n_rows = 0
    with open(path, "r") as f:
        for line in f:
            n_rows += 1
            lines.append(line.encode())

            if len(lines) == n:
                # line 0 -> X, line 1 -> y
                out_blocks = Array._get_out_blocks(1, ceil(n_features / m))
                out_blocks.append([object()])
                # out_blocks.append([])
                _read_libsvm(lines, out_blocks, col_size=m,
                             n_features=n_features, store_sparse=store_sparse)
                # we append only the list forming the row (out_blocks depth=2)
                x_blocks.append(out_blocks[0])
                y_blocks.append(out_blocks[1])
                lines = []

    if lines:
        out_blocks = Array._get_out_blocks(1, ceil(n_features / m))
        out_blocks.append([object()])
        _read_libsvm(lines, out_blocks, col_size=m,
                     n_features=n_features, store_sparse=store_sparse)
        # we append only the list forming the row (out_blocks depth=2)
        x_blocks.append(out_blocks[0])
        y_blocks.append(out_blocks[1])

    x = Array(x_blocks, block_size=block_size, shape=(n_rows, n_features),
              sparse=store_sparse)

    # y has only a single line but it's treated as a 'column'
    y = Array(y_blocks, block_size=(n, 1), shape=(n_rows, 1), sparse=False)

    return x, y


@task(out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _read_libsvm(lines, out_blocks, col_size, n_features, store_sparse):
    from tempfile import SpooledTemporaryFile
    from sklearn.datasets import load_svmlight_file

    # Creating a tmp file to use load_svmlight_file method should be more
    # efficient than parsing the lines manually
    tmp_file = SpooledTemporaryFile(mode="wb+", max_size=2e8)

    tmp_file.writelines(lines)

    tmp_file.seek(0)

    x, y = load_svmlight_file(tmp_file, n_features)
    if not store_sparse:
        x = x.toarray()

    # tried also converting to csc/ndarray first for faster splitting but it's
    # not worth. Position 0 contains the X
    for i in range(ceil(n_features / col_size)):
        out_blocks[0][i] = x[:, i * col_size:(i + 1) * col_size]

    # Position 1 contains the y block
    out_blocks[1][0] = y.reshape(-1, 1)

    print("X length: %s" % len(out_blocks[0]))
    print("y length: %s" % len(out_blocks[1]))


class Array(object):
    # """ A dataset containing samples and, optionally, labels that can be
    # stored in a distributed manner.
    #
    # Dataset works as a list of Subset instances, which can be future objects
    # stored remotely. Accessing Dataset.labels and Dataset.samples runs
    # collect() and transfers all the data to the local machine.
    #
    # Parameters
    # ----------
    # n_features : int
    #     Number of features of the samples.
    # sparse : boolean, optional (default=False)
    #     Whether this dataset uses sparse data structures.
    #
    # Attributes
    # ----------
    # n_features : int
    #     Number of features of the samples.
    # _samples : ndarray
    #     Samples of the dataset.
    # _labels : ndarray
    #     Labels of the samples.
    # sparse: boolean
    #     True if this dataset uses sparse data structures.
    # """

    # TODO: after implementing more constructors decide a better way to avoid
    # having to synchronize bot_right block size to compute shape
    def __init__(self, blocks, block_size, shape, sparse):
        self._validate_blocks(blocks)

        self._blocks = blocks
        self._block_size = block_size
        self._blocks_shape = (len(blocks), len(blocks[0]))
        self._shape = shape
        self._sparse = sparse

    @staticmethod
    def _validate_blocks(blocks):
        if len(blocks) == 0 or len(blocks[0]) == 0:
            raise AttributeError('Blocks must a list of lists, with at least'
                                 ' an empty numpy/scipy matrix.')
        row_length = len(blocks[0])
        for i in range(1, len(blocks)):
            if len(blocks[i]) != row_length:
                raise AttributeError(
                    'All rows must contain the same number of blocks.')

    @staticmethod
    def _merge_blocks(blocks):
        sparse = None
        b0 = blocks[0][0]
        if sparse is None:
            sparse = issparse(b0)

        if sparse:
            ret = sp.bmat(blocks, format=b0.getformat(), dtype=b0.dtype)
        else:
            ret = np.block(blocks)

        return ret

    @staticmethod
    def _get_out_blocks(x, y):
        return [[object() for _ in range(y)] for _ in range(x)]

    # def __getitem__(self, item):
    #     return self._subsets.__getitem__(item)
    #
    # def __len__(self):
    #     return len(self._subsets)
    #
    # def __iter__(self):
    #     for j in range(len(self._blocks)):
    #         yield _collect_blocks(self._blocks[j], axis=1)

    def _get_row_shape(self, row_idx):
        if row_idx < self._blocks_shape[0] - 1:
            return self._block_size[0], self.shape[1]

        # this is the last chunk of rows, number of rows might be smaller
        n_rows = self.shape[0] - \
                 (self._blocks_shape[0] - 1) * self._block_size[0]
        return n_rows, self.shape[1]

    def _get_col_shape(self, col_idx):
        if col_idx < self._blocks_shape[1] - 1:
            return self.shape[0], self._block_size[1]

        # this is the last chunk of cols, number of cols might be smaller
        n_cols = self.shape[1] - \
                 (self._blocks_shape[1] - 1) * self._block_size[1]
        return self.shape[0], n_cols

    @property
    def shape(self):
        return self._shape

    def iterator(self, axis=0):
        # iterate through rows
        if axis == 0 or axis == 'rows':
            for i, row in enumerate(self._blocks):
                row_shape = self._get_row_shape(i)
                yield Array(blocks=[row], block_size=self._block_size,
                            shape=row_shape, sparse=self._sparse)

        # iterate through columns
        elif axis == 1 or axis == 'columns':
            for j in range(self._blocks_shape[1]):
                col_shape = self._get_col_shape(j)
                col_blocks = [[self._blocks[i][j]] for i in
                              range(self._blocks_shape[0])]
                yield Array(blocks=col_blocks, block_size=self._block_size,
                            shape=col_shape, sparse=self._sparse)

        else:
            raise Exception(
                "Axis must be [0|'rows'] or [1|'columns']. Got: %s" % axis)

    def transpose(self, mode='auto'):

        if mode == 'all':
            n, m = self._blocks_shape[0], self._blocks_shape[1]
            out_blocks = self._get_out_blocks(n, m)
            _tranpose(self._blocks, out_blocks)
        elif mode == 'rows':
            out_blocks = []
            for r in self.iterator(axis=0):
                _blocks = self._get_out_blocks(*r._blocks_shape)
                _tranpose(r._blocks, _blocks)
                out_blocks.append(_blocks[0])
        elif mode == 'columns':
            out_blocks = [[] for _ in range(self._blocks_shape[0])]
            for i, c in enumerate(self.iterator(axis=1)):
                _blocks = self._get_out_blocks(*c._blocks_shape)
                _tranpose(c._blocks, _blocks)
                for i2 in range(len(_blocks)):
                    out_blocks[i2].append(_blocks[i2][0])
        else:
            raise Exception(
                "Unknown transpose mode '%s'. Options are: [all|rows|columns]"
                % mode)

        blocks_t = list(map(list, zip(*out_blocks)))

        bn, bm = self._block_size[0], self._block_size[1]

        new_shape = self.shape[1], self.shape[0]
        # notice block_size is transposed
        return Array(blocks_t, block_size=(bm, bn), shape=new_shape,
                     sparse=self._sparse)

    def collect(self):
        self._blocks = compss_wait_on(self._blocks)
        return self._merge_blocks(self._blocks)


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _tranpose(blocks, out_blocks):
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            out_blocks[i][j] = blocks[i][j].transpose()

#
#
# def min_features(self):
#     """ Returns the minimum value of each feature in the dataset. This
#     method might compute the minimum and perform a synchronization.
#
#     Returns
#     -------
#     min_features : array, shape = [n_features,]
#         Array representing the minimum value that each feature takes in
#         the dataset.
#     """
#     if self._min_features is None:
#         self._compute_min_max()
#
#     return self._min_features
#
# def max_features(self):
#     """ Returns the maximum value of each feature in the dataset. This
#     method might compute the maximum and perform a synchronization.
#
#     Returns
#     -------
#     max_features : array, shape = [n_features,]
#         Array representing the maximum value that each feature takes in
#         the dataset.
#     """
#     if self._max_features is None:
#         self._compute_min_max()
#
#     return self._max_features
#
#
# def _reset_attributes(self):
#     self._max_features = None
#     self._min_features = None
#     self._samples = None
#     self._labels = None
#
# def _compute_min_max(self):
#     minmax = []
#
#     for subset in self._subsets:
#         minmax.append(_get_min_max(subset))
#
#     minmax = compss_wait_on(minmax)
#     self._min_features = np.nanmin(minmax, axis=0)[0]
#     self._max_features = np.nanmax(minmax, axis=0)[1]

#
# @task(returns=np.array)
# def _get_min_max(subset):
#     mn = np.min(subset.samples, axis=0)
#     mx = np.max(subset.samples, axis=0)
#
#     if issparse(subset.samples):
#         mn = mn.toarray()[0]
#         mx = mx.toarray()[0]
#
#     return np.array([mn, mx])
