from uuid import uuid4

import numpy as np
from scipy.sparse import issparse, vstack


class Dataset(object):
    """ Set of vectors with or without labels.

    Parameters
    ----------
        vectors : ndarray
            Array of shape (n_vectors, n_features).
        labels : ndarray, optional
            Array of shape (n_vectors)

    Attributes
    ----------
    vectors : ndarray

    labels : ndarray


    Methods
    -------
    concatenate(data, remove_duplicates=False)
        Vertically concatenates this Dataset to another.
    """

    def __init__(self, vectors, labels=None):
        self.vectors = vectors
        self.labels = labels

        idx = [uuid4().int for _ in range(vectors.shape[0])]
        self._ids = np.array(idx)

    def concatenate(self, data, remove_duplicates=False):
        """ Vertically concatenates this Dataset to another.

        Parameters
        ----------
        data : Dataset
            Dataset to concatenate.
        remove_duplicates : boolean, optional (default=False)
            Whether to remove duplicate vectors.
        """
        assert issparse(self.vectors) == issparse(data.vectors), \
            "Cannot concatenate sparse data with non-sparse data."
        assert (self.labels is None) == (data.labels is None), \
            "Cannot concatenate labeled data with non-labeled data"

        if issparse(self.vectors):
            self.vectors = vstack([self.vectors, data.vectors])
        else:
            self.vectors = np.concatenate([self.vectors, data.vectors])

        if self.labels is not None:
            self.labels = np.concatenate([self.labels, data.labels])

        self._ids = np.concatenate([self._ids, data._ids])

        if remove_duplicates:
            self._ids, uniques = np.unique(self._ids, return_index=True)
            self.vectors = self.vectors[uniques]
            self.labels = self.labels[uniques]

    def __getitem__(self, item):
        if self.labels is not None:
            ds = Dataset(self.vectors[item], self.labels[item])
        else:
            ds = Dataset(self.vectors[item])

        ds._ids = self._ids[item]
        return ds
