from uuid import uuid4

import numpy as np
from pycompss.api.api import compss_wait_on
from scipy.sparse import issparse, vstack


class Dataset(object):
    def __init__(self, n_features):
        self._subsets = list()
        self.n_features = n_features

    def __getitem__(self, item):
        return self._subsets.__getitem__(item)

    def __len__(self):
        return len(self._subsets)

    def append(self, subset):
        self._subsets.append(subset)

    def extend(self, *subsets):
        self._subsets.extend(subsets)

    def collect(self):
        self._subsets = compss_wait_on(self._subsets)


class Subset(object):
    """ A subset of data for machine learning.

    Parameters
    ----------
        samples : ndarray
            Array of shape (n_samples, n_features).
        labels : ndarray, optional
            Array of shape (n_samples)

    Attributes
    ----------
    samples : ndarray

    labels : ndarray


    Methods
    -------
    concatenate(subset, remove_duplicates=False)
        Vertically concatenates this Subset to another.
    """

    def __init__(self, samples, labels=None):
        self.samples = samples
        self.labels = labels

        idx = [uuid4().int for _ in range(samples.shape[0])]
        self._ids = np.array(idx)

    def concatenate(self, subset, remove_duplicates=False):
        """ Vertically concatenates this Subset to another.

        Parameters
        ----------
        subset : Subset
            Subset to concatenate.
        remove_duplicates : boolean, optional (default=False)
            Whether to remove duplicate samples.
        """
        assert issparse(self.samples) == issparse(subset.samples), \
            "Cannot concatenate sparse data with non-sparse data."
        assert (self.labels is None) == (subset.labels is None), \
            "Cannot concatenate labeled data with non-labeled data"

        if issparse(self.samples):
            self.samples = vstack([self.samples, subset.samples])
        else:
            self.samples = np.concatenate([self.samples, subset.samples])

        if self.labels is not None:
            self.labels = np.concatenate([self.labels, subset.labels])

        self._ids = np.concatenate([self._ids, subset._ids])

        if remove_duplicates:
            self._ids, uniques = np.unique(self._ids, return_index=True)
            self.samples = self.samples[uniques]

            if self.labels is not None:
                self.labels = self.labels[uniques]

    def set_label(self, index, label):
        """ Sets sample labels.

        Parameters
        ----------
        index : int or sequence of ints
            Indices of the target samples.
        label : float
            Label value.

        Notes
        -----
        If the Subset does not contain labels, this method initializes all
        labels different from ``index'' to ``None''.
        """
        if self.labels is None:
            self.labels = np.array([None] * self.samples.shape[0])

        self.labels[index] = label

    def __getitem__(self, item):
        if self.labels is not None:
            subset = Subset(self.samples[item], self.labels[item])
        else:
            subset = Subset(self.samples[item])

        subset._ids = self._ids[item]
        return subset
