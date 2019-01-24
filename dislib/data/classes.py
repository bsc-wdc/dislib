from uuid import uuid4

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy.sparse import issparse, vstack, csr_matrix


class Dataset(object):
    """ A dataset containing samples and, optionally, labels that can be
    stored in a distributed manner.

    Dataset works as a list of Subset instances, which can be future objects
    stored remotely. Accessing Dataset.labels and Dataset.samples runs
    collect() and transfers all the data to the local machine.

    Parameters
    ----------
    n_features : int
        Number of features of the samples.

    Attributes
    ----------
    n_features : int
        Number of features of the samples.
    samples : ndarray
        Samples of the dataset.
    labels : ndarray
        Labels of the samples.
    """

    def __init__(self, n_features):
        self._subsets = list()
        self.n_features = n_features
        self._sizes = list()
        self._max_features = None
        self._min_features = None
        self._samples = None
        self._labels = None

    def __getitem__(self, item):
        return self._subsets.__getitem__(item)

    def __len__(self):
        return len(self._subsets)

    def __iter__(self):
        return self._subsets.__iter__()

    def append(self, subset, n_samples=None):
        """ Appends a Subset to this Dataset.

        Parameters
        ----------
        subset : Subset
            Subset to add to this Dataset.
        n_samples : int, optional (default=None)
            Number of samples in subset.
        """
        self._subsets.append(subset)
        self._sizes.append(n_samples)
        self._reset_attributes()

    def extend(self, subsets):
        """ Appends one or more Subset instances to this Dataset.

        Parameters
        ----------
        subsets : list
            A list of Subset instances.
        """
        self._subsets.extend(subsets)
        self._sizes.extend([None] * len(subsets))
        self._reset_attributes()

    def subset_size(self, index):
        """ Returns the number of samples in the Subset referenced by index.
        If the size is unknown, this method performs a synchronization on
        Subset.samples.shape[0].

        Parameters
        ----------
        index : int
            Index of the Subset.

        Returns
        -------
        n_samples : int
            Number of samples.
        """
        if self._sizes[index] is None:
            size = compss_wait_on(_subset_size(self._subsets[index]))
            self._sizes[index] = size
        elif type(self._sizes[index]) != int:
            self._sizes = compss_wait_on(self._sizes)

        return self._sizes[index]

    def min_features(self):
        """ Returns the minimum value of each feature in the dataset. This
        method might compute the minimum and perform a synchronization.

        Returns
        -------
        min_features : array, shape = [n_features,]
            Array representing the minimum value that each feature takes in
            the dataset.
        """
        if self._min_features is None:
            self._compute_min_max()

        return self._min_features

    def max_features(self):
        """ Returns the maximum value of each feature in the dataset. This
        method might compute the maximum and perform a synchronization.

        Returns
        -------
        max_features : array, shape = [n_features,]
            Array representing the maximum value that each feature takes in
            the dataset.
        """
        if self._max_features is None:
            self._compute_min_max()

        return self._max_features

    def collect(self):
        self._subsets = compss_wait_on(self._subsets)

    @property
    def labels(self):
        self._update_labels()
        return self._labels

    @property
    def samples(self):
        self._update_samples()
        return self._samples

    def _reset_attributes(self):
        self._max_features = None
        self._min_features = None
        self._samples = None
        self._labels = None

    def _compute_min_max(self):
        minmax = []

        for subset in self._subsets:
            minmax.append(_get_min_max(subset))

        minmax = compss_wait_on(minmax)
        self._min_features = np.nanmin(minmax, axis=0)[0]
        self._max_features = np.nanmax(minmax, axis=0)[1]

    def _update_labels(self):
        self.collect()
        labels_list = []

        for subset in self._subsets:
            if subset.labels is not None:
                labels_list.append(subset.labels)

        if len(labels_list) > 0:
            self._labels = np.concatenate(labels_list)

    def _update_samples(self):
        self.collect()
        self._samples = np.empty((0, self.n_features))

        for subset in self._subsets:
            self._samples = np.concatenate((self._samples, subset.samples))


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
        Samples.
    labels : ndarray
        Labels.
    """

    def __init__(self, samples, labels=None):
        if issparse(samples):
            self.samples = csr_matrix(samples)
        else:
            self.samples = np.array(samples)

        if labels is not None:
            self.labels = np.array(labels)
        else:
            self.labels = None

        idx = [uuid4().int for _ in range(samples.shape[0])]
        self._ids = np.array(idx)

    def copy(self):
        """ Return a copy of this Subset

        Returns
        -------
        subset : Subset
            A copy of this Subset.
        """

        subset = Subset(samples=self.samples, labels=self.labels)
        subset._ids = np.array(self._ids)
        return subset

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


@task(returns=int)
def _subset_size(subset):
    return subset.samples.shape[0]


@task(returns=np.array)
def _get_min_max(subset):
    mn = np.min(subset.samples, axis=0)
    mx = np.max(subset.samples, axis=0)

    if issparse(subset.samples):
        print(mn.shape)
        mn = mn.toarray()
        print(mn.shape)
        mx = mx.toarray()

    return np.array([mn, mx])
