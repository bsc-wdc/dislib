from uuid import uuid4

import numpy as np
import scipy.sparse as sp
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy.sparse import issparse


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
    sparse : boolean, optional (default=False)
        Whether this dataset uses sparse data structures.

    Attributes
    ----------
    n_features : int
        Number of features of the samples.
    _samples : ndarray
        Samples of the dataset.
    _labels : ndarray
        Labels of the samples.
    sparse: boolean
        True if this dataset uses sparse data structures.
    """

    def __init__(self, n_features, sparse=False):
        self._subsets = list()
        self.n_features = n_features
        self._sizes = list()
        self._max_features = None
        self._min_features = None
        self._samples = None
        self._labels = None
        self._sparse = sparse

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

    def transpose(self, n_subsets=None):
        """ Transposes the Dataset.

        Parameters
        ----------
        n_subsets : int, optional (default=None)
            Number of subsets in the transposed dataset. If none, defaults to
            the original number of subsets

        Returns
        -------
        dataset_t: Dataset
            Transposed dataset divided by rows.
        """

        if n_subsets is None:
            n_subsets = len(self._subsets)

        subsets_t = []
        for i in range(n_subsets):
            subsets_i = [_get_split_i(s, i, n_subsets) for s in self._subsets]
            new_subset = _merge_split_subsets(self.sparse, *subsets_i)
            subsets_t.append(new_subset)

        n_rows = np.sum(self.subsets_sizes())

        dataset_t = Dataset(n_features=n_rows, sparse=self._sparse)

        dataset_t.extend(subsets_t)

        return dataset_t

    def _apply(self, f, sparse=None, return_dataset=False):
        """ Returns the result of applying function f to each sample of the
         dataset.
        Parameters
        ----------
        f : function
            Function to be applied to each samples.
        sparse: bool
            Whether the dataset to be returned should be in sparse format. If
            return_dataset == False, this parameter is ignored.
        return_dataset: bool
            Whether the results of applying function 'f' should be returned as
            a Dataset instance.

        Returns
        -------
        result : Dataset / list
            Result of applying f to each of the dataset's samples.
        """

        if sparse is None:
            sparse = self._sparse

        new_elems = []
        for i in range(len(self)):
            new_elems.append(_subset_apply(self._subsets[i], f,
                                           return_subset=return_dataset))

        if return_dataset:
            n_features = _subset_size(new_elems[0])

            dataset = Dataset(n_features=n_features, sparse=sparse)

            dataset.extend(new_elems)

            return dataset
        else:
            return new_elems

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

    def subsets_sizes(self):
        """ Returns the number of samples in all the Subsets.
        If the size is unknown, this method performs a synchronization on
        Subset.samples.shape[0] for all subsets.

        Returns
        -------
        subsets_sizes : ndarray
            Number of samples in each subset.
        """

        for index in range(len(self)):
            if self._sizes[index] is None:
                self._sizes[index] = _subset_size(self._subsets[index])

        self._sizes = compss_wait_on(self._sizes)

        return list(self._sizes)

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

    @property
    def sparse(self):
        return self._sparse

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
        if len(self._subsets) > 0:
            # use the first subset to init to keep the subset's dtype
            self._samples = self._subsets[0].samples

            concat_f = sp.vstack if self._sparse else np.concatenate

            for subset in self._subsets[1:]:
                self._samples = concat_f((self._samples, subset.samples))


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
        self.samples = samples.copy()

        if labels is not None:
            self.labels = np.array(labels)
        else:
            self.labels = None

        idx = [uuid4().int for _ in range(self.samples.shape[0])]
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
            self.samples = sp.vstack([self.samples, subset.samples])
        else:
            self.samples = np.concatenate([self.samples, subset.samples])

        if self.labels is not None:
            self.labels = np.concatenate([self.labels, subset.labels])

        self._ids = np.concatenate([self._ids, subset._ids])

        if remove_duplicates:
            self._ids, uniques = np.unique(self._ids, return_index=True)

            indices = np.argsort(uniques)
            uniques = uniques[indices]
            self._ids = self._ids[indices]

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


@task(returns=object)
def _subset_apply(subset, f, return_subset=False):
    samples = [f(sample) for sample in subset.samples]
    s = np.array(samples).reshape(len(samples), -1)

    if return_subset:
        s = Subset(samples=s)

    return s


@task(returns=np.array)
def _get_min_max(subset):
    mn = np.min(subset.samples, axis=0)
    mx = np.max(subset.samples, axis=0)

    if issparse(subset.samples):
        mn = mn.toarray()[0]
        mx = mx.toarray()[0]

    return np.array([mn, mx])


@task(returns=1)
def _get_split_i(subset, i, n_subsets):
    """
    Returns the columns corresponding to group i, if the subset is divided
    into n_subsets groups of columns.
    """

    # number of elements per group
    stride = subset.samples.shape[1] // n_subsets

    # if it's the last group, just add all remaining element (end = None)
    start_idx = i * stride
    end_idx = (i + 1) * stride
    if i == n_subsets - 1:
        end_idx = None

    samples_i = subset.samples[:, start_idx:end_idx]

    return samples_i


@task(returns=1)
def _merge_split_subsets(sparse, *split_subsets):
    stack_f = sp.vstack if sparse else np.vstack

    # each sublist (sl) contains samples with a subset of columns. Each
    # sublist must be stacked vertical first. Then all sublists must be
    # stacked among themselves forming the final columns.
    col_samples = stack_f([stack_f(sl) for sl in split_subsets])

    # finally we transpose the columns.
    return Subset(samples=col_samples.transpose())
