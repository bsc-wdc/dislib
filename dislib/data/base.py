import os

import numpy as np
from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task
from scipy.sparse import issparse

from dislib.data import Subset, Dataset


def load_data(x, subset_size, y=None):
    """
    Loads data into a Dataset.

    Parameters
    ----------
    x : ndarray, shape=[n_samples, n_features]
        Array of samples.
    y : ndarray, optional, shape=[n_features,]
        Array of labels.
    subset_size : int
        Subset size in number of samples.

    Returns
    -------
    dataset : Dataset
        A distributed representation of the data divided in Subsets of
        subset_size.
    """
    dataset = Dataset(n_features=x.shape[1], sparse=issparse(x))

    for i in range(0, x.shape[0], subset_size):
        if y is not None:
            subset = Subset(x[i: i + subset_size], y[i: i + subset_size])
        else:
            subset = Subset(x[i: i + subset_size])
        dataset.append(subset)

    return dataset


def load_libsvm_file(path, subset_size, n_features, store_sparse=True):
    """ Loads a LibSVM file into a Dataset.

     Parameters
    ----------
    path : string
        File path.
    subset_size : int
        Subset size in lines.
    n_features : int
        Number of features.
    store_sparse : boolean, optional (default = True).
        Whether to use scipy.sparse data structures to store data. If False,
        numpy.array is used instead.

    Returns
    -------
    dataset : Dataset
        A distributed representation of the data divided in Subsets of
        subset_size.
    """

    return _load_file(path, subset_size, fmt="libsvm",
                      store_sparse=store_sparse,
                      n_features=n_features)


def load_libsvm_files(path, n_features, store_sparse=True):
    """ Loads a set of LibSVM files into a Dataset.

        Parameters
       ----------
       path : string
           Path to a directory containing LibSVM files.
       n_features : int
           Number of features.
       store_sparse : boolean, optional (default = True).
           Whether to use scipy.sparse data structures to store data. If False,
           numpy.array is used instead.

       Returns
       -------
       dataset : Dataset
           A distributed representation of the data divided in a Subset for
           each file in path.
       """

    return _load_files(path, fmt="libsvm", store_sparse=store_sparse,
                       n_features=n_features)


def load_txt_file(path, subset_size, n_features, delimiter=",",
                  label_col=None):
    """ Loads a text file into a Dataset.

     Parameters
    ----------
    path : string
        File path.
    subset_size : int
        Subset size in lines.
    n_features : int
        Number of features.
    delimiter : string, optional (default ",")
        String that separates features in the file.
    label_col : int, optional (default=None)
        Column representing data labels. Can be 'first' or 'last'.

    Returns
    -------
    dataset : Dataset
        A distributed representation of the data divided in Subsets of
        subset_size.
    """
    return _load_file(path, subset_size, fmt="txt", n_features=n_features,
                      delimiter=delimiter, label_col=label_col)


def load_txt_files(path, n_features, delimiter=",", label_col=None):
    """ Loads a set of text files into a Dataset.

    Parameters
   ----------
    path : string
        Path to a directory containing text files.
    n_features : int
        Number of features.
    delimiter : string, optional (default ",")
        String that separates features in the file.
    label_col : int, optional (default=None)
        Column representing data labels. Can be 'first' or 'last'.

   Returns
   -------
   dataset : Dataset
       A distributed representation of the data divided in a Subset for
       each file in path.
   """

    return _load_files(path, fmt="txt", n_features=n_features,
                       delimiter=delimiter, label_col=label_col)


def _load_file(path, subset_size, fmt, n_features, delimiter=None,
               label_col=None, store_sparse=False):
    lines = []
    dataset = Dataset(n_features, store_sparse)

    with open(path, "r") as f:
        for line in f:
            lines.append(line.encode())

            if len(lines) == subset_size:
                subset = _read_lines(lines, fmt, n_features, delimiter,
                                     label_col, store_sparse)
                dataset.append(subset)
                lines = []

    if lines:
        dataset.append(_read_lines(lines, fmt, n_features, delimiter,
                                   label_col, store_sparse))

    return dataset


def _load_files(path, fmt, n_features, delimiter=None, label_col=None,
                store_sparse=False):
    assert os.path.isdir(path), "Path is not a directory."

    files = os.listdir(path)
    subsets = Dataset(n_features, store_sparse)

    for file_ in files:
        full_path = os.path.join(path, file_)
        subset = _read_file(full_path, fmt, n_features, delimiter, label_col,
                            store_sparse)
        subsets.append(subset)

    return subsets


@task(returns=1)
def _read_lines(lines, fmt, n_features, delimiter, label_col, store_sparse):
    if fmt == "libsvm":
        subset = _read_libsvm(lines, n_features, store_sparse)
    else:
        samples = np.genfromtxt(lines, delimiter=delimiter)

        if label_col == "first":
            subset = Subset(samples[:, 1:], samples[:, 0])
        elif label_col == "last":
            subset = Subset(samples[:, :-1], samples[:, -1])
        else:
            subset = Subset(samples)

    return subset


@task(file=FILE_IN, returns=1)
def _read_file(file, fmt, n_features, delimiter, label_col, store_sparse):
    from sklearn.datasets import load_svmlight_file

    if fmt == "libsvm":
        x, y = load_svmlight_file(file, n_features)

        if not store_sparse:
            x = x.toarray()

        subset = Subset(x, y)
    else:
        samples = np.genfromtxt(file, delimiter=delimiter)

        if label_col == "first":
            subset = Subset(samples[:, 1:], samples[:, 0])
        elif label_col == "last":
            subset = Subset(samples[:, :-1], samples[:, -1])
        else:
            subset = Subset(samples)

    return subset


def _read_libsvm(lines, n_features, store_sparse):
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

    data = Subset(x, y)
    return data
