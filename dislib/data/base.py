import os

import numpy as np
from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task

from dislib.data import Dataset


def load_file(path, part_size, fmt="labeled", n_features=None, use_array=False):
    """ Loads and parses a CSV or LibSVM file.

    Parameters
    ----------
    path : string
        File path
    part_size : int
        Partition size in lines
    fmt : string, optional (default = 'labeled')
        Format of the text file. It can be 'labeled' for labeled CSV files
        with the label at the last column, 'unlabeled' for unlabeled CSV
        files, and 'libsvm' for LibSVM/SVMLight files.
    n_features : int, optional
        Number of features. This parameter is mandatory for LibSVM files.
    use_array : boolean, optional (default = False)
        Whether to convert LibSVM files to numpy.array. If False, scipy.sparse
        data structures are employed.

    Returns
    -------
    datasets : list
        A list of Dataset instances of size part_size.
    """

    assert (n_features or fmt != "libsvm"), \
        "Number of features must be specified for LibSVM files."

    lines = []
    parts = []
    idx = 0

    with open(path, "r") as f:
        for line in f:
            lines.append(line.encode())
            idx += 1

            if idx == part_size:
                parts.append(_read_lines(lines, fmt, n_features, use_array))
                lines = []
                idx = 0

    if lines:
        parts.append(_read_lines(lines, fmt, n_features, use_array))

    return parts


def load_files(path, fmt="labeled", n_features=None, use_array=False):
    """ Loads and parses a set of CSV or LibSVM files.

    path : string
        Path to a directory containing input files.
    fmt : string, optional (default = 'labeled')
        Format of the text files. It can be 'labeled' for labeled CSV files
        with the label at the last column, 'unlabeled' for unlabeled CSV
        files, and 'libsvm' for LibSVM/SVMLight files.
    n_features : int, optional
        Number of features. This parameter is mandatory for LibSVM files.
    use_array: boolean, optional (default = False)
        Whether to convert LibSVM files to numpy.array. If False, scipy.sparse
        data structures are employed.

    Returns
    -------
     datasets: list
        A list of Dataset instances. One instance per file in ``path''.
    """
    assert (n_features or fmt != "libsvm"), \
        "Number of features must be specified for LibSVM files."

    files = os.listdir(path)
    parts = []

    for file in files:
        parts.append(_read_file(os.path.join(path, file), fmt, n_features,
                                use_array))

    return parts


@task(returns=1)
def _read_lines(lines, fmt, n_features, use_array):
    if fmt == "libsvm":
        data = _read_libsvm(lines, n_features, use_array)
    elif fmt == "unlabeled":
        data = Dataset(np.genfromtxt(lines, delimiter=","))
    else:
        vecs = np.genfromtxt(lines, delimiter=",")
        data = Dataset(vecs[:, :-1], vecs[:, -1])

    return data


@task(file=FILE_IN, returns=1)
def _read_file(file, fmt, n_features, use_array):
    from sklearn.datasets import load_svmlight_file

    if fmt == "libsvm":
        x, y = load_svmlight_file(file, n_features)

        if use_array:
            x = x.toarray()

        data = Dataset(x, y)

    elif fmt == "unlabeled":
        data = Dataset(np.loadtxt(file, delimiter=","))
    else:
        vecs = np.loadtxt(file, delimiter=",")
        data = Dataset(vecs[:, :-1], vecs[:, -1])

    return data


def _read_libsvm(lines, n_features, use_array):
    from tempfile import SpooledTemporaryFile
    from sklearn.datasets import load_svmlight_file
    # Creating a tmp file to use load_svmlight_file method should be more
    # efficient than parsing the lines manually
    tmp_file = SpooledTemporaryFile(mode="wb+", max_size=2e8)
    tmp_file.writelines(lines)
    tmp_file.seek(0)
    x, y = load_svmlight_file(tmp_file, n_features)

    if use_array:
        x = x.toarray()

    data = Dataset(x, y)
    return data
