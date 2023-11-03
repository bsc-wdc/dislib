import tempfile
import numpy as np
from numpy.lib import format
from pycompss.api.constraint import constraint
from pycompss.api.parameter import (
    FILE_IN,
    FILE_INOUT,
    COLLECTION_IN,
    Depth,
    Type,
)
from pycompss.api.task import task
from dislib.data.array import Array


class RfBaseDataset:
    """Base class for Dataset format used by the fit() of the
    RandomForestRegressor and RandomForestClassifier.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self, samples_path, targets_path, features_path=None):
        self.samples_path = samples_path
        self.targets_path = targets_path
        self.features_path = features_path
        self.n_samples = None
        self.n_features = None
        self.y_targets = None

    def get_n_samples(self):
        """Gets the number of samples obtained from the samples file.

        Returns
        -------
        n_samples: int

        Raises
        ------
        AssertionError
            If self.n_samples is None and self.samples_path is not a string.
        ValueError
            If invalid content is encountered in the samples file.
        """
        if self.n_samples is None:
            if not isinstance(self.samples_path, str):
                raise TypeError(
                    "self.n_samples must be set manually if self.samples_path "
                    "is a pycompss.runtime.Future object"
                )
            shape = _NpyFile(self.samples_path).get_shape()
            self.n_samples, self.n_features = shape
        return self.n_samples

    def get_n_features(self):
        """Gets the number of features obtained from the samples file.

        Returns
        -------
        n_features: int

        Raises
        ------
        AssertionError
            If self.n_features is None and self.samples_path is not a string.
        ValueError
            If invalid content is encountered in the samples file.
        """
        if self.n_features is None:
            if not isinstance(self.samples_path, str):
                raise TypeError(
                    "self.n_features must be set manually if self.samples_path"
                    " is a pycompss.runtime.Future object"
                )
            shape = _NpyFile(self.samples_path).get_shape()
            self.n_samples, self.n_features = shape
        return self.n_features

    def validate_features_file(self):
        """Validates the features file header information.

        Raises
        ------
        ValueError
            If the shape of the array in the features_file doesn't match this
            class n_samples and n_features or if the array is in fortran order.
        """
        features_npy_file = _NpyFile(self.features_path)
        shape = features_npy_file.get_shape()
        fortran_order = features_npy_file.get_fortran_order()
        if (self.get_n_features(), self.get_n_samples()) != shape:
            raise ValueError("Invalid dimensions for the features_file.")
        if fortran_order:
            raise ValueError("Fortran order not supported for features array.")


class RfClassifierDataset(RfBaseDataset):
    """Dataset format used by the fit() of the RandomForestClassifier.

    The RfDataset contains a file path for the samples and another one for the
    labels. Optionally, a path can be provided for a transposed version of the
    samples matrix, i.e., the features.

    Note: For a representation of a dataset distributed in multiple files, use
    dislib.data.Dataset instead.

    Parameters
    ----------
    samples_path: str
        Path of the .npy file containing the 2-d array of samples. It can be a
        pycompss.runtime.Future object. If so, self.n_samples and
        self.n_features must be set manually (they can also be
        pycompss.runtime.Future objects).
    targets_path: str
        Path of the .dat file containing the 1-d array of target labels.
        It can be a pycompss.runtime.Future object.
    features_path: str, optional (default=None)
        Path of the .npy file containing the 2-d array of samples transposed.
        The array must be C-ordered. Providing this array may improve the
        performance as it allows sequential access to the features.

    Attributes
    ----------
    n_samples: int
        The number of samples of the dataset. It can be a
        pycompss.runtime.Future object.
    n_features: int
        The number of features of the dataset. It can be a
        pycompss.runtime.Future object.
    y_targets: ndarray
        The codified array of labels for this RfDataset. The values are indices
        of the array of classes, which contains the corresponding labels. The
        dtype is np.int8. It can be a pycompss.runtime.Future object.
    y_categories: ndarray
        The array of classes for this RfDataset. The values are unique. It can
        be a pycompss.runtime.Future object.
    n_classes: int
        The number of classes of this RfDataset. It can be a
        pycompss.runtime.Future object.

    """

    def __init__(self, samples_path, targets_path, features_path=None):
        super().__init__(samples_path, targets_path, features_path)
        self.y_categories = None
        self.n_classes = None

    def get_y_targets(self):
        """Obtains the codified array of target labels.

        Returns
        -------
        y_targets: ndarray
        """
        if self.y_targets is None:
            labels = _get_labels(self.targets_path)
            self.y_targets, self.y_categories, self.n_classes = labels
        return self.y_targets

    def get_classes(self):
        """Obtains the array of label categories.

        Returns
        -------
        y_categories: ndarray
        """
        if self.y_categories is None:
            labels = _get_labels(self.targets_path)
            self.y_targets, self.y_categories, self.n_classes = labels
        return self.y_categories

    def get_n_classes(self):
        """Obtains the number of classes.

        Returns
        -------
        n_classes: int
        """
        if self.n_classes is None:
            labels = _get_labels(self.targets_path)
            self.y_targets, self.y_categories, self.n_classes = labels
        return self.n_classes


class RfRegressorDataset(RfBaseDataset):
    """Dataset format used by the fit() of the RandomForestRegressor.

    The RfDataset contains a file path for the samples and another one for the
    targets. Optionally, a path can be provided for a transposed version of the
    samples matrix, i.e., the features.

    Note: For a representation of a dataset distributed in multiple files, use
    dislib.data.Dataset instead.

    Parameters
    ----------
    samples_path: str
        Path of the .npy file containing the 2-d array of samples. It can be a
        pycompss.runtime.Future object. If so, self.n_samples and
        self.n_features must be set manually (they can also be
        pycompss.runtime.Future objects).
    targets_path: str
        Path of the .dat file containing the 1-d array of target values.
        It can be a pycompss.runtime.Future object.
    features_path: str, optional (default=None)
        Path of the .npy file containing the 2-d array of samples transposed.
        The array must be C-ordered. Providing this array may improve the
        performance as it allows sequential access to the features.

    Attributes
    ----------
    n_samples: int
        The number of samples of the dataset. It can be a
        pycompss.runtime.Future object.
    n_features: int
        The number of features of the dataset. It can be a
        pycompss.runtime.Future object.
    y_targets: ndarray
        The array of targets for this RfDataset. It can be a
        pycompss.runtime.Future object.

    """

    def __init__(self, samples_path, targets_path, features_path=None):
        super().__init__(samples_path, targets_path, features_path)

    def get_y_targets(self):
        """Obtains the array of target values.

        Returns
        -------
        y_targets: ndarray
        """
        if self.y_targets is None:
            targets = _get_values(self.targets_path)
            self.y_targets = targets
        return self.y_targets

    def get_n_classes(self):
        return None

    def get_classes(self):
        return None


def transform_to_rf_dataset(
    x: Array,
    y: Array,
    base_dataset: RfRegressorDataset or RfClassifierDataset,
    features_file=False,
) -> RfRegressorDataset or RfClassifierDataset:
    """Creates a RfDataset object from samples x and targets y.

    This function creates a `RfDataset` by saving
    x and y in files.

    Parameters
    ----------
    x: ds-array, shape = (n_samples, n_features)
        The training input samples.
    y: ds-array, shape = (n_samples,) or (n_samples, n_outputs)
        The target values.
    task: {"classification", "regression"}
        Task of the Random Forest.

    Returns
    -------
    rf_dataset: dislib.regression.rf._data.RfDataset

    """
    n_samples = x.shape[0]
    n_features = x.shape[1]

    # Samples
    samples_file = tempfile.NamedTemporaryFile(
        mode="wb", prefix="tmp_rf_samples_", delete=False
    )
    samples_path = samples_file.name
    samples_file.close()
    _allocate_samples_file(samples_path, n_samples, n_features)

    start_idx = 0
    row_blocks_iterator = x._iterator(axis=0)
    top_row = next(row_blocks_iterator)
    _fill_samples_file(samples_path, top_row._blocks, start_idx)
    start_idx += x._top_left_shape[0]
    blocks = []
    for idx, x_row in enumerate(row_blocks_iterator):
        blocks.extend(x_row._blocks)
        if idx != 0 and idx % 20 == 0 or len(x._blocks) - 3 == idx:
            _fill_samples_file(samples_path, blocks, start_idx)
            start_idx += x._reg_shape[0] * len(blocks)
            blocks = []
    if len(blocks) > 0:
        _fill_samples_file(samples_path, blocks, start_idx)

    # Targets
    targets_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="tmp_rf_targets_", delete=False
    )
    targets_path = targets_file.name
    targets_file.close()
    row_blocks_iterator = y._iterator(axis=0)
    top_row = next(row_blocks_iterator)
    _fill_targets_file(targets_path, top_row._blocks)
    blocks = []
    for idx, y_row in enumerate(row_blocks_iterator):
        blocks.extend(y_row._blocks)
        if idx != 0 and idx % 20 == 0 or len(y._blocks) - 3 == idx:
            _fill_targets_file(targets_path, blocks)
            blocks = []
    if len(blocks) > 0:
        _fill_targets_file(targets_path, blocks)

    # Features
    if features_file:
        features_file = tempfile.NamedTemporaryFile(
            mode="wb", prefix="tmp_rf_features_", delete=False
        )
        features_path = features_file.name
        features_file.close()
        _allocate_features_file(features_path, n_samples, n_features)

        start_idx = 0
        row_blocks_iterator = x._iterator(axis=0)
        top_row = next(row_blocks_iterator)
        _fill_features_file(features_path, top_row._blocks, start_idx)
        start_idx += x._top_left_shape[0]
        for x_row in row_blocks_iterator:
            _fill_features_file(features_path, x_row._blocks, start_idx)
            start_idx += x._reg_shape[0]
    else:
        features_path = None

    rf_dataset = base_dataset(samples_path, targets_path, features_path)
    rf_dataset.n_samples = n_samples
    rf_dataset.n_features = n_features
    return rf_dataset


class _NpyFile(object):
    def __init__(self, path):
        self.path = path

        self.shape = None
        self.fortran_order = None
        self.dtype = None

    def get_shape(self):
        if self.shape is None:
            self._read_header()
        return self.shape

    def get_fortran_order(self):
        if self.fortran_order is None:
            self._read_header()
        return self.fortran_order

    def get_dtype(self):
        if self.dtype is None:
            self._read_header()
        return self.dtype

    def _read_header(self):
        with open(self.path, "rb") as fp:
            version = format.read_magic(fp)
            try:
                format._check_version(version)
            except ValueError:
                raise ValueError("Invalid file format.")
            header_data = format._read_array_header(fp, version)
            self.shape, self.fortran_order, self.dtype = header_data


@constraint(computing_units="${ComputingUnits}")
@task(targets_path=FILE_IN, returns=3)
def _get_labels(targets_path):
    # Classification
    y = np.genfromtxt(targets_path, dtype=None, encoding="utf-8")
    categories, codes = np.unique(y, return_inverse=True)
    return codes.astype(np.int8), categories, len(categories)


@constraint(computing_units="${ComputingUnits}")
@task(targets_path=FILE_IN, returns=1)
def _get_values(targets_path):
    # Regression
    y = np.genfromtxt(targets_path, dtype=None, encoding="utf-8")
    return y.astype(np.float64)


@constraint(computing_units="${ComputingUnits}")
@task(samples_path=FILE_INOUT)
def _allocate_samples_file(samples_path, n_samples, n_features):
    np.lib.format.open_memmap(
        samples_path,
        mode="w+",
        dtype="float32",
        shape=(int(n_samples), int(n_features)),
    )


@constraint(computing_units="${ComputingUnits}")
@task(samples_path=FILE_INOUT)
def _allocate_features_file(samples_path, n_samples, n_features):
    np.lib.format.open_memmap(
        samples_path,
        mode="w+",
        dtype="float32",
        shape=(int(n_features), int(n_samples)),
    )


@constraint(computing_units="${ComputingUnits}")
@task(samples_path=FILE_INOUT, row_blocks={Type: COLLECTION_IN, Depth: 3})
def _fill_samples_file(samples_path, row_blocks, start_idx):
    samples = np.lib.format.open_memmap(samples_path, mode="r+")
    rows_samples = Array._merge_blocks(row_blocks)
    rows_samples = rows_samples.astype(dtype="float32", casting="same_kind")
    samples[start_idx: start_idx + rows_samples.shape[0]] = rows_samples


@constraint(computing_units="${ComputingUnits}")
@task(samples_path=FILE_INOUT, row_blocks={Type: COLLECTION_IN, Depth: 3})
def _fill_features_file(samples_path, row_blocks, start_idx):
    samples = np.lib.format.open_memmap(samples_path, mode="r+")
    rows_samples = Array._merge_blocks(row_blocks)
    rows_samples = rows_samples.astype(dtype="float32", casting="same_kind")
    samples[:, start_idx: start_idx + rows_samples.shape[0]] = rows_samples.T


@constraint(computing_units="${ComputingUnits}")
@task(targets_path=FILE_INOUT, row_blocks={Type: COLLECTION_IN, Depth: 3})
def _fill_targets_file(targets_path, row_blocks):
    rows_targets = Array._merge_blocks(row_blocks)
    with open(targets_path, "at") as f:
        np.savetxt(f, rows_targets, fmt="%s", encoding="utf-8")
