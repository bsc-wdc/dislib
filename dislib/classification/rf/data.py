from numpy.lib import format
import numpy as np
import tempfile

from pycompss.api.parameter import FILE_IN, FILE_INOUT
from pycompss.api.task import task

from dislib.data import Dataset


@task(labels_path=FILE_IN, returns=3)
def get_labels(labels_path):
    y = np.genfromtxt(labels_path, dtype=None, encoding='utf-8')
    def_intp = np.intp  # Save default np.intp
    np.intp = np.int8  # Modify np.intp, so that codes.dtype=int8
    categories, codes = np.unique(y, return_inverse=True)
    np.intp = def_intp  # Reset np.intp to default value
    return codes, categories, len(categories)


class NpyFile(object):
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
        with open(self.path, 'rb') as fp:
            version = format.read_magic(fp)
            try:
                format._check_version(version)
            except ValueError:
                raise ValueError('Invalid file format.')
            header_data = format._read_array_header(fp, version)
            self.shape, self.fortran_order, self.dtype = header_data


class RfDataset(object):
    """
    A dataset formatted for the random forest algorithm of this library.
    """
    def __init__(self, samples_path, labels_path, features_path=None):
        """
        Constructor for RfDataset.

        :param samples_path:  Path of the .npy file containing the 2-d array of
                              samples. Must be a str or a future COMPSs object.
                              If it is the later, self.n_samples and
                              self.n_features must be set manually (they can be
                              future objects).
        :param labels_path:   Path of the .dat file containing the 1-d array of
                              labels. Must be a str or a future COMPSs object.
        :param features_path: Optional. Path of the .npy file containing the
                              2-d array of samples transposed. The array must
                              be C-ordered. Providing this array may improve
                              the performance as it allows sequential access to
                              the features.
        """
        self.samples_path = samples_path
        self.labels_path = labels_path
        self.features_path = features_path
        self.n_samples = None
        self.n_features = None

        self.y_codes = None
        self.y_categories = None
        self.n_classes = None

    def get_n_samples(self):
        if self.n_samples is None:
            if not isinstance(self.samples_path, str):
                raise AssertionError('Invalid state for self.samples_path and '
                                     'self.n_samples. If self.samples_path is '
                                     'a future COMPSs object, self.n_samples '
                                     'and self.n_features must be set '
                                     'manually.')
            shape = NpyFile(self.samples_path).get_shape()
            if len(shape) != 2:
                raise ValueError('Cannot read 2D array from the samples file.')
            self.n_samples, self.n_features = shape
        return self.n_samples

    def get_n_features(self):
        if self.n_features is None:
            shape = NpyFile(self.samples_path).get_shape()
            if len(shape) != 2:
                raise ValueError('Cannot read 2D array from the samples file.')
            self.n_samples, self.n_features = shape
        return self.n_features

    def get_y_codes(self):
        if self.y_codes is None:
            labels = get_labels(self.labels_path)
            self.y_codes, self.y_categories, self.n_classes = labels
        return self.y_codes

    def get_classes(self):
        if self.y_categories is None:
            labels = get_labels(self.labels_path)
            self.y_codes, self.y_categories, self.n_classes = labels
        return self.y_categories

    def get_n_classes(self):
        if self.n_classes is None:
            labels = get_labels(self.labels_path)
            self.y_codes, self.y_categories, self.n_classes = labels
        return self.n_classes

    def validate_features_file(self):
        features_npy_file = NpyFile(self.features_path)
        shape = features_npy_file.get_shape()
        fortran_order = features_npy_file.get_fortran_order()
        if len(shape) != 2:
            raise ValueError('Cannot read 2D array from features_file.')
        if (self.get_n_features(), self.get_n_samples()) != shape:
            raise ValueError('Invalid dimensions for the features_file.')
        if fortran_order:
            raise ValueError('Fortran order not supported for features array.')


def transform_to_rf_dataset(dataset: Dataset) -> RfDataset:
    samples_shapes = []
    for subset in dataset:
        samples_shapes.append(get_samples_shape(subset))
    samples_shapes, n_samples, n_features = merge_shapes(*samples_shapes)

    samples_file = tempfile.NamedTemporaryFile(mode='wb',
                                               prefix='tmp_rf_samples_',
                                               delete=False)
    samples_path = samples_file.name
    samples_file.close()
    allocate_samples_file(samples_path, n_samples, n_features)
    for i, subset in enumerate(dataset):
        fill_samples_file(samples_path, i, subset, samples_shapes)

    labels_file = tempfile.NamedTemporaryFile(mode='w',
                                              prefix='tmp_rf_labels_',
                                              delete=False)
    labels_path = labels_file.name
    labels_file.close()
    for subset in dataset:
        fill_labels_file(labels_path, subset)

    rf_dataset = RfDataset(samples_path, labels_path)
    rf_dataset.n_samples = n_samples
    rf_dataset.n_features = n_features
    return rf_dataset


@task(returns=1)
def get_samples_shape(subset):
    return subset.samples.shape


@task(returns=3)
def merge_shapes(*samples_shapes):
    n_samples = 0
    n_features = samples_shapes[0][1]
    for shape in samples_shapes:
        n_samples += shape[0]
        assert shape[1] == n_features, 'Subsets with different n_features.'
    return samples_shapes, n_samples, n_features


@task(samples_path=FILE_INOUT)
def allocate_samples_file(samples_path, n_samples, n_features):
    np.lib.format.open_memmap(samples_path, mode='w+', dtype='float32',
                              shape=(n_samples, n_features))


@task(samples_path=FILE_INOUT)
def fill_samples_file(samples_path, i, subset, samples_shapes):
    samples = np.lib.format.open_memmap(samples_path, mode='r+')
    first = sum(shape[0] for shape in samples_shapes[0:i])
    ss_samples = subset.samples.astype(dtype='float32', casting='same_kind')
    samples[first:first+samples_shapes[i][0]] = ss_samples


@task(labels_path=FILE_INOUT)
def fill_labels_file(labels_path, subset):
    with open(labels_path, 'at') as f:
        np.savetxt(f, subset.labels, fmt='%s', encoding='utf-8')
