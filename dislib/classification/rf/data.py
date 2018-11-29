from numpy.lib import format
import pandas as pd
from pandas import read_csv
from pandas.api.types import CategoricalDtype

from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task

from dislib.data import Dataset


@task(labels_path=FILE_IN, returns=3)
def get_labels(labels_path):
    y = read_csv(labels_path, dtype=CategoricalDtype(), header=None, squeeze=True).values
    return y.codes, y.categories, len(y.categories)


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
            self.shape, self.fortran_order, self.dtype = format._read_array_header(fp, version)


class RfDataset(object):
    """
    A dataset formatted for the random forest algorithm of this library.
    """
    def __init__(self, samples_path, labels_path, features_path=None):
        """
        Constructor for RfDataset.

        :param samples_path:  Path of the .npy file containing the 2-d array of samples. Must be a str or a future
                              COMPSs object. If it is the later, self.n_samples and self.n_features must be set
                              manually.
        :param labels_path:   Path of the .dat file containing the 1-d array of labels. Must be a str or a future COMPSs
                              object.
        :param features_path: Optional. Path of the .npy file containing the 2-d array of samples transposed. The array
                              must be C-ordered. Providing this array may improve the performance as it allows
                              sequential access to the features.
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
                raise AssertionError('Invalid state for self.samples_path and self.n_samples. If self.samples_path is a'
                                     'future COMPSs object, self.n_samples and self.n_features must be set manually.')
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
            self.y_codes, self.y_categories, self.n_classes = get_labels(self.labels_path)
        return self.y_codes

    def get_classes(self):
        if self.y_categories is None:
            self.y_codes, self.y_categories, self.n_classes = get_labels(self.labels_path)
        return self.y_categories

    def get_n_classes(self):
        if self.n_classes is None:
            self.y_codes, self.y_categories, self.n_classes = get_labels(self.labels_path)
        return self.n_classes

    def validate_features_file(self):
        features_npy_file = NpyFile(self.features_path)
        shape = features_npy_file.get_shape()
        fortran_order = features_npy_file.get_fortran_order()
        if len(shape) != 2:
            raise ValueError('Cannot read 2D array from features_file.')
        if (self.get_n_features(), self.get_n_samples()) != shape:
            raise ValueError('Invalid dimensions for the array in features_file.')
        if fortran_order:
            raise ValueError('Fortran order not supported for features array.')


def transform_to_rf_dataset(dataset: Dataset) -> RfDataset:
    s = pd.Series(["a", "b", "c", "a"], dtype=CategoricalDtype())
    for subset in dataset:
        pass
    pass
