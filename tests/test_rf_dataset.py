import unittest

import os
import shutil
from sklearn.datasets import make_classification
import dislib as ds
from dislib.trees import data, test_split
from dislib.data.array import Array
import numpy as np
from sys import float_info
from pycompss.api.api import compss_wait_on

DIRPATH = "tests/files/saving"


class RFDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs(DIRPATH, exist_ok=True)
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(DIRPATH)
        return super().tearDown()

    def test_rf_dataset(self):
        # Save samples and features
        x, y = make_classification(
            n_samples=900,
            n_features=10,
            n_classes=3,
            n_informative=4,
            random_state=0,
        )
        x_ds_1 = ds.array(x, (300, 10))
        x_ds_2 = ds.array(x[:600], (300, 10))
        y_ds_1 = ds.array(y[:, np.newaxis], (300, 1))
        y_ds_2 = ds.array(y[:600][:, np.newaxis], (300, 1))
        samples_path_1 = os.path.join(DIRPATH, "feats_1")
        samples_path_2 = os.path.join(DIRPATH, "feats_2")
        targets_path_1 = os.path.join(DIRPATH, "targets_1")
        targets_path_2 = os.path.join(DIRPATH, "targets_2")
        features_path_f = os.path.join(DIRPATH, "targets_f")
        save_samples(x_ds_1, samples_path_1, False)
        save_samples(x_ds_2, samples_path_2, False)
        save_targets(y_ds_1, targets_path_1)
        save_targets(y_ds_2, targets_path_2)
        save_features(x_ds_2, features_path_f, True)

        # Regression and classification datatser
        rf_regr = data.RfRegressorDataset(samples_path_1, targets_path_1)
        rf_class = data.RfClassifierDataset(samples_path_1, targets_path_1)

        # Test get number of samples and features
        self.assertEqual(rf_regr.get_n_samples(), 900)
        self.assertEqual(rf_class.get_n_samples(), 900)
        self.assertEqual(rf_regr.get_n_features(), 10)
        self.assertEqual(rf_class.get_n_features(), 10)

        # Test get y targets
        y_regr = compss_wait_on(rf_regr.get_y_targets())
        y_class = compss_wait_on(rf_class.get_y_targets())
        self.assertTrue(np.all(y_regr == y_ds_1.collect()))
        self.assertTrue(np.all(y_class == y_ds_1.collect()))

        # Test get number of classes and classes
        n_class = compss_wait_on(rf_regr.get_n_classes())
        classes = compss_wait_on(rf_regr.get_classes())
        self.assertTrue(n_class is None)
        self.assertTrue(classes is None)

        rf_class.n_classes = None
        n_class = compss_wait_on(rf_class.get_n_classes())
        rf_class.y_categories = None
        classes = compss_wait_on(rf_class.get_classes())
        self.assertEqual(n_class, 3)
        self.assertTrue(np.all(classes == [0, 1, 2]))

        # Sample and feature paths must be str
        rf_dataset = data.RfBaseDataset(None, None)
        with self.assertRaises(TypeError):
            rf_dataset.get_n_samples()
        with self.assertRaises(TypeError):
            rf_dataset.get_n_features()

        # Validate dimension
        rf_dataset = data.RfBaseDataset(
            samples_path_1, targets_path_1, features_path_f
        )
        rf_dataset.samples_path = samples_path_2
        with self.assertRaises(ValueError):
            rf_dataset.validate_features_file()

        # Validate Fortran order
        rf_dataset = data.RfBaseDataset(
            samples_path_1, targets_path_1, features_path_f
        )
        with self.assertRaises(ValueError):
            rf_dataset.validate_features_file()

        # Dataset creation
        rf_regr = data.transform_to_rf_dataset(
            x_ds_1, y_ds_1, data.RfRegressorDataset, features_file=True
        )
        rf_class = data.transform_to_rf_dataset(
            x_ds_1, y_ds_1, data.RfClassifierDataset, features_file=True
        )
        self.assertEquals(compss_wait_on(rf_regr.get_n_samples()), 900)
        self.assertEquals(compss_wait_on(rf_regr.get_n_features()), 10)
        self.assertEquals(compss_wait_on(rf_class.get_n_samples()), 900)
        self.assertEquals(compss_wait_on(rf_class.get_n_features()), 10)

        # Npy files
        file = data._NpyFile(features_path_f)
        file.shape = None
        self.assertEqual(file.get_shape(), (10, 600))
        file.fortran_order = None
        self.assertTrue(file.get_fortran_order())
        file.dtype = None
        self.assertEqual(file.get_dtype().name, "float32")

        file = data._NpyFile(samples_path_2)
        file.shape = None
        self.assertEqual(file.get_shape(), (600, 10))
        file.fortran_order = None
        self.assertFalse(file.get_fortran_order())
        file.dtype = None
        self.assertEqual(file.get_dtype().name, "float32")

        # Test returns for empty size
        score, value = test_split.test_split(None, np.array([]), None, None)
        self.assertEqual(score, float_info.max)
        self.assertEqual(value, np.float64(np.inf))


def _fill_samples_file(samples_path, row_blocks, start_idx, fortran_order):
    rows_samples = Array._merge_blocks(row_blocks)
    rows_samples = rows_samples.astype(dtype="float32", casting="same_kind")
    samples = np.lib.format.open_memmap(
        samples_path, mode="r+", fortran_order=fortran_order
    )
    samples[start_idx: start_idx + rows_samples.shape[0]] = rows_samples


def _fill_features_file(samples_path, row_blocks, start_idx, fortran_order):
    rows_samples = Array._merge_blocks(row_blocks)
    rows_samples = rows_samples.astype(dtype="float32", casting="same_kind")
    samples = np.lib.format.open_memmap(
        samples_path, mode="r+", fortran_order=fortran_order
    )
    samples[:, start_idx: start_idx + rows_samples.shape[0]] = rows_samples.T


def _fill_targets_file(targets_path, row_blocks):
    rows_targets = Array._merge_blocks(row_blocks)
    with open(targets_path, "at") as f:
        np.savetxt(f, rows_targets, fmt="%s", encoding="utf-8")


def save_samples(x, samples_path, fortran_order):
    n_samples = x.shape[0]
    n_features = x.shape[1]

    open(samples_path, "w").close()
    np.lib.format.open_memmap(
        samples_path,
        mode="w+",
        dtype="float32",
        fortran_order=fortran_order,
        shape=(int(n_samples), int(n_features)),
    )
    start_idx = 0
    row_blocks_iterator = x._iterator(axis=0)
    top_row = next(row_blocks_iterator)
    _fill_samples_file(samples_path, top_row._blocks, start_idx, fortran_order)
    start_idx += x._top_left_shape[0]
    for x_row in row_blocks_iterator:
        _fill_samples_file(
            samples_path, x_row._blocks, start_idx, fortran_order
        )
        start_idx += x._reg_shape[0]


def save_targets(y, targets_path):
    open(targets_path, "w").close()
    for y_row in y._iterator(axis=0):
        _fill_targets_file(targets_path, y_row._blocks)


def save_features(x, features_path, fortran_order):
    n_samples = x.shape[0]
    n_features = x.shape[1]

    np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype="float32",
        fortran_order=fortran_order,
        shape=(int(n_features), int(n_samples)),
    )
    start_idx = 0
    row_blocks_iterator = x._iterator(axis=0)
    top_row = next(row_blocks_iterator)
    _fill_features_file(
        features_path, top_row._blocks, start_idx, fortran_order
    )
    start_idx += x._top_left_shape[0]
    for x_row in row_blocks_iterator:
        _fill_features_file(
            features_path, x_row._blocks, start_idx, fortran_order
        )
        start_idx += x._reg_shape[0]


def main():
    unittest.main()


if __name__ == "__main__":
    main()
