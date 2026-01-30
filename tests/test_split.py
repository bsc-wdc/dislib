import math
import unittest

import numpy as np

import dislib as ds
from dislib.model_selection import train_test_split
from tests import BaseTimedTestCase


class TrainTestSplitTest(BaseTimedTestCase):

    def test_train_test_split_x(self):
        x = np.random.rand(40, 40)
        x_ds = ds.array(x, (9, 9))
        train, test = train_test_split(x_ds)
        self.assertTrue(train.shape[0] == int((math.floor(9 * 0.75) *
                                               (train._n_blocks[0] - 1))
                                              + 4 * 0.75))
        self.assertTrue(test.shape[0] == int((math.floor(9 * 0.25) *
                                              (train._n_blocks[0] - 1))
                                             + 4 * 0.25))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(9 * 0.75))
        self.assertTrue(test._reg_shape[0] == int(9 * 0.25))
        self.assertTrue(train._reg_shape[1] == int(9))
        self.assertTrue(test._reg_shape[1] == int(9))
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_x_reg_shape(self):
        x = np.random.rand(40, 40)
        x_ds = ds.array(x, (8, 8))
        train, test = train_test_split(x_ds)
        self.assertTrue(train.shape[0] == int(40 * 0.75))
        self.assertTrue(test.shape[0] == int(40 * 0.25))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(8 * 0.75))
        self.assertTrue(test._reg_shape[0] == int(8 * 0.25))
        self.assertTrue(train._reg_shape[1] == int(8))
        self.assertTrue(test._reg_shape[1] == int(8))
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_xy_reg_shape(self):
        np.random.seed(0)
        x = np.random.rand(40, 40)
        y = np.random.rand(40, 1)
        x_ds = ds.array(x, (8, 8))
        y_ds = ds.array(y, (8, 1))
        train, test, y_train, y_test = train_test_split(x_ds, y_ds)
        self.assertTrue(train.shape[0] == int(40 * 0.75))
        self.assertTrue(test.shape[0] == int(40 * 0.25))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(8 * 0.75))
        self.assertTrue(test._reg_shape[0] == int(8 * 0.25))
        self.assertTrue(train._reg_shape[1] == int(8))
        self.assertTrue(test._reg_shape[1] == int(8))
        self.assertTrue(y_train.shape[0] == int(40 * 0.75))
        self.assertTrue(y_test.shape[0] == int(40 * 0.25))
        self.assertTrue(y_train.shape[1] == 1)
        self.assertTrue(y_test.shape[1] == 1)
        self.assertTrue(y_train._reg_shape[0] == int(8 * 0.75))
        self.assertTrue(y_test._reg_shape[0] == int(8 * 0.25))
        self.assertTrue(y_train._reg_shape[1] == 1)
        self.assertTrue(y_test._reg_shape[1] == 1)
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)
        y_train = y_train.collect()
        y_test = y_test.collect()
        for idx, x_row in enumerate(y):
            found = False
            for train_row in y_train:
                if (train_row == x_row).all():
                    found = True
            for test_row in y_test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_xy(self):
        np.random.seed(0)
        x = np.random.rand(40, 40)
        y = np.random.rand(40, 1)
        x_ds = ds.array(x, (9, 9))
        y_ds = ds.array(y, (9, 1))
        train, test, y_train, y_test = train_test_split(x_ds, y_ds)
        self.assertTrue(train.shape[0] == int((math.floor(9 * 0.75) *
                                               (train._n_blocks[0] - 1))
                                              + 4 * 0.75))
        self.assertTrue(test.shape[0] == int((math.ceil(9 * 0.25) *
                                              (train._n_blocks[0] - 1))
                                             + 4 * 0.25))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(9 * 0.75))
        self.assertTrue(test._reg_shape[0] == math.ceil(9 * 0.25))
        self.assertTrue(train._reg_shape[1] == int(9))
        self.assertTrue(test._reg_shape[1] == int(9))
        self.assertTrue(y_train.shape[0] == int((math.floor(9 * 0.75) *
                                                 (train._n_blocks[0] - 1))
                                                + 4 * 0.75))
        self.assertTrue(y_test.shape[0] == int((math.ceil(9 * 0.25) *
                                                (train._n_blocks[0] - 1))
                                               + 4 * 0.25))
        self.assertTrue(y_train.shape[1] == 1)
        self.assertTrue(y_test.shape[1] == 1)
        self.assertTrue(y_train._reg_shape[0] == int(9 * 0.75))
        self.assertTrue(y_test._reg_shape[0] == math.ceil(9 * 0.25))
        self.assertTrue(y_train._reg_shape[1] == 1)
        self.assertTrue(y_test._reg_shape[1] == 1)
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)
        y_train = y_train.collect()
        y_test = y_test.collect()
        for idx, x_row in enumerate(y):
            found = False
            for train_row in y_train:
                if (train_row == x_row).all():
                    found = True
            for test_row in y_test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_x_train_size(self):
        x = np.random.rand(40, 40)
        x_ds = ds.array(x, (10, 10))
        train, test = train_test_split(x_ds, train_size=0.60)
        self.assertTrue(train.shape[0] == int(40 * 0.60))
        self.assertTrue(test.shape[0] == int(40 * 0.40))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(10 * 0.60))
        self.assertTrue(test._reg_shape[0] == int(10 * 0.40))
        self.assertTrue(train._reg_shape[1] == int(10))
        self.assertTrue(test._reg_shape[1] == int(10))
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_x_test_size(self):
        x = np.random.rand(40, 40)
        x_ds = ds.array(x, (8, 8))
        train, test = train_test_split(x_ds, test_size=0.5)
        self.assertTrue(train.shape[0] == int(40 * 0.50))
        self.assertTrue(test.shape[0] == int(40 * 0.50))
        self.assertTrue(train.shape[1] == int(40))
        self.assertTrue(test.shape[1] == int(40))
        self.assertTrue(train._reg_shape[0] == int(8 * 0.50))
        self.assertTrue(test._reg_shape[0] == int(8 * 0.50))
        self.assertTrue(train._reg_shape[1] == int(8))
        self.assertTrue(test._reg_shape[1] == int(8))
        train = train.collect()
        test = test.collect()
        for idx, x_row in enumerate(x):
            found = False
            for train_row in train:
                if (train_row == x_row).all():
                    found = True
            for test_row in test:
                if (test_row == x_row).all():
                    found = True
            self.assertTrue(found)

    def test_train_test_split_excep(self):
        x = ds.random_array((3000, 3000), (400, 400), random_state=88)
        y = ds.random_array((3000, 1), (400, 1), random_state=88)
        with self.assertRaises(ValueError):
            train_test_split(x, y, test_size=1.25, train_size=1)
        with self.assertRaises(ValueError):
            train_test_split(x, y, test_size=0.95, train_size=0.1)
        with self.assertRaises(ValueError):
            train_test_split(x, y=np.zeros((2, 2)), test_size=0.95,
                             train_size=0.1)
        with self.assertRaises(ValueError):
            train_test_split(x=np.zeros((2, 2)), test_size=0.95,
                             train_size=0.1)
        with self.assertRaises(ValueError):
            train_test_split(x, y=np.zeros((2, 2)), test_size=0.95)
        with self.assertRaises(ValueError):
            train_test_split(x=np.zeros((2, 2)), test_size=0.95)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
