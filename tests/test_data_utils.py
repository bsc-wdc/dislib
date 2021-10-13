import unittest

import numpy as np
from parameterized import parameterized
from dislib.data.array import random_array
from dislib.data.util import pad, pad_last_blocks_with_zeros, \
    compute_bottom_right_shape, remove_last_columns


class DataUtilsTest(unittest.TestCase):

    @parameterized.expand([
        ((3, 4), (3, 3), (3, 1)),
        ((4, 3), (3, 3), (1, 3)),
        ((4, 4), (2, 2), (2, 2)),
        ((1, 1), (1, 1), (1, 1)),
        ((3, 3), (3, 3), (3, 3)),
        ((6, 3), (3, 3), (3, 3)),
        ((3, 6), (3, 3), (3, 3)),
        ((3, 6), (3, 2), (3, 2)),
        ((3, 6), (2, 3), (1, 3)),
        ((7, 11), (2, 3), (1, 2)),
    ])
    def test_compute_bottom_right_shape(self, shape, block_size,
                                        desired_br_shape):
        np.random.seed(8)
        a = random_array(shape, block_size)

        self.assertTrue(compute_bottom_right_shape(a), desired_br_shape)

    @parameterized.expand([
        ((7, 8), (3, 3), 1),
        ((4, 4), (3, 3), 2),
        ((4, 4), (1, 1), 2),
    ])
    def test_pad(self, shape, block_size, value):
        """Tests dislib.data.util.pad"""
        np.random.seed(8)

        a = random_array(shape, block_size)
        a_original = a.copy()

        bottom = 0 if shape[0] % block_size[0] == 0 else block_size[0] - shape[
            0] % block_size[0]
        right = 0 if shape[1] % block_size[1] == 0 else block_size[1] - shape[
            1] % block_size[1]

        pad(a, ((0, bottom), (0, right)), constant_value=value)

        a_np = a.collect()
        # check if original content is the same
        self.assertTrue(
            np.allclose(a_np[0:shape[0], 0:shape[1]], a_original.collect()))
        # check if bottom rows are added correctly
        self.assertTrue(np.allclose(
            a_np[shape[0]:shape[0] + bottom, :],
            np.full((bottom, shape[1] + right), value))
        )
        # check if right columns are added correctly
        self.assertTrue(
            np.allclose(a_np[0:shape[0] + bottom, shape[1]:shape[1] + right],
                        np.full((shape[0] + bottom, right), value)))
        # check changed bottom blocks
        for i in range(len(a._blocks[-1])):
            self.assertTrue(a._blocks[-1][i].shape == a._reg_shape)
        # check changed right blocks
        for i in range(len(a._blocks)):
            self.assertTrue(a._blocks[i][-1].shape == a._reg_shape)
        # check top-left shape
        if shape[0] < block_size[0]:
            self.assertTrue(a._top_left_shape[0] == a._reg_shape[0])
        if shape[1] < block_size[1]:
            self.assertTrue(a._top_left_shape[1] == a._reg_shape[1])

    @parameterized.expand([
        ((2, 2), (1, 1)),
        ((2, 2), (2, 2)),
        ((5, 5), (4, 4)),
        ((5, 5), (3, 3)),
        ((4, 7), (3, 3)),
        ((4, 7), (2, 2)),
        ((7, 3), (2, 2)),
    ])
    def test_pad_with_zeros(self, shape, block_size):
        """Tests dislib.data.util.pad_last_blocks_with_zeros"""
        np.random.seed(8)

        a = random_array(shape, block_size)
        a_original = a.copy()

        bottom = 0 if shape[0] % block_size[0] == 0 else block_size[0] - shape[
            0] % block_size[0]
        right = 0 if shape[1] % block_size[1] == 0 else block_size[1] - shape[
            1] % block_size[1]

        pad_last_blocks_with_zeros(a)

        a_np = a.collect()
        # check if original content is the same
        self.assertTrue(
            np.allclose(a_np[0:shape[0], 0:shape[1]], a_original.collect()))
        # check if bottom rows are added correctly
        self.assertTrue(np.allclose(a_np[shape[0]:shape[0] + bottom, :],
                                    np.full((bottom, shape[1] + right), 0)))
        # check if right columns are added correctly
        self.assertTrue(
            np.allclose(a_np[0:shape[0] + bottom, shape[1]:shape[1] + right],
                        np.full((shape[0] + bottom, right), 0)))
        # check changed bottom blocks
        for i in range(len(a._blocks[-1])):
            self.assertTrue(a._blocks[-1][i].shape == a._reg_shape)
        # check changed right blocks
        for i in range(len(a._blocks)):
            self.assertTrue(a._blocks[i][-1].shape == a._reg_shape)
        # check top-left shape
        if shape[0] < block_size[0]:
            self.assertTrue(a._top_left_shape[0] == a._reg_shape[0])
        if shape[1] < block_size[1]:
            self.assertTrue(a._top_left_shape[1] == a._reg_shape[1])

    def test_exceptions(self):
        """Tests exceptions thrown by utility functions"""
        np.random.seed(8)

        a = random_array((20, 20), (6, 6))
        with self.assertRaises(NotImplementedError):
            pad(a, ((2, 0), (0, 0)))

        with self.assertRaises(NotImplementedError):
            pad(a, ((0, 0), (2, 0)))

        with self.assertRaises(NotImplementedError):
            pad(a, ((0, 8), (0, 0)))

        with self.assertRaises(NotImplementedError):
            pad(a, ((0, 0), (0, 8)))

        with self.assertRaises(ValueError):
            remove_last_columns(a, 8)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
