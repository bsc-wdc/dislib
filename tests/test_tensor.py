import unittest
from dislib.data.tensor import Tensor
from pycompss.api.api import compss_wait_on, compss_barrier
import numpy as np
import dislib as ds
import torch
import tensorflow as tf


def _validate_tensor(x):
    x_aux = x.collect()
    tl = x.tensors[0][0].shape
    br = x.tensors[-1][-1].shape

    # single element arrays might contain only the value and not a NumPy
    # array (and thus there is no shape)
    if not tl:
        tl = (1, 1)
    if not br:
        br = (1, 1)

    br0 = x.shape[1] * x.tensor_shape[0] - \
        ((x.shape[1] - 1) * x.tensor_shape[0])
    return x_aux, (tl == x.tensor_shape and br0 == br[0] and x.n_tensors /
                   x.shape[0] == len(x.tensors[0]) and x.shape[0] == 1)


def _validate_tensor_col(x):
    x_aux = x.collect()
    tl = x.tensors[0][0].shape
    br = x.tensors[-1][-1].shape

    # single element arrays might contain only the value and not a NumPy
    # array (and thus there is no shape)
    if not tl:
        tl = (1, 1)
    if not br:
        br = (1, 1)

    br0 = x.shape[1] * x.tensor_shape[0] - \
        ((x.shape[1] - 1) * x.tensor_shape[0])
    return x_aux, (tl == x.tensor_shape and br0 == br[0] and x.n_tensors /
                   x.shape[1] == len(x.tensors) and x.shape[1] == 1)


def _equal_valued_tensors(x1, x2):
    if torch.is_tensor(x1[0][0]):
        tensors = []
        for tensor in x1:
            tensors_aux = []
            for tens in tensor:
                tensors_aux.append(tens.numpy())
            tensors.append(tensors_aux)
    else:
        tensors = x1
    if len(tensors[0]) > 1:
        tensors = np.concatenate(*tensors)
    else:
        tensors = np.concatenate(tensors)

    if torch.is_tensor(x2[0][0]):
        tensors_2 = []
        for tensor in x2:
            tensors_2_aux = []
            for tens in tensor:
                tensors_2_aux.append(tens.numpy())
            tensors_2.append(tensors_2_aux)
    else:
        tensors_2 = x2
    if len(tensors_2[0]) > 1:
        tensors_2 = np.concatenate(*tensors_2)
    else:
        tensors_2 = np.concatenate(tensors_2)
    return np.allclose(tensors, tensors_2)


class TensorTest(unittest.TestCase):

    def test_sizes_and_type(self):
        """ Tests sizes consistency. """
        x_np = ds.random_tensors("np", (3, 3, 3, 11, 11))
        x_torch = ds.random_tensors("torch", (3, 3, 3, 11, 11))

        self.assertEqual(x_np.shape, (3, 3))
        self.assertEqual(x_torch.shape, (3, 3))
        self.assertEqual(x_np.tensor_shape, (3, 11, 11))
        self.assertEqual(x_torch.tensor_shape, (3, 11, 11))
        self.assertEqual(x_np.n_tensors, 9)
        self.assertEqual(x_torch.n_tensors, 9)
        self.assertEqual(x_np.dtype, np.float64)
        self.assertEqual(x_torch.dtype, torch.float64)

    def test_string(self):
        x_np = ds.random_tensors("np", (3, 3, 3, 11, 11))
        self.assertEqual(str(x_np), "ds-tensor(tensors=(...), "
                                    "tensors_shape=%r, "
                                    "n_tensors=%r, "
                                    "num_samples=%r, "
                                    "shape=%r)" % ((3, 11, 11),
                                                   9,
                                                   99,
                                                   (3, 3)))

    def test_iterate_rows(self):
        """ Testing the row _iterator of the ds.tensor """

        np_rand = np.random.rand(3, 3, 3, 11, 11)
        x_np = ds.from_array(np_rand)
        x_torch = ds.from_array(np_rand)

        for i, h_block in enumerate(x_np._iterator(axis='rows')):
            computed = h_block
            expected = x_torch[i: (i + 1)]
            computed, validated_tensor = _validate_tensor(computed)
            self.assertTrue(validated_tensor)
            self.assertTrue(_equal_valued_tensors(computed,
                                                  expected.collect()))

        torch_rand = torch.rand(3, 3, 3, 11, 11)
        x_np = ds.from_pt_tensor(torch_rand)
        x_torch = ds.from_pt_tensor(torch_rand)

        for i, h_block in enumerate(x_np._iterator(axis='rows')):
            computed = h_block
            expected = x_torch[i: (i + 1)]
            computed, validated_tensor = _validate_tensor(computed)
            self.assertTrue(validated_tensor)
            self.assertTrue(_equal_valued_tensors(computed,
                                                  expected.collect()))

    def test_iterate_cols(self):
        """ Testing the col _iterator of the ds.tensor """
        np_rand = np.random.rand(3, 3, 3, 11, 11)
        x_np = ds.from_array(np_rand)
        x_torch = ds.from_array(np_rand)

        for i, v_block in enumerate(x_np._iterator(axis='columns')):
            expected = x_torch[:, i: (i + 1)]
            v_block, validated_tensor = _validate_tensor_col(v_block)
            self.assertTrue(validated_tensor)
            self.assertTrue(_equal_valued_tensors(v_block,
                                                  expected.collect()))

        torch_rand = torch.rand(3, 3, 3, 11, 11)
        x_np = ds.from_pt_tensor(torch_rand)
        x_torch = ds.from_pt_tensor(torch_rand)

        for i, v_block in enumerate(x_np._iterator(axis='columns')):
            expected = x_torch[:, i: (i + 1)]
            v_block, validated_tensor = _validate_tensor_col(v_block)
            self.assertTrue(validated_tensor)
            self.assertTrue(_equal_valued_tensors(v_block,
                                                  expected.collect()))

    def test_iterate_exception(self):
        compss_barrier()
        x_np = ds.random_tensors("np", (3, 3, 3, 11, 11))
        with self.assertRaises(Exception):
            for i, v_block in enumerate(x_np._iterator(axis='diagonal')):
                print(i)

    def test_invalid_indexing(self):
        """ Tests invalid indexing """
        compss_barrier()
        x = ds.random_tensors("np", (3, 3, 3, 11, 11))
        with self.assertRaises(IndexError):
            x[8]
        with self.assertRaises(IndexError):
            x["8"]
        with self.assertRaises(IndexError):
            x[[3], [4]]
        with self.assertRaises(IndexError):
            x[4, 4]
        with self.assertRaises(IndexError):
            x[4, [4, 5]]
        with self.assertRaises(IndexError):
            x[[4, 5]]
        with self.assertRaises(IndexError):
            x[0, 5]
        with self.assertRaises(IndexError):
            x[[4, 5], [4, 5]]
        with self.assertRaises(IndexError):
            x[[0, 1], [4, 5]]
        with self.assertRaises(NotImplementedError):
            x[0:1, -3:-1]
        with self.assertRaises(NotImplementedError):
            x[0:1:2]
        with self.assertRaises(NotImplementedError):
            x[0:1, 0:1:2]
        with self.assertRaises(NotImplementedError):
            x[-2:-1]
        with self.assertRaises(IndexError):
            x[4, 0, 0, 2, 3]
        with self.assertRaises(IndexError):
            x[0, 4, 0, 2, 3]
        with self.assertRaises(IndexError):
            x[[0, 4], 0, 2, 3]
        with self.assertRaises(IndexError):
            x[[0, 1], [4, 5], 0, 2, 3]
        with self.assertRaises(IndexError):
            x[0, 4, 0, 2, 3, 0, 0]
        with self.assertRaises(IndexError):
            x[0, 0, 0, 2, 3, 0, 0, 0]

    def test_indexing(self):
        compss_barrier()
        x = ds.random_tensors("np", (3, 3, 5, 11, 11))
        aux = x[0]
        self.assertEqual(aux.shape, (1, x.shape[1]))
        self.assertEqual(aux.tensor_shape, x.tensor_shape)
        aux = x[[0, 1]]
        self.assertEqual(aux.shape, (2, x.shape[1]))
        self.assertEqual(aux.tensor_shape, x.tensor_shape)
        aux = x[0, [1, 2]]
        self.assertEqual(aux.shape, (1, 2))
        self.assertEqual(aux.tensor_shape, x.tensor_shape)
        aux = x[[0, 1], 1]
        self.assertEqual(aux.shape, (2, 1))
        self.assertEqual(aux.tensor_shape, x.tensor_shape)
        x = ds.random_tensors("np", (3, 3, 5, 11, 11))
        aux = x[0, [0, 1], 1:3, 2:4, 2:4]
        self.assertEqual(aux.shape, (1, 2))
        self.assertEqual(aux.tensor_shape, (2, 2, 2))
        x = ds.random_tensors("np", (3, 3, 5, 11, 11))
        aux = x[0:1, 0:2, 1:3, 2:4, 2:4]
        self.assertEqual(aux.shape, (1, 2))
        self.assertEqual(aux.tensor_shape, (2, 2, 2))
        x = ds.random_tensors("np", (3, 3, 5, 11, 11))
        aux = x[0:1, 0:2, [0, 2], [4, 6], [3, 7]]
        self.assertEqual(aux.shape, (1, 2))
        self.assertEqual(aux.tensor_shape, (2,))
        aux = x[0:1, 0:2, 0, [4, 6]]
        self.assertEqual(aux.shape, (1, 2))
        self.assertEqual(aux.tensor_shape, (1, 2))
        aux = x[1, 2, [1, 2], 0:11, 0:11]
        self.assertEqual(aux.shape, (1, 1))
        self.assertEqual(aux.tensor_shape, (2, 11, 11))
        aux = x[1, 2, 1, 0:11, 0:11]
        self.assertEqual(aux.shape, (1, 1))
        self.assertEqual(aux.tensor_shape, (1, 11, 11))
        aux = x[1, 2]
        self.assertEqual(aux.shape, (1, 1))
        self.assertEqual(aux.tensor_shape, (5, 11, 11))
        aux = x[0:0]
        self.assertEqual(aux.shape, (1, 1))
        self.assertEqual(aux.tensor_shape, (0, 0))
        aux = x[0:1, :4]
        self.assertEqual(aux.shape, (1, 3))
        self.assertEqual(aux.tensor_shape, (5, 11, 11))
        aux = x[0:1, 0:0]
        self.assertEqual(aux.shape, (1, 1))
        self.assertEqual(aux.tensor_shape, (0, 0))

    def test_invalid_assignation(self):
        compss_barrier()
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4, 4))
        with self.assertRaises(ValueError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[0, 0] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[3, 0] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[[0, 1, 3], [0]] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[3, 1, 3] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[0, 1, 5, 5] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[3, 1, [5, 5]] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[3, [0, 1], [5, 5]] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[0, [0, 3], [5, 5]] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[[0, 1], 3, [5, 5]] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[[0, 3], 0, [5, 5]] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[[0, 1], [0, 3], [5, 5]] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(IndexError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[[0, 3], [0, 1], [5, 5]] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(NotImplementedError):
            tensor_to_assign = np.array([[2, 3, 4, 5], [2, 3, 4, 5],
                                         [2, 3, 4, 5], [2, 3, 4, 5]])
            x_train_tensor[0] = tensor_to_assign
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4))
        with self.assertRaises(ValueError):
            x_train_tensor[0, 0] = tf.constant(np.array([[2, 3, 4, 5],
                                                         [2, 3, 4, 5],
                                                         [2, 3, 4, 5],
                                                         [2, 3, 4, 5]]))

    def test_assignation(self):
        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4, 4))
        tensor_to_assign = np.ones((4, 4, 4))
        x_train_tensor[0, 0] = tensor_to_assign
        self.assertTrue(x_train_tensor.tensor_shape == (4, 4, 4))
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[0][0]).shape == (4, 4, 4))
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[0][0]).all() == np.ones((4, 4, 4)).all())

        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4, 4))
        tensor_to_assign = np.ones((4, 4, 4))
        x_train_tensor[[0, 1], [0, 1]] = tensor_to_assign
        self.assertTrue(x_train_tensor.tensor_shape == (4, 4, 4))
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[0][0]).shape == (4, 4, 4))
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[1][0]).all() == np.ones((4, 4, 4)).all())
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[0][0]).all() == np.ones((4, 4, 4)).all())
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[0][1]).all() == np.ones((4, 4, 4)).all())
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[1][1]).all() == np.ones((4, 4, 4)).all())

        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4, 4))
        x_train_tensor[0, 0, 0, 0, 1] = 8
        self.assertTrue(x_train_tensor.tensor_shape == (4, 4, 4))
        x_train_tensor.tensors[0][0] = compss_wait_on(
            x_train_tensor.tensors[0][0])
        self.assertTrue(x_train_tensor.tensors[0][0].shape == (4, 4, 4))
        self.assertTrue(x_train_tensor.tensors[0][0][0, 0, 1] == 8)

        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4, 4))
        tensor_to_assign = np.ones((4, 4))
        x_train_tensor[0, 0, [0, 1]] = tensor_to_assign
        self.assertTrue(x_train_tensor.tensor_shape == (4, 4, 4))
        x_train_tensor.tensors[0][0] = compss_wait_on(
            x_train_tensor.tensors[0][0])
        self.assertTrue(x_train_tensor.tensors[0][0].shape == (4, 4, 4))
        self.assertTrue(x_train_tensor.tensors[0][0].all() ==
                        np.ones((4, 4)).all())

        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4, 4))
        tensor_to_assign = np.ones((4, 4))
        x_train_tensor[[0, 1], 0, [0, 1]] = tensor_to_assign
        self.assertTrue(x_train_tensor.tensor_shape == (4, 4, 4))
        x_train_tensor.tensors[0][0] = \
            compss_wait_on(x_train_tensor.tensors[0][0])
        self.assertTrue(x_train_tensor.tensors[0][0].shape == (4, 4, 4))
        self.assertTrue(x_train_tensor.tensors[0][0].all() ==
                        np.ones((4, 4)).all())
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[1][0]).all() == np.ones((4, 4)).all())

        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4, 4))
        tensor_to_assign = np.ones((4, 4))
        x_train_tensor[0, [0, 1], [0, 1]] = tensor_to_assign
        self.assertTrue(x_train_tensor.tensor_shape == (4, 4, 4))
        x_train_tensor.tensors[0][0] = \
            compss_wait_on(x_train_tensor.tensors[0][0])
        self.assertTrue(x_train_tensor.tensors[0][0].shape == (4, 4, 4))
        self.assertTrue(x_train_tensor.tensors[0][0].all() ==
                        np.ones((4, 4)).all())
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[0][1]).all() == np.ones((4, 4)).all())

        x_train_tensor = ds.random_tensors("np", (2, 2, 4, 4, 4))
        tensor_to_assign = np.ones((4, 4))
        x_train_tensor[[0, 1], [0, 1], [0, 1]] = tensor_to_assign
        self.assertTrue(x_train_tensor.tensor_shape == (4, 4, 4))
        x_train_tensor.tensors[0][0] = \
            compss_wait_on(x_train_tensor.tensors[0][0])
        self.assertTrue(x_train_tensor.tensors[0][0].shape == (4, 4, 4))
        self.assertTrue(x_train_tensor.tensors[0][0].all() ==
                        np.ones((4, 4)).all())
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[0][1]).all() == np.ones((4, 4)).all())
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[1][0]).all() == np.ones((4, 4)).all())
        self.assertTrue(compss_wait_on(
            x_train_tensor.tensors[1][1]).all() == np.ones((4, 4)).all())

    def test_basic_operations(self):
        x_np = ds.from_array(np.array([[[1, 2, 3], [3, 2, 1]]]))
        x_np_pow = x_np**2
        x_np_pow = np.array(x_np_pow.collect())
        self.assertEqual(x_np_pow.all(),
                         np.array([[np.array([1, 4, 9]),
                                    np.array([9, 4, 1])]]).all())
        with self.assertRaises(NotImplementedError):
            x_np**(2.21, 2)
        x_np = ds.from_array(np.array([[[1, 2, 3], [3, 2, 1]]]))
        x_np_mul = x_np * 2
        x_np_mul = np.array(x_np_mul.collect())
        self.assertEqual(x_np_mul.all(),
                         np.array([[np.array([1, 4, 6]),
                                    np.array([6, 4, 1])]]).all())
        with self.assertRaises(NotImplementedError):
            x_np*(2.21, 2)
        x_np_2 = ds.from_array(np.array([[[2, 4, 8], [8, 16, 32]]]))
        x_np_div = x_np_2 / 2
        x_np_div = np.array(x_np_div.collect())
        self.assertEqual(x_np_div.all(),
                         np.array([[np.array([1, 2, 4]),
                                    np.array([4, 8, 16])]]).all())
        with self.assertRaises(NotImplementedError):
            x_np/(2.21, 2)
        x_np = ds.from_array(np.array([[[1, 2, 3], [3, 2, 1]]]))
        x_np_2 = ds.from_array(np.array([[[2, 4, 8], [8, 16, 32]]]))
        x_add = x_np + x_np_2
        x_add = np.array(x_add.collect())
        self.assertEqual(x_add.all(),
                         np.array([[np.array([3, 6, 11]),
                                    np.array([11, 18, 33])]]).all())
        x_np = ds.from_array(np.array([[[1, 2, 3], [3, 2, 1]]]))
        x_np_2 = ds.from_array(np.array([[[2, 4, 8], [8, 16, 32]]]))
        x_sub = x_np - x_np_2
        x_sub = np.array(x_sub.collect())
        self.assertEqual(x_sub.all(),
                         np.array([[np.array([-1, -2, -5]),
                                    np.array([-5, -14, -31])]]).all())
        with self.assertRaises(ValueError):
            x_np_2 = ds.from_array(
                np.array([[[2, 4, 8], [8, 16, 32], [1, 2, 3]]]))
            x_np + x_np_2
        with self.assertRaises(ValueError):
            x_np_2 = ds.from_array(
                np.array([[[2, 4, 8], [8, 16, 32, 4]]]))
            x_np + x_np_2
        with self.assertRaises(ValueError):
            x_np_2 = ds.from_array(
                np.array([[[2, 4, 8], [8, 16, 32], [1, 2, 3]]]))
            x_np - x_np_2
        with self.assertRaises(ValueError):
            x_np_2 = ds.from_array(
                np.array([[[2, 4, 8], [8, 16, 32, 4]]]))
            x_np - x_np_2

        x_np = ds.from_pt_tensor(torch.tensor([[[1, 2, 3], [3, 2, 1]]]))
        x_np_2 = ds.from_pt_tensor(torch.tensor([[[2, 4, 8], [8, 16, 32]]]))
        x_np_pow = x_np ** 2
        x_np_pow = x_np_pow.collect()
        self.assertEqual(x_np_pow[0][0].all(), torch.tensor([1, 4, 9]).all())
        self.assertEqual(x_np_pow[0][1].all(), torch.tensor([9, 4, 1]).all())
        x_np = ds.from_pt_tensor(torch.tensor([[[1, 2, 3], [3, 2, 1]]]))
        x_add = x_np + x_np_2
        x_add = x_add.collect()
        self.assertEqual(x_add[0][0].all(), torch.tensor([3, 6, 11]).all())
        self.assertEqual(x_add[0][1].all(), torch.tensor([11, 18, 33]).all())
        x_np = ds.from_pt_tensor(torch.tensor([[[1, 2, 3], [3, 2, 1]]]))
        x_np_2 = ds.from_pt_tensor(torch.tensor([[[2, 4, 8], [8, 16, 32]]]))
        x_sub = x_np - x_np_2
        x_sub = x_sub.collect()
        self.assertEqual(x_sub[0][0].all(), torch.tensor([-1, -2, -5]).all())
        self.assertEqual(x_sub[0][1].all(), torch.tensor([-5, -14, -31]).all())

        x_np_2 = Tensor(tensors=[["String1", "String2"]],
                        tensor_shape=(2, 2),
                        number_samples=4,
                        dtype=np.float64)

        with self.assertRaises(ValueError):
            ds.data.tensor.Tensor._merge_tensors(x_np_2)

    def test_tensor_creation(self):
        np_array = np.random.rand(2, 2, 3, 3, 3)
        x_np = ds.from_array(np_array)
        self.assertEqual(x_np.shape, (2, 2))
        self.assertEqual(x_np.tensor_shape, (3, 3, 3))
        np_array = np.random.rand(12, 3, 3)
        with self.assertRaises(ValueError):
            ds.from_array(np_array, shape=(2, 0))
        with self.assertRaises(ValueError):
            ds.from_array(np_array, shape=(6, 6))
        with self.assertRaises(ValueError):
            pt_tensor = torch.rand(2, 2, 3, 3, 3)
            ds.from_array(pt_tensor)
        x_np = ds.from_array(np_array, shape=(2, 2))
        self.assertEqual(x_np.shape, (2, 2))
        self.assertEqual(x_np.tensor_shape, (3, 3, 3))
        pt_tensor = torch.rand(2, 2, 3, 3, 3)
        x_np = ds.from_pt_tensor(pt_tensor)
        self.assertEqual(x_np.shape, (2, 2))
        self.assertEqual(x_np.tensor_shape, (3, 3, 3))
        pt_tensor = torch.rand(12, 3, 3)
        with self.assertRaises(ValueError):
            ds.from_pt_tensor(pt_tensor, shape=(2, 0))
        with self.assertRaises(ValueError):
            ds.from_pt_tensor(pt_tensor, shape=(6, 6))
        with self.assertRaises(ValueError):
            ds.from_pt_tensor(np_array)
        x_np = ds.from_pt_tensor(pt_tensor, shape=(2, 2))
        self.assertEqual(x_np.shape, (2, 2))
        self.assertEqual(x_np.tensor_shape, (3, 3, 3))

        with self.assertRaises(ValueError):
            ds.data.tensor.create_ds_tensor([[np.array([2, 2]),
                                              np.array([2, 2])],
                                             [np.array([2, 2]),
                                              np.array([2, 2])]],
                                            tensors_shape=(1, 2),
                                            shape=(3, 3))
        x_np = ds.data.tensor.create_ds_tensor([[np.array([2, 2]),
                                                 np.array([2, 2])],
                                                [np.array([2, 2]),
                                                 np.array([2, 2])]],
                                               tensors_shape=(1, 2),
                                               shape=(2, 2))
        self.assertEqual(x_np.shape, (2, 2))
        self.assertEqual(x_np.tensor_shape, (1, 2))

    def test_tensor_random_creation_exception(self):
        with self.assertRaises(NotImplementedError):
            ds.random_tensors("eddl", (3, 3, 3, 11, 11))

    def test_tensor_concatenation(self):
        np_array = np.random.rand(2, 2, 3, 3, 3)
        x_np = ds.from_array(np_array)
        with self.assertRaises(ValueError):
            ds.data.tensor.cat([x_np], dimension=2)
        np_array_2 = np.random.rand(2, 3, 3, 3, 3)
        x_np_2 = ds.from_array(np_array_2)
        with self.assertRaises(ValueError):
            ds.data.tensor.cat([x_np, x_np_2], dimension=0)
        np_array_3 = np.random.rand(2, 3, 3, 6, 3)
        x_np_3 = ds.from_array(np_array_3)
        with self.assertRaises(ValueError):
            ds.data.tensor.cat([x_np_2, x_np_3], dimension=0)
        new_tensor = ds.data.tensor.cat([x_np_2, x_np_3], dimension=1)
        self.assertEqual(new_tensor.tensor_shape, (3, 9, 3))
        self.assertEqual(new_tensor.shape, (2, 3))

        torch_tensor = torch.rand(2, 2, 3, 3, 3)
        torch_2 = torch.rand(2, 2, 3, 3, 3)
        tensor_ds = ds.from_pt_tensor(torch_tensor)
        tensor_ds_2 = ds.from_pt_tensor(torch_2)
        new_tensor = ds.data.tensor.cat([tensor_ds, tensor_ds_2], dimension=1)
        self.assertEqual(new_tensor.tensor_shape, (3, 6, 3))
        self.assertEqual(new_tensor.shape, (2, 2))

    def test_change_shape(self):
        compss_barrier()
        np_array = np.random.rand(2, 2, 3, 3, 3)
        x_np = ds.from_array(np_array)
        with self.assertRaises(ValueError):
            ds.data.tensor.change_shape(x_np, (3, 3))
        tensor = ds.data.tensor.change_shape(x_np, (4, 1))
        self.assertEqual(tensor.tensor_shape, (3, 3, 3))
        self.assertEqual(tensor.shape, (4, 1))

    def test_rechunk_tensor(self):
        np_array = np.random.rand(2, 2, 9, 3, 3)
        x_np = ds.from_array(np_array)
        tensor = ds.data.tensor.rechunk_tensor(x_np, 6, dimension=0)
        self.assertEqual(tensor.tensor_shape, (6, 3, 3))

        np_array = np.random.rand(2, 2, 38, 3, 3)
        x_np = ds.from_array(np_array)
        tensor = ds.data.tensor.rechunk_tensor(x_np, 8, dimension=0)
        self.assertEqual(tensor.tensor_shape, (8, 3, 3))

    def test_create_empty_tensor(self):
        compss_barrier()
        empty_tensor = ds.data.tensor._empty_tensor((2, 2), (2, 3, 4))
        self.assertEqual(empty_tensor.shape, (2, 2))
        self.assertEqual((len(empty_tensor.tensors),
                          len(empty_tensor.tensors[0])), (2, 2))
        self.assertEqual(empty_tensor.tensor_shape, (2, 3, 4))

    def test_from_ds_array(self):
        ds_array = ds.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]], (2, 2))
        ds_tensor = ds.from_ds_array(ds_array, shape=(2, 2))
        np_array = np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]])
        for i_block, i_tensor in zip(ds_array._blocks,
                                     ds_tensor.tensors):
            for j_block, j_tensor in zip(i_block,
                                         i_tensor):
                self.assertTrue(np.all(compss_wait_on(j_block) ==
                                       np.array(compss_wait_on(j_tensor))))
        with self.assertRaises(TypeError):
            ds.from_ds_array(np_array, (2, 2))
        ds_array = ds.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]], (2, 2))
        with self.assertRaises(ValueError):
            ds.from_ds_array(ds_array, (6,))
        with self.assertRaises(ValueError):
            ds.from_ds_array(ds_array, (6, 2))

    def test_load_dataset(self):
        path = "tests/datasets/npy"
        x_train_tensor = ds.data.tensor.load_dataset(2, path)
        self.assertEqual(x_train_tensor.shape, (2, 2))
        x_train_tensor = x_train_tensor.collect()
        self.assertTrue(isinstance(x_train_tensor[0][0],
                                   (np.ndarray, np.generic)))
        path = "tests/datasets/pt"
        x_train_tensor = ds.data.tensor.load_dataset(2, path)
        self.assertEqual(x_train_tensor.shape, (2, 2))
        x_train_tensor = x_train_tensor.collect()
        self.assertTrue(torch.is_tensor(x_train_tensor[0][0]))
        with self.assertRaises(NotImplementedError):
            path = "tests/datasets/other"
            ds.data.tensor.load_dataset(2, path)

    def test_shuffle(self):
        x_train_tensor = ds.random_tensors("np", (2, 2, 2, 2, 2, 2))
        x_train_tensor_2 = ds.data.tensor.shuffle(x_train_tensor)
        self.assertTrue(x_train_tensor_2.shape == (2, 2) ==
                        (len(x_train_tensor_2.tensors),
                         len(x_train_tensor_2.tensors[0])))
        x_train_tensor_2 = np.array(x_train_tensor_2.collect())
        self.assertFalse((x_train_tensor_2 == np.array(
            x_train_tensor.collect())).all())
        self.assertTrue(x_train_tensor_2.shape == (2, 2, 2, 2, 2, 2))
        x_train_tensor = ds.random_tensors("np", (1, 3, 2, 2, 2, 2))
        x_train_tensor_2 = ds.data.tensor.shuffle(x_train_tensor)
        self.assertTrue(x_train_tensor_2.shape == (1, 3) ==
                        (len(x_train_tensor_2.tensors),
                         len(x_train_tensor_2.tensors[0])))
        x_train_tensor_2 = np.array(x_train_tensor_2.collect())
        self.assertFalse((x_train_tensor_2 == np.array(
            x_train_tensor.collect())).all())
        self.assertTrue(x_train_tensor_2.shape == (1, 3, 2, 2, 2, 2))
        x_train_tensor = ds.random_tensors("np", (2, 2, 16, 8, 11, 11))
        y_train_tensor = ds.random_tensors("np", (2, 2, 16, 1))
        x_train_tensor_2, y_train_tensor_2 = \
            ds.data.tensor.shuffle(x_train_tensor, y_train_tensor)
        x_train_tensor_2 = np.array(x_train_tensor_2.collect())
        y_train_tensor_2 = np.array(y_train_tensor_2.collect())
        self.assertFalse(np.all(x_train_tensor_2 == np.array(
            x_train_tensor.collect())))
        self.assertFalse(np.all(y_train_tensor_2 == np.array(
            y_train_tensor.collect())))
        x_train_tensor = ds.random_tensors("torch", (2, 2, 2, 2, 2, 2))
        x_train_tensor_2 = ds.data.tensor.shuffle(x_train_tensor)
        self.assertTrue(
            x_train_tensor_2.shape == (2, 2) ==
            (len(x_train_tensor_2.tensors),
             len(x_train_tensor_2.tensors[0])))
        x_train_tensor = ds.random_tensors("torch", (2, 2, 2, 2, 2, 2))
        x_train_tensor_2 = ds.data.tensor.shuffle(x_train_tensor)
        x_train_tensor_2 = np.array(x_train_tensor_2.collect())
        self.assertFalse(np.all((x_train_tensor_2 ==
                                 np.array(x_train_tensor.collect()))))
        x_train_tensor = ds.random_tensors("torch", (2, 2, 16, 8, 11, 11))
        y_train_tensor = ds.random_tensors("torch", (2, 2, 16, 1))
        x_train_tensor_2, y_train_tensor_2 = \
            ds.data.tensor.shuffle(x_train_tensor, y_train_tensor)
        x_train_tensor_2 = np.array(x_train_tensor_2.collect())
        y_train_tensor_2 = np.array(y_train_tensor_2.collect())
        self.assertFalse(np.all(x_train_tensor_2 ==
                                np.array(x_train_tensor.collect())))
        self.assertFalse(np.all(y_train_tensor_2 ==
                                np.array(y_train_tensor.collect())))
        x_train_tensor = ds.random_tensors("torch", (1, 3, 16, 8, 11, 11))
        y_train_tensor = ds.random_tensors("torch", (1, 3, 16, 1))
        x_train_tensor_2, y_train_tensor_2 = \
            ds.data.tensor.shuffle(x_train_tensor, y_train_tensor)
        x_train_tensor_2 = np.array(x_train_tensor_2.collect())
        y_train_tensor_2 = np.array(y_train_tensor_2.collect())
        self.assertFalse(np.all(x_train_tensor_2 ==
                                np.array(x_train_tensor.collect())))
        self.assertFalse(np.all(y_train_tensor_2 ==
                                np.array(y_train_tensor.collect())))

    def test_apply_to_tensors(self):
        x_train_tensor = ds.random_tensors("np", (2, 2, 1, 2, 3))
        x_train_tensor_2 = x_train_tensor.apply_to_tensors(np.transpose)
        x_train_tensor_2 = x_train_tensor_2.collect()
        self.assertTrue(x_train_tensor_2[0][0].shape == (3, 2, 1))
        self.assertTrue(x_train_tensor_2[0][1].shape == (3, 2, 1))
        self.assertTrue(x_train_tensor_2[1][0].shape == (3, 2, 1))
        self.assertTrue(x_train_tensor_2[1][1].shape == (3, 2, 1))

        self.assertTrue(x_train_tensor_2[0][0].all() ==
                        compss_wait_on(x_train_tensor.tensors[0][0]).all())
        self.assertTrue(x_train_tensor_2[0][1].all() ==
                        compss_wait_on(x_train_tensor.tensors[0][1]).all())
        self.assertTrue(x_train_tensor_2[1][0].all() ==
                        compss_wait_on(x_train_tensor.tensors[1][0]).all())
        self.assertTrue(x_train_tensor_2[1][1].all() ==
                        compss_wait_on(x_train_tensor.tensors[1][1]).all())
