import unittest

import numpy as np
from scipy.sparse import csr_matrix

import dislib as ds
from dislib.recommendation import ALS
import dislib.data.util.model as utilmodel
from tests import BaseTimedTestCase


def load_movielens(train_ratio=0.9):
    file = 'tests/datasets/sample_movielens_ratings.csv'

    # 'user_id', 'movie_id', 'rating', 'timestamp'

    data = np.genfromtxt(file, dtype='int', delimiter=',', usecols=range(3))

    # just in case there are movies/user without rating
    # movie_id
    n_m = max(len(np.unique(data[:, 1])), max(data[:, 1]) + 1)
    # user_id
    n_u = max(len(np.unique(data[:, 0])), max(data[:, 0]) + 1)

    idx = int(data.shape[0] * train_ratio)

    train_data = data[:idx]
    test_data = data[idx:]

    train = csr_matrix(
        (train_data[:, 2], (train_data[:, 0], train_data[:, 1])),
        shape=(n_u, n_m))

    test = csr_matrix(
        (test_data[:, 2], (test_data[:, 0], test_data[:, 1])))

    x_size, y_size = train.shape[0] // 4, train.shape[1] // 4
    train_arr = ds.array(train, block_size=(x_size, y_size))

    x_size, y_size = test.shape[0] // 4, test.shape[1] // 4
    test_arr = ds.array(test, block_size=(x_size, y_size))

    return train_arr, test_arr


class ALSTest(BaseTimedTestCase):
    def test_init_params(self):
        # Test all parameters
        seed = 666
        n_f = 100
        lambda_ = 0.001
        convergence_threshold = 0.1
        max_iter = 10
        verbose = True
        arity = 12

        als = ALS(random_state=seed, n_f=n_f, lambda_=lambda_,
                  tol=convergence_threshold,
                  max_iter=max_iter, verbose=verbose, arity=arity)

        self.assertEqual(als.random_state, seed)
        self.assertEqual(als.n_f, n_f)
        self.assertEqual(als.lambda_, lambda_)
        self.assertEqual(als.tol, convergence_threshold)
        self.assertEqual(als.max_iter, max_iter)
        self.assertEqual(als.verbose, verbose)
        self.assertEqual(als.arity, arity)

    def test_fit(self):
        train, test = load_movielens()

        als = ALS(tol=0.01, random_state=666, n_f=100, verbose=False,
                  check_convergence=True)

        als.fit(train, test)
        self.assertTrue(als.converged)

        als.fit(train)

        self.assertTrue(als.converged)

    def test_predict(self):
        data = np.array([[0, 0, 5], [3, 0, 5], [3, 1, 2]])
        ratings = csr_matrix(data)
        train = ds.array(x=ratings, block_size=(1, 1))
        als = ALS(tol=0.01, random_state=666, n_f=5, verbose=False)
        als.fit(train)
        predictions = als.predict_user(user_id=0)

        # Check that the ratings for user 0 are similar to user 1 because they
        # share preferences (third movie), thus it is expected that user 0
        # will rate movie 1 similarly to user 1.
        self.assertTrue(2.75 < predictions[0] < 3.25 and
                        predictions[1] < 1 and
                        predictions[2] > 4.5)

    def test_save_load(self):
        """ Tests that the save and load methods work properly with the three
        expected formats and that an exception is raised when a non-supported
        format is provided.
        """
        data = np.array([[0, 0, 5], [3, 0, 5], [3, 1, 2]])
        ratings = csr_matrix(data)
        train = ds.array(x=ratings, block_size=(1, 1))
        als = ALS(tol=0.01, random_state=666, n_f=5, verbose=False)
        als.fit(train)
        als.save_model("model_als")
        als2 = ALS()
        als2.load_model("model_als")
        predictions2 = als2.predict_user(user_id=0)
        self.assertTrue(2.75 < predictions2[0] < 3.25 and
                        predictions2[1] < 1 and
                        predictions2[2] > 4.5)

        als.save_model("model_als", save_format="cbor")
        als2 = ALS()
        als2.load_model("model_als", load_format="cbor")
        predictions2 = als2.predict_user(user_id=0)
        self.assertTrue(2.75 < predictions2[0] < 3.25 and
                        predictions2[1] < 1 and
                        predictions2[2] > 4.5)

        als.save_model("model_als", save_format="pickle")
        als2 = ALS()
        als2.load_model("model_als", load_format="pickle")
        predictions2 = als2.predict_user(user_id=0)
        self.assertTrue(2.75 < predictions2[0] < 3.25 and
                        predictions2[1] < 1 and
                        predictions2[2] > 4.5)
        with self.assertRaises(ValueError):
            data = np.array([[0, 0, 5], [3, 0, 5], [3, 1, 2]])
            ratings = csr_matrix(data)
            train = ds.array(x=ratings, block_size=(1, 1))
            als = ALS(tol=0.01, random_state=666, n_f=5, verbose=False)
            als.fit(train)
            als.save_model("model_als", save_format="txt")
        with self.assertRaises(ValueError):
            als2 = ALS()
            als2.load_model("model_als", load_format="txt")

        als = ALS(tol=0.01, random_state=666, n_f=5, verbose=False)

        data = np.array([[15, 15, 15], [15, 15, 15], [15, 15, 15]])
        ratings = csr_matrix(data)
        train = ds.array(x=ratings, block_size=(1, 1))

        als.fit(train)
        als.save_model("./model_als", overwrite=False)

        als2 = ALS()
        als2.load_model("./model_als", load_format="pickle")
        predictions2 = als2.predict_user(user_id=0)
        self.assertTrue(2.75 < predictions2[0] < 3.25 and
                        predictions2[1] < 1 and
                        predictions2[2] > 4.5)

        cbor2_module = utilmodel.cbor2
        utilmodel.cbor2 = None
        with self.assertRaises(ModuleNotFoundError):
            als.save_model("./model_als", save_format="cbor")
        with self.assertRaises(ModuleNotFoundError):
            als2.load_model("./model_als", load_format="cbor")
        utilmodel.cbor2 = cbor2_module


def main():
    unittest.main()


if __name__ == '__main__':
    main()
