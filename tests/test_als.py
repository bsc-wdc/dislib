import unittest
from math import ceil

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from dislib.data import load_data
from dislib.recommendation import ALS


def load_movielens(train_ratio=0.9, num_subsets=4):
    file = 'tests/files/sample_movielens_ratings.csv'

    cols = ['user_id', 'movie_id', 'rating', 'timestamp']

    # 30 users, 100 movies
    df = pd.read_csv(file, names=cols, usecols=cols[0:3])

    # just in case there are movies/user without rating
    n_m = max(df.movie_id.nunique(), max(df.movie_id) + 1)
    n_u = max(df.user_id.nunique(), max(df.user_id) + 1)

    idx = int(df.shape[0] * train_ratio)

    train_df = df.iloc[:idx]
    test_df = df.iloc[idx:]

    train = csr_matrix(
        (train_df.rating, (train_df.user_id, train_df.movie_id)),
        shape=(n_u, n_m)).transpose().tocsr()
    test = csr_matrix(
        (test_df.rating, (test_df.user_id, test_df.movie_id)))

    dataset = load_data(train, int(ceil(train.shape[0] / num_subsets)))

    return dataset, test


class ALSTest(unittest.TestCase):
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

        self.assertEqual(als._seed, seed)
        self.assertEqual(als._n_f, n_f)
        self.assertEqual(als._lambda, lambda_)
        self.assertEqual(als._tol, convergence_threshold)
        self.assertEqual(als._max_iter, max_iter)
        self.assertEqual(als._verbose, verbose)
        self.assertEqual(als._arity, arity)

    def test_fit(self):
        train, test = load_movielens()

        als = ALS(tol=0.01, random_state=666, n_f=100,
                  verbose=False)

        als.fit(train, test)
        self.assertTrue(als.converged)

        als.fit(train)
        self.assertTrue(als.converged)

    def test_predict(self):
        data = np.array([[0, 0, 5], [3, 0, 5], [3, 1, 2]])
        ratings = csr_matrix(data).transpose().tocsr()
        train = load_data(x=ratings, subset_size=1)
        als = ALS(tol=0.01, random_state=666, n_f=100,
                  verbose=False)
        als.fit(train)
        predictions = als.predict_user(user_id=0)

        # Check that the ratings for user 0 are similar to user 1 because they
        # share preferences (third movie), thus it is expected that user 0
        # will rate movie 1 similarly to user 1.
        self.assertTrue(2.75 < predictions[0] < 3.25 and
                        predictions[1] < 1 and
                        predictions[2] > 4.5)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
