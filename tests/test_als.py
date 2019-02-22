import unittest
from math import ceil

import pandas as pd
from scipy.sparse import csr_matrix

from dislib.data import load_data
from dislib.recommendation import ALS


def load_movielens(train_ratio=0.9, num_subsets=4):
    file = 'tests/files/csv/sample_movielens_ratings.csv'

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
        lambda_ = 0.01
        convergence_threshold = 0.1
        max_iter = 10
        verbose = True
        merge_arity = 12

        als = ALS(seed=seed, n_f=n_f, lambda_=lambda_,
                  tol=convergence_threshold,
                  max_iter=max_iter, verbose=verbose, merge_arity=merge_arity)

        self.assertEqual(als._seed, seed)
        self.assertEqual(als._n_f, n_f)
        self.assertEqual(als._lambda, lambda_)
        self.assertEqual(als._tol, convergence_threshold)
        self.assertEqual(als._max_iter, max_iter)
        self.assertEqual(als._verbose, verbose)
        self.assertEqual(als._merge_arity, merge_arity)

    def test_fit_wo_test(self):
        train, test = load_movielens()

        als = ALS(tol=0.01, seed=666, n_f=100,
                  verbose=False)

        als.fit(train)

        self.assertTrue(als.converged)

        als.fit(train, test)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
