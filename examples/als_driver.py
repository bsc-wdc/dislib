import argparse
import os
from math import ceil
from time import time

import pandas as pd
from scipy.sparse import csr_matrix

from dislib.data import load_data
from dislib.recommendation import ALS


def load_movielens(data_path, file, delimiter=',', train_ratio=0.9,
                   num_subsets=13):
    print("Loading movielens debug dataset.")
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']

    # 30 users, 100 movies
    df = pd.read_csv(os.path.join(data_path, file),
                     delimiter=delimiter,
                     names=cols,
                     usecols=cols[0:3]).sample(frac=1, random_state=666)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_subsets", type=int, default=48)
    parser.add_argument("--num_factors", type=int, default=100)
    parser.add_argument("--data_path", type=str, default='../tests/files/')

    args = parser.parse_args()

    num_subsets = args.num_subsets

    data_path = args.data_path
    n_f = args.num_factors
    file = 'sample_movielens_ratings.csv'

    train_ds, test = load_movielens(data_path, file)

    exec_start = time()
    als = ALS(tol=0.0001, n_f=n_f, verbose=True)

    als.fit(train_ds)
    # als.fit(train_ds, test)
    exec_end = time()

    print("Ratings for user 0:\n%s" % als.predict_user(0))

    print("Execution time: %.2f" % (exec_end - exec_start))
