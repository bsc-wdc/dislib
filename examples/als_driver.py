import argparse
import os
from math import ceil
from time import time

import pandas as pd
from scipy.sparse import csr_matrix

import dislib as ds
from dislib.recommendation import ALS


def load_movielens(data_path, train_ratio=0.9):
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    file = 'sample_movielens_ratings.csv'

    # 30 users, 100 movies
    df = pd.read_csv(os.path.join(data_path, file),
                     delimiter=',',
                     names=cols,
                     usecols=cols[0:3]).sample(frac=1, random_state=666)

    # just in case there are movies/user without rating
    n_m = max(df.movie_id.nunique(), max(df.movie_id) + 1)
    n_u = max(df.user_id.nunique(), max(df.user_id) + 1)

    idx = int(df.shape[0] * train_ratio)

    tr_df = df.iloc[:idx]
    te_df = df.iloc[idx:]

    train = csr_matrix((tr_df.rating, (tr_df.user_id, tr_df.movie_id)),
                       shape=(n_u, n_m))
    test = csr_matrix(
        (te_df.rating, (te_df.user_id, te_df.movie_id)))

    x_size, y_size = ceil(train.shape[0] / 2), ceil(train.shape[1] / 3)
    train_arr = ds.array(train, block_size=(x_size, y_size))

    x_size, y_size = ceil(test.shape[0] / 2), ceil(test.shape[1] / 3)
    test_arr = ds.array(test, block_size=(x_size, y_size))

    return train_arr, test_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_subsets", type=int, default=48)
    parser.add_argument("--num_factors", type=int, default=100)
    parser.add_argument("--data_path", type=str, default='./tests/files/')

    args = parser.parse_args()

    num_subsets = args.num_subsets

    data_path = args.data_path
    n_f = args.num_factors

    train, test = load_movielens(data_path=data_path)

    exec_start = time()
    als = ALS(tol=0.0001, n_f=n_f, max_iter=2, verbose=True)

    # Fit using training data to check convergence
    # als.fit(train)

    # Fit using test data to check convergence
    als.fit(train, test)

    exec_end = time()

    print("Ratings for user 0:\n%s" % als.predict_user(0))

    print("Execution time: %.2f" % (exec_end - exec_start))
