import argparse
import os
from math import ceil
from time import time

import pandas as pd
from scipy.sparse import csr_matrix

from dislib.data import load_data
from dislib.data import load_libsvm_file
from dislib.recommendation import ALS


def load_movielens(data_path, file, delimiter=',', train_ratio=0.8,
                   num_subsets=4):
    print("Loading movielens debug dataset.")
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']

    # 30 users, 100 movies
    df = pd.read_csv(os.path.join(data_path, file),
                     delimiter=delimiter,
                     names=cols,
                     usecols=cols[0:3]).sample(frac=1, random_state=666)

    valid_df = pd.read_csv(os.path.join(data_path, 'test.data'),
                           names=cols[:3])

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

    valid = csr_matrix((valid_df.rating,
                        (valid_df.user_id, valid_df.movie_id)))

    dataset = load_data(train, int(ceil(train.shape[0] // num_subsets)))

    return dataset, test, valid


def load_movielens_20m(data_path, file, delimiter=',', train_ratio=0.9,
                       num_subsets=4):
    print("Loading movielens debug dataset.")
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']

    # 30 users, 100 movies
    df = pd.read_csv(os.path.join(data_path, file),
                     delimiter=delimiter,
                     names=cols,
                     usecols=cols[0:3],
                     header=0).sample(frac=1, random_state=666)

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

    dataset = load_data(train, int(ceil(train.shape[0] // num_subsets)))

    return dataset, test, None


def load_netflix(data_path, debug, train_ratio=0.8, num_subsets=48):
    print("Loading netflix dataset [debug:%s]." % debug)
    cols = ['user_id', 'movie_id', 'rating']
    if debug:
        df = pd.read_csv(os.path.join(data_path, 'sample_netflix_data.csv'),
                         names=cols)
    else:
        df = pd.read_csv(os.path.join(data_path, 'netflix_data.csv'),
                         names=cols)

    # just in case there are movies/user without rating
    n_m = max(df.movie_id.nunique(), max(df.movie_id) + 1)
    n_u = max(df.user_id.nunique(), max(df.user_id) + 1)

    idx = int(df.shape[0] * train_ratio)

    train_df = df.iloc[:idx]
    test_df = df.iloc[idx:]

    train = csr_matrix(
        (train_df.rating, (train_df.user_id, train_df.movie_id)),
        shape=(n_u, n_m))
    test = csr_matrix(
        (test_df.rating, (test_df.user_id, test_df.movie_id)))

    print("Train and tests matrices sizes: %s , %s" % (
        train.data.nbytes, test.data.nbytes))
    dataset = load_data(train, int(ceil(train.shape[0] // num_subsets)))

    return dataset, test, None


def load_netflix_libsvm(data_path, subset_size=100):
    dataset = load_libsvm_file(
        os.path.join(data_path, 'netflix_data_libsvm.txt'),
        subset_size=subset_size, n_features=480189)

    return dataset, None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--num_subsets", type=int, default=48)
    parser.add_argument("--num_factors", type=int, default=100)
    parser.add_argument("--data_path", type=str, default='./data/')
    parser.add_argument("--example", type=int, choices=range(1, 4), default=1,
                        help='Choose execution: 1->movielens for debug; '
                             '2->movielens ml-20, 3->movielesn ml-latest, 4->netflix for debug, 5->netflix full size.')

    args = parser.parse_args()

    num_subsets = args.num_subsets
    # debug = args.debug
    data_path = args.data_path
    example = args.example
    n_f = args.num_factors

    if example == 1:
        file = 'sample_movielens_ratings.txt'
        train_ds, test, valid = load_movielens(data_path, file, delimiter='::')
    elif example == 2:
        file = 'ml-20m/ratings.csv'
        train_ds, test, valid = load_movielens_20m(data_path, file,
                                                   num_subsets=num_subsets)
    elif example == 3:
        file = 'ml-latest/ratings.csv'
        train_ds, test, valid = load_movielens_20m(data_path, file,
                                                   num_subsets=num_subsets)
    else:
        train_ds, test, valid = load_netflix_libsvm(data_path=data_path)
        # train_ds, test, valid = load_netflix(num_subsets=num_subsets,
        #                                      debug=example == 2,
        #                                      data_path=data_path)

    exec_start = time()
    als = ALS(convergence_threshold=0.0001, max_iter=5, seed=666, n_f=n_f,
              verbose=True)

    als.fit(train_ds, test)
    # als.fit(train_ds, test)
    exec_end = time()

    print("Prediction user 0:\n%s" % als.predict_user(0))

    print("Execution time: %.2f" % (exec_end - exec_start))
    # cx = sparse.find(valid)
    # preds = []
    # for i, j, r in zip(cx[0], cx[1], cx[2]):
    #     pred = als.predict_user(i)[j]
    #     preds.append(pred)
    #     print("Rating vs prediction: %.1f - %1.f" % (r, pred))
