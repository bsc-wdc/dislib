import argparse

import pandas as pd
from scipy.sparse import csr_matrix

from dislib.data import load_data
from dislib.recommendation import ALS


def load_movielens(train_ratio=0.8, num_subsets=4):
    print("Loading movielens debug dataset.")
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_csv('./data/sample_movielens_ratings.txt',
                     delimiter='::',
                     names=cols,
                     usecols=cols[0:3]).sample(frac=1, random_state=666)
    valid_df = pd.read_csv('./data/test.data', names=cols[:3])

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

    valid = csr_matrix((valid_df.rating,
                        (valid_df.user_id, valid_df.movie_id)))

    dataset = load_data(train, train.shape[0] // num_subsets)

    return dataset, test, valid


def load_netflix(train_ratio=0.8, num_subsets=48):
    print("Loading netflix dataset.")
    cols = ['user_id', 'movie_id', 'rating']
    df = pd.read_csv('./sample_merged_data.csv', names=cols)

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

    dataset = load_data(train, train.shape[0] // num_subsets)

    return dataset, test, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--num_subsets", type=int, default=48)

    args = parser.parse_args()

    num_subsets = args.num_subsets
    debug = args.debug

    if debug:
        train_ds, test, valid = load_movielens()

    else:
        train_ds, test, valid = load_netflix(num_subsets)

    als = ALS(convergence_threshold=0.0001, max_iter=5, seed=666, n_f=50,
              verbose=True)

    als.fit(train_ds, test)

    print("Prediction user 0:\n%s" % als.predict_user(0))

    # cx = sparse.find(valid)
    # preds = []
    # for i, j, r in zip(cx[0], cx[1], cx[2]):
    #     pred = als.predict_user(i)[j]
    #     preds.append(pred)
    #     print("Rating vs prediction: %.1f - %1.f" % (r, pred))
